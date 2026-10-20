use super::*;
use rayon::prelude::*;
use std::{
    hash::{BuildHasher, BuildHasherDefault},
    ops::Deref,
    sync::OnceLock,
};

/// A non-empty set of valid, distinct individuals, ranked by their score.
///
/// A population is never empty, and it holds only valid and distinct individuals, sorted by
/// their score from worst to best according to its [`objective`](Population::objective).
/// This is ascending score order for maximization and descending for minimization.
/// Individuals are distinct according to their [`PartialEq`] implementation.
///
/// A population holds only the individuals of one generation. The history of a run, such as
/// its generation number and the fittest individual of any generation, is kept by the
/// [`RunState`] that holds the population.
pub struct Population<I: Individual> {
    individuals: Vec<I>,
    stats: OnceLock<PopulatedStats>,
    objective: Objective,
}

impl<I: Individual> Population<I> {
    /// Creates a new population that maximizes the given individuals' scores.
    ///
    /// Invalid individuals (see [`Individual::is_valid`]) and duplicates are discarded, and the
    /// remaining individuals are sorted by their score.
    /// Deduplication uses [`PartialEq`] across all scores, comparing only individuals with equal
    /// [`Hash`](std::hash::Hash) values (see [`Individual`]).
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if no valid individual is given.
    pub fn new(individuals: Vec<I>) -> Result<Self, EvolutionError> {
        Self::with_objective(individuals, Objective::Maximize)
    }

    /// Creates a population with an explicit optimization direction.
    ///
    /// Scores and statistics retain the model's original units. Individuals are sorted from
    /// worst to best and deduplicated as in [`new`](Self::new). When equal solutions have
    /// different cached scores, the better evaluation is retained. The objective remains fixed
    /// when the population is [replaced](Self::replace) or [advanced](Self::advance).
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if no valid individual is given.
    pub fn with_objective(
        mut individuals: Vec<I>,
        objective: Objective,
    ) -> Result<Self, EvolutionError> {
        normalize(&mut individuals, objective)?;
        Ok(Self {
            individuals,
            stats: OnceLock::new(),
            objective,
        })
    }

    /// Replaces the individuals of the population with those of `offspring`.
    ///
    /// The individuals in `offspring` are normalized the same way as in [`Population::new`].
    /// Upon success, `offspring` contains the individuals that the population held before.
    ///
    /// A [`Generation`] already holds only valid and distinct individuals, so
    /// [`advance`](Population::advance) replaces the population with one without normalizing it
    /// again.
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if `offspring` contains no valid individual.
    /// In that case, the population is left unchanged and `offspring` is left empty.
    pub fn replace(&mut self, offspring: &mut Vec<I>) -> Result<(), EvolutionError> {
        normalize(offspring, self.objective)?;
        std::mem::swap(&mut self.individuals, offspring);
        self.stats = OnceLock::new();
        Ok(())
    }

    /// Replaces the individuals of the population with those of `generation`, which it leaves
    /// empty for the generation after.
    ///
    /// The individuals are sorted as in [`Population::new`].
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if `generation` is empty. In that case, the
    /// population is left unchanged.
    pub fn advance(&mut self, generation: &mut Generation<I>) -> Result<(), EvolutionError> {
        if generation.is_empty() {
            return Err(EvolutionError::EmptyPopulation);
        }
        generation.move_into(&mut self.individuals);
        sort(&mut self.individuals, self.objective);
        self.stats = OnceLock::new();
        Ok(())
    }

    /// Returns the fixed optimization direction by which the population is ranked.
    #[inline]
    pub fn objective(&self) -> Objective {
        self.objective
    }

    /// Summarizes the scores of the population in their original units.
    ///
    /// Minimum and maximum mean the numerical extrema, independently of the objective; use
    /// [`fittest`](Self::fittest) for the best individual under that objective.
    ///
    /// The statistics are computed when they are first requested, so runs that do not use them
    /// do not pay for them.
    #[inline]
    pub fn stats(&self) -> &PopulatedStats {
        self.stats.get_or_init(|| {
            PopulatedStats::of_sorted(&self.individuals, I::score)
                .expect("a population is never empty")
        })
    }

    /// Returns the fittest individual in the population.
    #[inline]
    pub fn fittest(&self) -> &I {
        &self.individuals[self.individuals.len() - 1]
    }

    /// Returns the fittest individual in the population while consuming the population.
    #[inline]
    pub fn take_fittest(mut self) -> I {
        self.individuals.pop().expect("a population is never empty")
    }
}

impl<I: Individual> Deref for Population<I> {
    type Target = [I];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.individuals
    }
}

impl<I: Individual> TryFrom<Vec<I>> for Population<I> {
    type Error = EvolutionError;

    #[inline]
    fn try_from(individuals: Vec<I>) -> Result<Self, Self::Error> {
        Self::new(individuals)
    }
}

impl<I: Individual> From<Population<I>> for Vec<I> {
    #[inline]
    fn from(population: Population<I>) -> Self {
        population.individuals
    }
}

/// Removes invalid and duplicate individuals and sorts the rest by score.
///
/// # Errors
/// Returns [`EvolutionError::EmptyPopulation`] if no individual remains.
fn normalize<I: Individual>(
    individuals: &mut Vec<I>,
    objective: Objective,
) -> Result<(), EvolutionError> {
    individuals.retain(Individual::is_valid);
    sort(individuals, objective);
    remove_duplicates(individuals);
    if individuals.is_empty() {
        Err(EvolutionError::EmptyPopulation)
    } else {
        Ok(())
    }
}

/// Sorts individuals by score from worst to best under `objective`.
fn sort<I: Individual>(individuals: &mut [I], objective: Objective) {
    // A stable sort keeps individuals with equal score in their given order, which keeps runs
    // reproducible.
    individuals.par_sort_by(|a, b| objective.cmp_fitness(a.score(), b.score()));
}

/// Removes duplicate individuals from a vector, keeping the order of the remaining ones and the
/// last of each group of equal individuals. In a vector sorted by score, equal solutions
/// therefore retain their best evaluation.
///
/// Equal individuals have equal hashes, so only individuals with equal hashes are compared, from
/// last to first. The individuals are hashed in parallel.
pub(crate) fn remove_duplicates<I: Individual>(individuals: &mut Vec<I>) {
    let mut order = individuals
        .par_iter()
        .enumerate()
        .map(|(i, individual)| (hash_of(individual), i))
        .collect::<Vec<_>>();
    order.sort_unstable();

    let mut keep = vec![false; individuals.len()];
    let mut kept = Vec::new();
    for group in order.chunk_by(|a, b| a.0 == b.0) {
        kept.clear();
        for &(_, i) in group.iter().rev() {
            if !kept.contains(&&individuals[i]) {
                kept.push(&individuals[i]);
                keep[i] = true;
            }
        }
    }

    let mut keep = keep.into_iter();
    individuals.retain(|_| keep.next() == Some(true));
}

/// Returns the hash by which equal individuals are found, since they have equal hashes.
#[inline]
pub(crate) fn hash_of<I: Individual>(individual: &I) -> u64 {
    BuildHasherDefault::<GeneHasher>::default().hash_one(individual)
}

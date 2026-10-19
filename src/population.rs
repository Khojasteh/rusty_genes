use super::*;
use rayon::prelude::*;
use std::{
    hash::{BuildHasher, BuildHasherDefault},
    ops::Deref,
};

/// Represents a population of individuals in an evolutionary algorithm.
///
/// # Note
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
    stats: PopulatedStats,
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
    /// when the population is [`replaced`](Self::replace).
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if no valid individual is given.
    pub fn with_objective(
        mut individuals: Vec<I>,
        objective: Objective,
    ) -> Result<Self, EvolutionError> {
        let stats = normalize(&mut individuals, objective)?;
        Ok(Self {
            individuals,
            stats,
            objective,
        })
    }

    /// Replaces the individuals of the population with those of `offspring`.
    ///
    /// The individuals in `offspring` are normalized the same way as in [`Population::new`].
    /// Upon success, `offspring` contains the individuals that the population held before.
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if `offspring` contains no valid individual.
    /// In that case, the population is left unchanged and `offspring` is left empty.
    pub fn replace(&mut self, offspring: &mut Vec<I>) -> Result<(), EvolutionError> {
        let stats = normalize(offspring, self.objective)?;
        std::mem::swap(&mut self.individuals, offspring);
        self.stats = stats;
        Ok(())
    }

    /// Returns the fixed optimization direction by which the population is ranked.
    #[inline]
    pub fn objective(&self) -> Objective {
        self.objective
    }

    /// Summarizes the current generation's scores in their original units.
    ///
    /// Minimum and maximum mean the numerical extrema, independently of the objective; use
    /// [`fittest`](Self::fittest) for the best individual under that objective.
    #[inline]
    pub fn stats(&self) -> &PopulatedStats {
        &self.stats
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

/// Removes invalid and duplicate individuals, sorts the rest by score, and collects the
/// statistics of their scores.
fn normalize<I: Individual>(
    individuals: &mut Vec<I>,
    objective: Objective,
) -> Result<PopulatedStats, EvolutionError> {
    individuals.retain(Individual::is_valid);
    // A stable sort keeps individuals with equal score in their given order, which keeps runs
    // reproducible.
    individuals.par_sort_by(|a, b| objective.cmp_fitness(a.score(), b.score()));
    remove_duplicates(individuals);
    individuals
        .iter()
        .map(Individual::score)
        .collect::<Stats>()
        .snapshot()
        .ok_or(EvolutionError::EmptyPopulation)
}

/// Removes duplicate individuals from a vector that is sorted by score, keeping the order of the
/// remaining ones. Equal solutions retain their best evaluation.
///
/// Equal individuals have equal hashes, so only individuals with equal hashes are compared, from
/// best to worst.
fn remove_duplicates<I: Individual>(individuals: &mut Vec<I>) {
    let hasher = BuildHasherDefault::<GeneHasher>::default();
    let mut order = individuals
        .iter()
        .enumerate()
        .map(|(i, individual)| (hasher.hash_one(individual), i))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        hash::{Hash, Hasher},
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    #[derive(Debug, Clone, PartialEq)]
    struct Mock {
        id: u32,
        score: f64,
    }

    impl Hash for Mock {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.id.hash(state);
        }
    }

    impl Individual for Mock {
        fn score(&self) -> f64 {
            self.score
        }
    }

    fn mock(id: u32, score: f64) -> Mock {
        Mock { id, score }
    }

    /// An individual whose equality ignores its score and counts the comparisons. It hashes its
    /// key, or nothing if it has no key, so that its hash can collide with other identities.
    #[derive(Debug, Clone)]
    struct Counted {
        id: u64,
        score: f64,
        key: Option<u64>,
        comparisons: Arc<AtomicUsize>,
    }

    impl PartialEq for Counted {
        fn eq(&self, other: &Self) -> bool {
            self.comparisons.fetch_add(1, Ordering::Relaxed);
            self.id == other.id
        }
    }

    impl Hash for Counted {
        fn hash<H: Hasher>(&self, state: &mut H) {
            if let Some(key) = self.key {
                key.hash(state);
            }
        }
    }

    impl Individual for Counted {
        fn score(&self) -> f64 {
            self.score
        }
    }

    fn counted_individuals(
        individuals: &[(u64, f64)],
        key: fn(u64) -> Option<u64>,
        comparisons: &Arc<AtomicUsize>,
    ) -> Vec<Counted> {
        individuals
            .iter()
            .map(|&(id, score)| Counted {
                id,
                score,
                key: key(id),
                comparisons: comparisons.clone(),
            })
            .collect()
    }

    #[test]
    fn deduplication_compares_only_individuals_with_equal_hashes() {
        let comparisons = Arc::new(AtomicUsize::new(0));
        let individuals = (0..100).map(|id| (id, id as f64)).collect::<Vec<_>>();
        let population =
            Population::new(counted_individuals(&individuals, Some, &comparisons)).unwrap();

        assert_eq!(population.len(), 100);
        assert_eq!(comparisons.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn deduplication_keeps_the_best_evaluation_with_colliding_or_constant_hashes() {
        let individuals = [(1, 1.0), (2, 2.0), (1, 3.0), (3, 2.0), (2, 0.0)];
        let keys: [fn(u64) -> Option<u64>; 3] =
            [|id| Some(id % 2), |_| None, |id| (id != 3).then_some(id)];
        for key in keys {
            let comparisons = Arc::new(AtomicUsize::new(0));
            let population =
                Population::new(counted_individuals(&individuals, key, &comparisons)).unwrap();

            let kept = population
                .iter()
                .map(|individual| (individual.id, individual.score))
                .collect::<Vec<_>>();
            assert_eq!(kept, vec![(2, 2.0), (3, 2.0), (1, 3.0)]);
        }
    }

    #[test]
    fn new_removes_non_adjacent_duplicates_with_equal_score() {
        let individuals = (0..300).map(|i| mock(i % 3, 1.0)).collect();
        let population = Population::new(individuals).unwrap();

        let mut ids = population.iter().map(|m| m.id).collect::<Vec<_>>();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn new_keeps_distinct_individuals_with_equal_score() {
        let individuals = vec![mock(1, 2.0), mock(2, 2.0), mock(3, 1.0)];
        let population = Population::new(individuals).unwrap();

        assert_eq!(population.len(), 3);
        assert_eq!(population[0].id, 3);
    }

    #[test]
    fn new_discards_invalid_individuals_and_sorts_by_score() {
        let individuals = vec![
            mock(1, 3.0),
            mock(2, f64::NAN),
            mock(3, -1.0),
            mock(4, f64::INFINITY),
        ];
        let population = Population::new(individuals).unwrap();

        let ids = population.iter().map(|m| m.id).collect::<Vec<_>>();
        assert_eq!(ids, vec![3, 1]);
        assert_eq!(population.fittest().id, 1);
    }

    #[test]
    fn new_fails_without_valid_individual() {
        assert_eq!(
            Population::<Mock>::new(vec![]).err(),
            Some(EvolutionError::EmptyPopulation)
        );
        assert_eq!(
            Population::new(vec![mock(1, f64::NAN)]).err(),
            Some(EvolutionError::EmptyPopulation)
        );
    }

    #[test]
    fn replace_without_valid_offspring_leaves_population_unchanged() {
        let mut population = Population::new(vec![mock(1, 1.0)]).unwrap();
        let mut offspring = vec![mock(2, f64::NAN)];

        let result = population.replace(&mut offspring);

        assert_eq!(result, Err(EvolutionError::EmptyPopulation));
        assert_eq!(population.fittest().id, 1);
        assert!(offspring.is_empty());
    }
}

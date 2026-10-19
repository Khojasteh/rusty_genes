use super::*;
use rayon::prelude::*;

/// Chooses the individuals of the next generation from a population and the candidates bred
/// from it.
///
/// The policy is part of algorithms that may be moved to other threads, so it must be [`Send`]
/// and [`Sync`].
pub trait SurvivalPolicy<I: Individual>: Send + Sync {
    /// Adds the survivors among the individuals of `population` and `candidates` to `next`.
    ///
    /// `candidates` arrive in breeding order and refer to parents by indices in `population`.
    /// Compare their scores under [`Population::objective`]. `next` may already hold elites;
    /// its insertion methods reject invalid or duplicate individuals and enforce its capacity.
    ///
    /// Fill unused places when this policy permits enough valid, distinct survivors.
    /// Unfilled places make the next population smaller; no further breeding is attempted
    /// during this generation to replace them.
    fn select_survivors(
        &self,
        population: &Population<I>,
        candidates: Vec<Candidate<I>>,
        next: &mut Generation<I>,
        rng: &mut Rng,
    );
}

/// The survival policy in which each candidate competes with its fittest parent.
///
/// A candidate survives if it is valid and at least as fit as its
/// [fittest parent](Candidate::fittest_parent) under the population's objective; otherwise that
/// parent survives, once however many of its candidates lose. When these survivors do not fill
/// the next generation, such as when candidates repeat one another, the fittest individuals of
/// the population that it lacks take the remaining places.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FamilyCompetition;

impl<I: Individual> SurvivalPolicy<I> for FamilyCompetition {
    fn select_survivors(
        &self,
        population: &Population<I>,
        candidates: Vec<Candidate<I>>,
        next: &mut Generation<I>,
        _: &mut Rng,
    ) {
        let objective = population.objective();
        let mut members = Members::new(population);
        for candidate in candidates {
            let parent = candidate.fittest_parent();
            let bred = candidate.individual();
            if bred.is_valid()
                && objective.is_at_least_as_fit(bred.score(), population[parent].score())
            {
                next.insert(candidate.into_individual());
            } else {
                members.add(parent, next);
            }
        }
        members.fill(next);
    }
}

/// The survival policy in which candidates compete with their parents for their places in the
/// population.
///
/// Each individual of the population holds a place. A candidate takes the place of its
/// [fittest parent](Candidate::fittest_parent) if it is valid, distinct from every individual of
/// the population and every candidate before it, and at least as fit as the holder of the place
/// under the population's objective: the parent itself, or a candidate that took the place
/// before it. Unlike in [`FamilyCompetition`], an individual therefore keeps its place whether
/// or not it is selected as a parent, until a candidate of its own is at least as fit, and the
/// fittest individual can only be replaced by one at least as fit.
///
/// The holders of the places join the next generation from the fittest down, after the elites
/// it already holds, so when there are more holders than the next generation has room for, the
/// least fit are left out. When the holders do not fill the next generation, the remaining
/// valid and distinct candidates fill it, from the fittest down.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PlaceCompetition;

impl<I: Individual> SurvivalPolicy<I> for PlaceCompetition {
    fn select_survivors(
        &self,
        population: &Population<I>,
        candidates: Vec<Candidate<I>>,
        next: &mut Generation<I>,
        _: &mut Rng,
    ) {
        // The individuals are identified by their positions: the members of the population
        // first, and the candidates after them.
        let members = population.len();
        let individual = |position: usize| {
            if position < members {
                &population[position]
            } else {
                candidates[position - members].individual()
            }
        };
        let hashes = population
            .par_iter()
            .chain(candidates.par_iter().map(Candidate::individual))
            .map(hash_of)
            .collect::<Vec<_>>();

        let objective = population.objective();
        let mut present = HashIndex::default();
        for (member, &hash) in hashes[..members].iter().enumerate() {
            present.insert(hash, member);
        }
        // The holder of each place, which is its member until a candidate takes the place.
        let mut holders = (0..members).collect::<Vec<_>>();
        // The valid and distinct candidates that hold no place.
        let mut spares = Vec::new();
        for (position, candidate) in (members..).zip(&candidates) {
            let hash = hashes[position];
            let bred = candidate.individual();
            if !bred.is_valid()
                || present
                    .find(hash, |other| individual(other) == bred)
                    .is_some()
            {
                continue;
            }
            present.insert(hash, position);

            let place = candidate.fittest_parent();
            let holder = holders[place];
            if objective.is_fitter(individual(holder).score(), bred.score()) {
                spares.push(position);
            } else {
                if holder >= members {
                    spares.push(holder);
                }
                holders[place] = position;
            }
        }

        let fittest_first = |a: &usize, b: &usize| {
            objective.cmp_fitness(individual(*b).score(), individual(*a).score())
        };
        holders.sort_by(fittest_first);
        spares.sort_by(fittest_first);

        let mut candidates = candidates
            .into_iter()
            .map(|candidate| Some(candidate.into_individual()))
            .collect::<Vec<_>>();
        for position in holders.into_iter().chain(spares) {
            if next.is_full() {
                break;
            }
            if position < members {
                next.insert_clone_hashed(&population[position], hashes[position]);
            } else {
                let individual = candidates[position - members]
                    .take()
                    .expect("a candidate holds at most one place");
                next.insert_hashed(individual, hashes[position]);
            }
        }
    }
}

/// The survival policy in which every valid candidate survives, regardless of its score, and the
/// individuals of the population survive only as elites.
///
/// When candidates are invalid or repeat other individuals, the fittest individuals of the
/// population that the next generation lacks take their places.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Generational;

impl<I: Individual> SurvivalPolicy<I> for Generational {
    fn select_survivors(
        &self,
        population: &Population<I>,
        candidates: Vec<Candidate<I>>,
        next: &mut Generation<I>,
        _: &mut Rng,
    ) {
        for candidate in candidates {
            next.insert(candidate.into_individual());
        }
        Members::new(population).fill(next);
    }
}

/// The individuals of a population, which survival policies add to the next generation.
///
/// Each individual is hashed at most once, however often it is added, and cloned only when the
/// next generation lacks it.
struct Members<'a, I: Individual> {
    population: &'a Population<I>,
    tried: Vec<bool>,
}

impl<'a, I: Individual> Members<'a, I> {
    fn new(population: &'a Population<I>) -> Self {
        Self {
            population,
            tried: vec![false; population.len()],
        }
    }

    /// Adds the individual at `index` of the population to `next`, unless `next` holds it or is
    /// full.
    fn add(&mut self, index: usize, next: &mut Generation<I>) {
        // A generation only grows, so once it holds an individual or is full, it stays so, and
        // trying the individual again would fail.
        if std::mem::replace(&mut self.tried[index], true) {
            return;
        }
        let member = &self.population[index];
        next.insert_clone_hashed(member, hash_of(member));
    }

    /// Adds the fittest individuals of the population that `next` lacks until `next` is full.
    fn fill(&mut self, next: &mut Generation<I>) {
        for index in (0..self.population.len()).rev() {
            if next.is_full() {
                break;
            }
            self.add(index, next);
        }
    }
}

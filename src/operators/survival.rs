use super::*;

/// Decides whether a bred offspring passes on to the next generation in place of its fittest
/// parent.
///
/// The policy is shared by the threads that breed offspring in parallel, so it must be [`Send`]
/// and [`Sync`].
pub trait SurvivalPolicy<I: Individual>: Send + Sync {
    /// Returns whether `offspring` survives.
    ///
    /// # Arguments
    /// * `offspring` - The evaluated offspring, which is valid.
    /// * `fittest_parent` - The fittest parent of the offspring.
    /// * `population` - The population the offspring is bred from. Its
    ///   [`objective`](Population::objective) determines whether lower or higher scores are better.
    /// * `rng` - A mutable reference to a random number generator.
    fn survives(
        &self,
        offspring: &I,
        fittest_parent: &I,
        population: &Population<I>,
        rng: &mut Rng,
    ) -> bool;
}

/// The survival policy in which an offspring competes with its fittest parent: it survives if its
/// score is at least as good as the parent's under the population's objective.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FamilyCompetition;

impl<I: Individual> SurvivalPolicy<I> for FamilyCompetition {
    #[inline]
    fn survives(
        &self,
        offspring: &I,
        fittest_parent: &I,
        population: &Population<I>,
        _: &mut Rng,
    ) -> bool {
        !population
            .objective()
            .is_fitter(fittest_parent.score(), offspring.score())
    }
}

/// The survival policy in which every valid offspring survives, regardless of its score.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Generational;

impl<I: Individual> SurvivalPolicy<I> for Generational {
    #[inline]
    fn survives(&self, _: &I, _: &I, _: &Population<I>, _: &mut Rng) -> bool {
        true
    }
}

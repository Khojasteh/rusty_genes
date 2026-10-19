use crate::Rng;

/// Represents an individual in a population of an evolutionary algorithm.
///
/// The model's [`objective`](crate::EvolutionModel::objective) determines whether lower or
/// higher scores are better. The engine evaluates candidates before admitting them to a
/// population, so new or changed genomes may carry an invalid score until then.
///
/// Equality determines whether individuals are duplicates in a [`Population`](crate::Population).
/// It may compare only the solution itself, excluding a cached score.
pub trait Individual: Clone + PartialEq + Send + Sync {
    /// Returns the cached evaluation in the model's own units, without evaluating the genome.
    ///
    /// The score may be negative or outside `[0, 1]`.
    /// It must be finite for a valid individual. The model's [`Objective`](crate::Objective)
    /// determines whether lower or higher scores are better.
    fn score(&self) -> f64;

    /// Returns [`true`] if the individual is a valid solution. Invalid individuals never become
    /// part of a [`Population`](crate::Population).
    ///
    /// # Note
    /// The default implementation considers an individual valid when its score is finite,
    /// i.e. neither infinite nor NaN.
    ///
    /// Implementations of this trait may override this method with a more specific definition of
    /// validity, but an individual whose score is not finite must never be reported as valid.
    #[inline]
    fn is_valid(&self) -> bool {
        self.score().is_finite()
    }
}

/// Represents an individual in a population of an evolutionary algorithm that can be generated
/// randomly.
///
/// The trait is parameterized over a type `A`, which represents domain-specific arguments for
/// generating the individual, such as the length of its genome.
pub trait RandomIndividual<A>: Individual {
    /// Generates a new individual with random characteristics.
    ///
    /// # Arguments
    /// * `args` - A reference to domain-specific arguments for generating the individual.
    /// * `rng` - A mutable reference to random number generator.
    ///
    /// # Returns
    /// A new individual whose characteristics are determined by the random number generator and
    /// the domain-specific arguments.
    fn new_random(args: &A, rng: &mut Rng) -> Self;
}

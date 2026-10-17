use crate::Rng;

/// Represents an individual in a population of an evolutionary algorithm.
///
/// # Note
/// An individual is characterized by its fitness score, which measures how well it performs
/// the task or problem that the evolutionary algorithm is trying to optimize. The library
/// maximizes the fitness score.
pub trait Individual: Clone + PartialEq + Send + Sync {
    /// Gets the fitness score of the individual.
    ///
    /// # Note
    /// A higher fitness score indicates that the individual is better suited to the task or
    /// problem being optimized. The score may be negative.
    fn fitness(&self) -> f64;

    /// Returns [`true`] if the individual is a valid solution. Invalid individuals never become
    /// part of a [`Population`](crate::Population).
    ///
    /// # Note
    /// The default implementation considers an individual valid when its fitness score is finite,
    /// i.e. neither infinite nor NaN.
    ///
    /// Implementations of this trait may override this method with a more specific definition of
    /// validity, but an individual whose fitness is not finite must never be reported as valid.
    #[inline]
    fn is_valid(&self) -> bool {
        self.fitness().is_finite()
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

use super::*;

/// The `EvolutionModel` trait describes a specific optimization problem to an evolutionary
/// algorithm.
///
/// The model defines the problem-specific parts of an evolutionary algorithm: the individuals,
/// and the operators that combine (crossover), vary (mutation), refine (local search), and
/// evaluate (fitness) them. How and when these operators are applied is decided by the algorithm,
/// for example by a [`Reproduction`].
///
/// The model is shared by the threads that breed new individuals in parallel, so it must be
/// [`Sync`].
pub trait EvolutionModel: Sync {
    /// The type of individual used by the model.
    type Individual: Individual;

    /// Creates an offspring by combining the genetic information of two individuals.
    ///
    /// # Arguments
    /// * `parent1` - The first parent individual.
    /// * `parent2` - The second parent individual.
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// # Returns
    /// The new offspring individual that combines the genetic information of the parents.
    fn crossover(
        &self,
        parent1: &Self::Individual,
        parent2: &Self::Individual,
        rng: &mut Rng,
    ) -> Self::Individual;

    /// Mutates the genetic information of an individual in place, in order to introduce
    /// variation into the population.
    ///
    /// # Arguments
    /// * `individual` - The individual to mutate.
    /// * `rng` - A mutable reference to a random number generator.
    fn mutate(&self, individual: &mut Self::Individual, rng: &mut Rng);

    /// Improves an individual by a local search.
    ///
    /// # Arguments
    /// * `individual` - The individual to improve.
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// # Note
    /// [`Reproduction`] calls this method on every new offspring before its fitness is evaluated.
    /// The default implementation performs no operation.
    fn improve(&self, _individual: &mut Self::Individual, _rng: &mut Rng) {}

    /// Evaluates the fitness of an individual and stores it in the individual, so that
    /// [`Individual::fitness`] returns it afterwards.
    ///
    /// # Arguments
    /// * `individual` - The individual to evaluate.
    ///
    /// # Note
    /// An individual with a higher fitness score is considered to be a better fit. An individual
    /// that is not a valid solution should be evaluated in a way that makes
    /// [`Individual::is_valid`] return [`false`].
    fn evaluate(&self, individual: &mut Self::Individual);

    /// Prepares the model for breeding the next generation from `population`.
    ///
    /// # Note
    /// The algorithms of this library call this method once per generation, before breeding.
    /// A model may use it to adapt its own state, such as the strength of its mutation, to the
    /// population. The default implementation performs no operation.
    fn prepare(&mut self, _population: &Population<Self::Individual>) {}
}

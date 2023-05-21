use super::*;
use object_pool::Pool;
use rayon::prelude::*;

mod ga;
pub use ga::*;

mod aga;
pub use aga::*;

/// Defines the minimum set of parameters required for an evolutionary algorithm.
///
/// This trait abstracts the concept of parameters for an evolutionary algorithm. Any type that implements
/// this trait can be used as a set of parameters for an evolutionary algorithm.
///
/// Implementations of this trait provide a way to control the key parameters of an evolutionary algorithm.
/// These parameters can greatly affect the performance and results of the algorithm.
pub trait EvolutionParameters {
    /// Returns the size of the initial population of individuals at the start of the evolutionary
    /// algorithm.
    ///
    /// The initial population provides the genetic diversity from which the algorithm
    /// evolves solutions to the problem.
    fn initial_population_size(&self) -> usize;

    /// Returns the maximum size of the population in any generation of the evolutionary algorithm.
    ///
    /// Maintaining an upper limit on the size of the population can be useful for controlling the
    /// computational resources used by the algorithm.
    fn max_population_size(&self) -> usize;

    /// Validates the parameters.
    ///
    /// # Panics
    /// This method will panic if any parameter is invalid.
    fn validate(&self);
}

/// Represents the core mechanism of an evolutionary algorithm.
///
/// # Note
/// This trait abstracts the fundamental tasks that an evolutionary algorithm performs.
/// The tasks include the configuration of the algorithm, selection of elites, initialization
/// of the population, and the generation of new populations.
pub trait EvolutionAlgorithm<M: EvolutionModel> {
    /// The type of parameters used for configuring the evolutionary algorithm.
    type Parameters: EvolutionParameters;

    /// Builds an [`Evolution`] instance by binding the evolutionary algorithm with the given model.
    ///
    /// # Arguments
    /// * `model` - The evolutionary model that will be employed by the algorithm.
    ///
    /// # Returns
    /// An [`Evolution`] instance that binds the evolutionary algorithm with the given model.
    ///
    /// # Note
    /// This method is an associated function of the [`EvolutionAlgorithm`] trait. It creates an [`Evolution`]
    /// instance, which is a wrapper around the evolutionary algorithm. The [`Evolution`] instance is necessary
    /// to run the algorithm.
    fn build_evolution(model: M) -> Evolution<Self, M>
    where
        Self: From<M> + Sized,
    {
        Evolution::from(Self::from(model))
    }

    /// Returns a reference to the evolutionary model used by the algorithm.
    fn model(&self) -> &M;

    /// Retrieves the reproduction parameters for the evolutionary model.
    ///
    /// # Returns
    /// A tuple where the first element is a reference to a `ReproductionParameters` struct and the second
    /// element is the target size of the new population.
    ///
    /// The `ReproductionParameters` struct includes the crossover rate, mutation rate, and the selection
    /// mechanism used in the model.
    ///
    /// The size of the new population represents the total number of individuals in the population after
    /// the reproductive process.
    ///
    /// # Note
    /// This method is typically used to access and potentially modify the reproduction parameters during the
    /// evolution process. The size of the new population can be used to understand the scale of the evolution
    /// process.
    fn reproduction_parameters(&self) -> (&ReproductionParameters<M>, usize);

    /// Configures the evolutionary algorithm using the specified parameters.
    ///
    /// # Arguments
    /// * `parameters` - A reference to an instance of the `Parameters` type, which contains the necessary
    ///   configuration for the evolutionary algorithm.
    ///
    /// # Note
    /// This method is used to set up the evolutionary algorithm according to the provided parameters. The
    /// parameters may include, but are not limited to:
    ///
    /// * Reproduction parameters like crossover and mutation rates
    /// * Selection mechanisms
    /// * Population size
    /// * Other algorithm-specific configurations
    ///
    /// The configuration process is typically done before the evolution process begins, and it's crucial
    /// in tailoring the algorithm to the specific problem at hand.
    ///
    /// # Panics
    /// This method will panic if any parameter is invalid. Different implementations may have different
    /// definitions of what constitutes a valid parameter.
    fn configure(&mut self, parameters: &Self::Parameters);

    /// Performs elitism on the population by selecting a subset of the fittest individuals and adding them
    /// to the elites.
    ///
    /// Elitism is a process in genetic algorithms where a portion of the fittest individuals
    /// from the current generation is carried over to the next generation. This helps to ensure
    /// that the most successful individuals, or "elites", are not lost due to crossover and mutation.
    ///
    /// # Arguments
    /// * `population` - A reference to the current population.
    /// * `elites` - A mutable reference to a vector where the selected elite individuals will be added.
    ///
    /// # Note
    /// This method is responsible for selecting the elite individuals from the given population
    /// and adding them to the `elites` vector. The method of selection is left to the specific
    /// implementation.
    ///
    /// The implementation of this method should ensure that the size of the `elites` vector
    /// does not exceed the allowed number of elite individuals. This number is typically
    /// defined as a percentage of the total population size.
    ///
    /// This method has a default implementation that adds the fittest individual in the population to
    /// the elites.
    fn select_elites(
        &self,
        population: &Population<M::Individual>,
        elites: &mut Vec<M::Individual>,
    ) {
        elites.push(population.fittest().clone())
    }

    /// Prepares the initial population for the evolutionary process.
    ///
    /// # Arguments
    /// * `individuals` - A vector containing the individuals in the initial population.
    ///
    /// # Note
    /// This method is responsible for preparing the initial population for the algorithm. Its tasks include,
    /// but are not limited to:
    ///
    /// * Evaluating the fitness of each individual in the population. Each individual is assigned a floating-point
    ///   fitness score that represents how well it solves or performs in the task or problem being optimized.
    ///
    /// * Removing invalid individuals from the population. Invalid individuals might be those that do not comply
    ///   with certain constraints or requirements of the problem being solved.
    ///
    /// An individual with a higher fitness score is considered to be a better fit. The algorithm tries to
    /// maximize the fitness score of the individuals in the population in order to find optimal solutions
    /// to the problem being optimized.
    ///
    /// This method has a default implementation that performs the described tasks.
    ///
    /// # Panics
    /// This method should/will panic when there is no valid individual in the initial population.
    fn initialize_population(
        &self,
        mut individuals: Vec<M::Individual>,
    ) -> Population<M::Individual> {
        individuals
            .par_iter_mut()
            .for_each_with(self.model(), |model, individual| model.evaluate(individual));
        individuals.retain(|individual| individual.is_valid());

        Population::new(individuals)
    }

    /// Prepares the evolutionary algorithm for the next generation.
    ///
    /// # Arguments
    /// * `population` - A reference to the current population.
    ///
    /// # Note
    /// This method is responsible for preparing the evolutionary algorithm for the generation of the next
    /// population. It could involve operations such as adjusting algorithm parameters, preparing the selection
    /// strategy, etc.
    fn prepare_for_next_population(&mut self, population: &Population<M::Individual>);

    /// Generates the new generation of individuals.
    ///
    /// # Arguments
    /// * `population` - A reference to the current population.
    /// * `offspring` - A mutable reference to a vector of offspring individuals.
    /// * `rng_pool` - A pool of random number generators used for stochastic aspects of the evolution process.
    ///
    /// # Note
    /// This method is responsible for generating the new population in the evolutionary process. Its tasks
    /// include, but are not limited to:
    ///
    /// * Breeding new individuals based on the current population. This typically involves processes such as
    ///   selection, crossover, and mutation.
    ///
    /// * Adding the newly bred individuals to the offspring vector.
    ///
    /// This method has a default implementation that uses parallel processing for efficient generation of new
    /// individuals.
    fn generate_new_population(
        &self,
        population: &Population<M::Individual>,
        offspring: &mut Vec<M::Individual>,
        rng_pool: &Pool<M::Rng>,
    ) {
        let (breeding_parameters, offspring_size) = self.reproduction_parameters();
        match offspring_size.saturating_sub(offspring.len()) {
            0 => {}
            1 => {
                let mut rng = rng_pool.pull(M::rng);
                self.model()
                    .breed(population, breeding_parameters, &mut *rng);
            }
            size => {
                offspring.par_extend(rayon::iter::repeatn(self.model(), size).map_init(
                    || rng_pool.pull(M::rng),
                    |rng, model| model.breed(population, breeding_parameters, &mut *rng),
                ));
            }
        }
    }

    /// Replaces the current population with the new one generated from the offspring.
    ///
    /// # Arguments
    /// * `population` - A mutable reference to the current population to be replaced.
    /// * `offspring` - A mutable reference to the vector of offspring individuals.
    ///
    /// # Note
    /// This method is responsible for replacing the current population with the newly generated offspring.
    /// The replacement strategy can significantly impact the performance and characteristics of the
    /// evolutionary algorithm.
    ///
    /// This method has a default implementation that replaces the entire old population with the offspring.
    #[inline]
    fn replace_population(
        &self,
        population: &mut Population<M::Individual>,
        offspring: &mut Vec<M::Individual>,
    ) {
        population.replace(offspring);
    }
}

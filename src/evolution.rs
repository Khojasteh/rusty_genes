use super::*;
use object_pool::Pool;
use std::cell::RefCell;

/// The `Evolution` struct is the primary user-facing interface for running an evolutionary algorithm.
///
/// It wraps an implementation of the [`EvolutionAlgorithm`] trait and provides functionality for the
/// algorithm to be run in various ways, such as from a random initial population, from a single initial
/// individual, or from an initial population.
///
/// # Type parameters
/// * `A` - Represents the core mechanism of an evolutionary algorithm.
/// * `M` - Represents the model of the evolutionary algorithm.
pub struct Evolution<A, M>
where
    A: EvolutionAlgorithm<M>,
    M: EvolutionModel,
{
    algorithm: RefCell<A>,
    _marker: std::marker::PhantomData<M>,
}

impl<A, M> Evolution<A, M>
where
    A: EvolutionAlgorithm<M>,
    M: EvolutionModel,
{
    /// Runs the evolutionary algorithm with a random initial population.
    ///
    /// # Arguments
    /// * `args` - A reference to the domain-specific arguments for generating the random individuals.
    /// * `parameters` - A reference to the algorithm's parameters.
    /// * `progress` - A closure that is called for each generation. The closure receives the current state of
    ///    the population, which includes information such as the generation number, the number of generations
    ///    without improvement (stagnation), and statistical information regarding the fitness scores of the
    ///    individuals in the population. The closure should return a boolean indicating whether the algorithm
    ///    should continue ([`true`]) or terminate ([`false`]).
    ///
    /// # Type Parameters
    /// * `T` - The type of domain-specific arguments for generating the random individuals.
    /// * `F` - The type of progress closure.
    ///
    /// # Returns
    /// The individual with the best fitness score that was found during the execution of the algorithm.
    ///
    /// # Panics
    /// This method will panic if the initial population size is zero or no valid individual could be generated.
    pub fn run<T, F>(&self, args: &T, parameters: &A::Parameters, progress: F) -> M::Individual
    where
        T: Sync,
        M::Individual: RandomIndividual<T>,
        F: FnMut(&Population<M::Individual>) -> bool,
    {
        let mut individuals = Vec::with_capacity(parameters.max_population_size());
        self.algorithm.borrow().model().extend_population(
            &mut individuals,
            parameters.initial_population_size(),
            args,
        );
        self.evolve_population(individuals, parameters, progress)
            .take_fittest()
    }

    /// Runs the evolutionary algorithm with an initial population based on a single individual.
    ///
    /// # Arguments
    /// * `individual` - The individual to use for generating the initial population.
    /// * `parameters` - A reference to the algorithm's parameters.
    /// * `progress` - A closure that is called for each generation. The closure receives the current population
    ///    and the current state of the algorithm, which includes information such as the generation number, the
    ///    number of generations without improvement (stagnation), and statistical information regarding the fitness
    ///    scores of the individuals in the population. The closure should return a boolean indicating whether the
    ///    algorithm should continue ([`true`]) or terminate ([`false`]).
    ///
    /// # Type Parameters
    /// * `F` - The type of progress closure.
    ///
    /// # Returns
    /// The individual with the best fitness score that was found during the execution of the algorithm.
    ///
    /// # Panics
    /// This method will panic if the initial population size is zero or no valid individual could be generated.
    ///
    /// # Note
    /// This method is intended to be used when a good initial individual has already been found. The
    /// method takes advantage of the fact that the genetic information of the initial individual is
    /// likely to be valuable in generating high-quality offspring.
    pub fn evolve_individual<F>(
        &self,
        individual: M::Individual,
        parameters: &A::Parameters,
        progress: F,
    ) -> M::Individual
    where
        F: FnMut(&Population<M::Individual>) -> bool,
    {
        let mut individuals = Vec::with_capacity(parameters.max_population_size());
        self.algorithm.borrow().model().extend_population_with(
            &mut individuals,
            parameters.initial_population_size(),
            individual,
        );
        self.evolve_population(individuals, parameters, progress)
            .take_fittest()
    }

    /// Runs the evolutionary algorithm on a given population.
    ///
    /// # Arguments
    /// * `initial_population` - The initial population to start the algorithm with.
    /// * `parameters` - A reference to the algorithm's parameters.
    /// * `progress` - A closure that is called for each generation. The closure receives the current state of
    ///    the population, which includes information such as the generation number, the number of generations
    ///    without improvement (stagnation), and statistical information regarding the fitness scores of the
    ///    individuals in the population. The closure should return a boolean indicating whether the algorithm
    ///    should continue ([`true`]) or terminate ([`false`]).
    ///
    /// # Type Parameters
    /// * `F` - The type of progress closure.
    ///
    /// # Returns
    /// The population of individual evolved during the run of the algorithm.
    ///
    /// # Panics
    /// This method will panic if there is no valid individual in the initial population to start the
    /// algorithm with.
    ///
    /// # Note
    /// This method allows the algorithm to be run on a pre-existing population, which can be useful
    /// in some contexts (e.g., when the initial population is obtained from some external source, or
    /// when the algorithm is run iteratively on multiple populations).
    pub fn evolve_population<F>(
        &self,
        initial_population: Vec<M::Individual>,
        parameters: &A::Parameters,
        mut progress: F,
    ) -> Population<M::Individual>
    where
        F: FnMut(&Population<M::Individual>) -> bool,
    {
        // Configure the algorithm
        self.algorithm.borrow_mut().configure(parameters);

        // Prepare the population
        let mut population = self
            .algorithm
            .borrow()
            .initialize_population(initial_population);

        // Initialize iterations
        let mut new_population = Vec::with_capacity(parameters.max_population_size());
        let pool = Pool::new(rayon::current_num_threads(), M::rng);

        // Start iterations
        while progress(&population) {
            // Reset the new population.
            new_population.clear();

            // Prepare the algorithm for the new generation
            self.algorithm
                .borrow_mut()
                .prepare_for_next_population(&population);

            // Perform elitism on the population
            self.algorithm
                .borrow()
                .select_elites(&population, &mut new_population);

            // Generate the new population
            self.algorithm.borrow().generate_new_population(
                &population,
                &mut new_population,
                &pool,
            );

            // Replace the current population with the new one
            self.algorithm
                .borrow()
                .replace_population(&mut population, &mut new_population);
        }

        population
    }
}

impl<A, M> Default for Evolution<A, M>
where
    A: EvolutionAlgorithm<M> + Default,
    M: EvolutionModel + Default,
{
    fn default() -> Self {
        Self {
            algorithm: Default::default(),
            _marker: Default::default(),
        }
    }
}

impl<A, M> From<A> for Evolution<A, M>
where
    A: EvolutionAlgorithm<M> + From<M>,
    M: EvolutionModel,
{
    fn from(algorithm: A) -> Self {
        Self {
            algorithm: RefCell::new(algorithm),
            _marker: Default::default(),
        }
    }
}

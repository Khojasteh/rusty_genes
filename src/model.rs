use super::*;
use rand::prelude::*;
use rayon::prelude::*;

/// The `EvolutionModel` trait represents a template for implementing an evolutionary model for a specific
/// optimization problem.
///
/// This trait provides a set of methods that define the evolutionary operators and parameters that are used
/// to evolve a population of individuals over multiple generations in order to find an optimal solution to
/// the problem being optimized.
///
/// The `EvolutionModel` trait is generic over two associated types: `Individual`, which represents an
/// individual solution to the optimization problem, `Rng`, which represents the random number generator
/// used by the algorithm.
///
/// The trait defines several methods that must be implemented by a concrete type in order to define the
/// evolutionary model for a specific problem. These methods include methods for generating an initial
/// population of individuals, breeding new individuals through crossover and mutation, improving the fitness
/// of individuals through local search, and evaluating the fitness of individuals.
///
/// Implementors of this trait are expected to provide their own implementation of these methods that is specific
/// to the problem being optimized.
pub trait EvolutionModel: Sync {
    /// The type of individual used by the model.
    type Individual: Individual;

    /// The type of random number generator used by the model.
    type Rng: Rng + Send;

    /// Returns a random number generator.
    ///
    /// # Returns
    /// A new instance of the random number generator type associated with this genetic algorithm.
    fn rng() -> Self::Rng;

    /// Creates an offspring by combining the genetic information of two individuals.
    ///
    /// # Arguments
    /// * `parent1` - The first parent individual.
    /// * `parent2` - The second parent individual.
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// # Returns
    /// The new offspring individual that combines the genetic information of the parents.
    ///
    /// # Note
    /// This method creates a new offspring individual by combining the genetic information of two
    /// parent individuals.
    ///
    /// The `crossover` method is a fundamental operator in evolutionary algorithms and is used to
    /// explore the search space by creating new individuals with genetic information from multiple
    /// parents.
    fn crossover(
        &self,
        parent1: &Self::Individual,
        parent2: &Self::Individual,
        rng: &mut Self::Rng,
    ) -> Self::Individual;

    /// Mutates the genetic information of an individual.
    ///
    /// # Note
    /// This method mutates the genetic information of an individual in order to introduce variation
    /// into the population.
    ///
    /// The `mutate` method is a key operator in evolutionary algorithms and allows the population
    /// to explore new parts of the search space by randomly modifying individuals.
    ///
    /// # Arguments
    /// * `individual` - The individual to mutate.
    /// * `rng` - A mutable reference to a random number generator.
    fn mutate(&self, individual: &mut Self::Individual, rng: &mut Self::Rng);

    /// Improves an individual by a local search.
    ///
    /// # Arguments
    /// * `individual` - The individual to improve.
    ///
    /// # Note
    /// This method improves the fitness of an individual by performing a local search on it.
    ///
    /// Local search is a common technique used in evolutionary algorithms to refine individuals that are
    /// already close to optimal solutions by making small adjustments to their genetic information.
    ///
    /// This method comes with a default implementation that performs no operation.
    fn improve(&self, _individual: &mut Self::Individual) {}

    /// Evaluates the fitness of an individual.
    ///
    /// # Note
    /// This method evaluates the fitness of an individual by assigning it a floating-point value that represents
    /// how well it performs the task or problem being optimized.
    ///
    /// Fitness evaluation is a key step in evolutionary algorithms, as it allows the algorithm to select the most
    /// promising individuals for breeding and to track the progress of the algorithm over time.
    ///
    /// An individual with a higher fitness score is considered to be a better fit. The genetic algorithm tries to
    /// maximize the fitness score of the individuals in the population in order to find optimal solutions to the
    /// problem being optimized.
    ///
    /// # Arguments
    /// * `individual` - The individual to evaluate.
    fn evaluate(&self, individual: &mut Self::Individual);

    /// Prepares the evolutionary model for the next generation of the evolutionary algorithm.
    ///
    /// # Arguments
    /// * `population` - A reference to the current population.
    /// * `breeding_parameters` - A mutable reference to the reproduction parameters.
    ///
    /// # Note
    /// This method is responsible for preparing the evolutionary model for the generation of the next
    /// population. It could involve operations such as adjusting parameters, choosing crossover and
    /// mutation strategies, etc.
    ///
    /// This method comes with a default implementation that performs no operation.
    fn pre_generation(
        &mut self,
        _population: &Population<Self::Individual>,
        _breeding_parameters: &mut ReproductionParameters<Self>,
    ) where
        Self: Sized,
    {
    }

    /// Breeds a new individual by combining genetic information from parent individuals.
    ///
    /// # Arguments
    /// * `population` - The population of individuals to select parents from.
    /// * `parameters` - The reproduction parameters.
    /// * `rng` - A mutable reference to the random number generator to use during breeding.
    ///
    /// # Returns
    /// A new offspring individual.
    ///
    /// # Note
    /// This method has a default implementation that selects parent (or parents) for a new offspring
    /// and, depending on the specified probabilities, applies crossover and mutation operations to
    /// generate the offspring. The method then applies a local search on the offspring before evaluating
    /// its fitness score.
    ///
    /// If the generated offspring is valid and wins a competition with its parents, the method returns
    /// the offspring; otherwise, it returns the fittest parent (keep-best reproduction).
    fn breed(
        &self,
        population: &Population<Self::Individual>,
        parameters: &ReproductionParameters<Self>,
        rng: &mut Self::Rng,
    ) -> Self::Individual
    where
        Self: Sized,
    {
        let (_, parent1) = parameters.parent_selector.select(population, rng);
        let parent2 = if population.len() > 1 && rng.gen::<f64>() < parameters.crossover_rate {
            parameters
                .parent_selector
                .select_distinct(population, rng, 3, parent1)
                .map(|(_, parent)| parent)
        } else {
            None
        };

        let mut fittest_parent = parent1;
        let mut offspring = if let Some(parent2) = parent2 {
            if parent2.fitness() > parent1.fitness() {
                fittest_parent = parent2;
            }
            self.crossover(parent1, parent2, rng)
        } else {
            parent1.clone()
        };

        if rng.gen::<f64>() < parameters.mutation_rate {
            self.mutate(&mut offspring, rng);
        }

        self.improve(&mut offspring);
        self.evaluate(&mut offspring);

        if offspring.is_valid() && self.can_survive(&offspring, fittest_parent, population, rng) {
            offspring
        } else {
            fittest_parent.clone()
        }
    }

    /// Determines whether an offspring individual can survive to the next generation.
    ///
    /// # Arguments
    /// * `offspring` - A reference to the offspring individual.
    /// * `fittest_parent` - A reference to the fittest parent of the offspring.
    /// * `population` - A reference to the the current population.
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// # Returns
    /// A boolean indicating whether the offspring can survive.
    ///
    /// # Note
    /// This function is used in the context of survival selection in the genetic algorithm.
    /// It helps to decide which individuals will be a part of the next generation, thus
    /// influencing the direction of the evolutionary search.
    ///
    /// This method has a default implementation that returns [`true`] if the offspring's fitness
    /// is better or same as the parents, [`false`] otherwise (family competition).
    #[inline]
    fn can_survive(
        &self,
        offspring: &Self::Individual,
        fittest_parent: &Self::Individual,
        _population: &Population<Self::Individual>,
        _rng: &mut Self::Rng,
    ) -> bool {
        offspring.fitness() >= fittest_parent.fitness()
    }

    /// Extends a population with randomly generated individuals.
    ///
    /// # Arguments
    /// * `individuals` - A mutable reference to vector of individuals.
    /// * `size` - The number of individuals to be added to the population.
    /// * `args` - A reference to domain-specific arguments for generating the individuals.
    ///
    /// # Type Parameters
    /// * `T` - The type of domain-specific arguments.
    ///
    /// # Note
    /// This method uses the `RandomIndividual` trait to generate individuals randomly. This
    /// allows the domain-specific arguments (`args`) to be used to customize the generation of
    /// the individuals, resulting in a population that is more tailored to the problem being
    /// solved.
    ///
    /// This method has a default implementation that efficiently generates individuals in parallel.
    fn extend_population<T>(&self, individuals: &mut Vec<Self::Individual>, size: usize, args: &T)
    where
        T: Sync,
        Self::Individual: RandomIndividual<T>,
    {
        if size == 0 {
            return;
        }

        individuals.reserve(size);
        individuals.par_extend(
            rayon::iter::repeatn(args, size).map_init(Self::rng, |rng, args| {
                Self::Individual::new_random(args, rng)
            }),
        );
    }

    /// Extends a population based on a single individual.
    ///
    /// # Arguments
    /// * `individuals` - A mutable reference to vector of individuals.
    /// * `size` - The number of individuals to be added to the population.
    /// * `model` - The individual to use as a model for the new individuals.
    ///
    /// # Note
    /// This method is intended to be used when a good initial individual has already been found.
    /// The method takes advantage of the fact that the genetic information of the initial individual
    /// is likely to be valuable in generating high-quality offspring.
    ///
    /// This method has a default implementation that efficiently generates and mutates individuals
    /// in parallel.
    fn extend_population_with(
        &self,
        individuals: &mut Vec<Self::Individual>,
        size: usize,
        model: Self::Individual,
    ) {
        if size == 0 {
            return;
        }

        individuals.reserve(size);
        individuals.push(model.clone());
        if size == 1 {
            return;
        }

        individuals.par_extend(rayon::iter::repeatn(model, size - 1).map_init(
            Self::rng,
            |rng, mut new_individual| {
                self.mutate(&mut new_individual, rng);
                new_individual
            },
        ));
    }
}

/// Represents the reproduction parameters of an evolutionary model.
pub struct ReproductionParameters<M: EvolutionModel> {
    /// The crossover rate represents the likelihood of two individuals combining their genetic information
    /// to produce offspring. It is a probability value between 0 and 1, where a higher value indicates a
    /// higher likelihood of crossover occurring.
    pub crossover_rate: f64,

    /// The mutation rate represents the likelihood of an individual undergoing a mutation process. It is
    /// a probability value between 0 and 1, where a higher value signifies a greater probability that
    /// an individual will undergo mutation.
    pub mutation_rate: f64,

    /// The parent selector defines the mechanism used to select individuals from the population for
    /// reproduction. Different selection mechanisms can influence the speed and direction of evolution by
    /// favoring individuals based on different criteria, such as fitness or diversity.
    pub parent_selector: Box<dyn SelectionMechanism<M::Individual, M::Rng>>,
}

impl<M: EvolutionModel> Default for ReproductionParameters<M> {
    fn default() -> Self {
        Self {
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            parent_selector: Box::<RouletteWheelSelection>::default(),
        }
    }
}

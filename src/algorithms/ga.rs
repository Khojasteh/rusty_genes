use super::*;

/// Represents the parameters used in a genetic algorithm.
///
/// # Note
/// The `GeneticAlgorithmParameters` struct contains the various parameters and options that control
/// the behavior of a genetic algorithm. These settings include the maximum number of generations
/// to run the algorithm, the crossover rate and mutation rate used in the genetic operators, and
/// the size of the population used in the algorithm.
pub struct GeneticAlgorithmParameters {
    /// The number of individuals in the population.
    ///
    /// # Note
    /// `population_size` is a positive integer that determines the size of the population
    /// used in the genetic algorithm. A larger population size can help improve the diversity
    /// of the population and reduce the risk of premature convergence, but it can also increase
    /// the computational cost of the algorithm.
    pub population_size: usize,

    /// The proportion of the fittest individuals in the current population that are directly carried
    /// over to the next generation.
    ///
    /// # Note
    /// `elitism_rate` is a floating-point value between 0.0 and 1.0, excluding. A value of 0 means no
    /// elitism (i.e., all individuals in the next generation are produced by crossover and mutation), while
    /// a value close to 1 means high elitism (i.e., most of the individuals in the next generation are the
    /// fittest ones from the current generation).
    ///
    /// Elitism ensures that the fittest individuals are preserved from generation to generation, aiding
    /// in overall performance improvement of the population over time. However, setting this value too
    /// high may limit the diversity of the population and could lead to premature convergence.
    pub elitism_rate: f64,

    /// The probability of two individuals combining their genetic information to produce offspring.
    ///
    /// # Note
    /// `crossover_rate` is a floating-point value between 0.0 and 1.0, inclusive. A higher value
    /// increases the likelihood of crossover events occurring.
    pub crossover_rate: f64,

    /// The probability of an individual undergoing the mutation process.
    ///
    /// # Note
    /// `mutation_rate` is a floating-point value between 0.0 and 1.0, inclusive. A higher value
    /// increases the likelihood of mutations occurring, which can introduce more genetic diversity
    /// in the population.
    pub mutation_rate: f64,

    /// The selection strategy used to select individuals from the population.
    ///
    /// # Note
    /// The selection strategy is an important aspect of an evolutionary algorithm. It determines how
    /// individuals are chosen from the population for reproduction and thereby, influences the diversity
    /// and convergence speed of the algorithm.
    pub selection_strategy: SelectionStrategy,
}

impl Default for GeneticAlgorithmParameters {
    /// Creates an instance of [`GeneticAlgorithmParameters`] with default values.
    ///
    /// # Default Values
    /// * population_size: 100
    /// * elitism_rate: 0.01
    /// * crossover_rate: 0.8
    /// * mutation_rate: 0.1
    /// * selection_strategy: [`SelectionStrategy::RouletteWheel`]
    fn default() -> Self {
        Self {
            population_size: 100,
            elitism_rate: 0.01,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            selection_strategy: SelectionStrategy::RouletteWheel,
        }
    }
}

impl EvolutionParameters for GeneticAlgorithmParameters {
    fn initial_population_size(&self) -> usize {
        self.population_size
    }

    fn max_population_size(&self) -> usize {
        self.population_size
    }

    fn validate(&self) {
        assert!(self.population_size != 0);
        assert!((0.0..1.0).contains(&self.elitism_rate));
        assert!((0.0..=1.0).contains(&self.crossover_rate));
        assert!((0.0..=1.0).contains(&self.mutation_rate));
    }
}

/// A genetic algorithm for evolving populations towards better fitness values.
///
/// # Note
/// This struct implements a standard genetic algorithm that relies on fixed parameters, such as
/// survival, crossover, and mutation rates, throughout the evolution process. These parameters
/// remain constant during the entire optimization process.
///
/// The algorithm operates on an [`EvolutionModel`] which defines the problem-specific components
/// such as crossover and mutation operators, as well as the fitness function for evaluating
/// individuals in the population.
///
/// While the `GeneticAlgorithm` can be effective in solving optimization problems, it may not be as
/// well-suited to tackle complex problems with varying search landscapes and diverse solution spaces
/// compared to adaptive genetic algorithms that dynamically adjust their parameters during the
/// iterations.
///
/// # Example
/// ```
/// use rusty_genes::*;
/// use rusty_genes::one_max::*;
///
/// const ONE_MAX_LENGTH: usize = 100;
///
/// let one_max = OneMaxEvolutionModel;
/// let ga = GeneticAlgorithm::build_evolution(one_max);
/// let ga_params = GeneticAlgorithmParameters {
///     crossover_rate: 0.7,
///     mutation_rate: 0.2,
///     selection_strategy: SelectionStrategy::Linear(1.25),
///     ..Default::default()
/// };
///
/// let solution = ga.run(&ONE_MAX_LENGTH, &ga_params, |population| {
///     population.fittest().fitness() < 0.999999 && population.generation() <= 100
/// });
///
/// assert!(solution.genome().iter().all(|&one| one == true));
/// ```
/// Implements the fundamental tasks in a genetic algorithm.
pub struct GeneticAlgorithm<M: EvolutionModel> {
    model: M,
    population_size: usize,
    elite_size: usize,
    breeding: ReproductionParameters<M>,
}

impl<M: EvolutionModel> EvolutionAlgorithm<M> for GeneticAlgorithm<M> {
    type Parameters = GeneticAlgorithmParameters;

    #[inline]
    fn model(&self) -> &M {
        &self.model
    }

    #[inline]
    fn reproduction_parameters(&self) -> (&ReproductionParameters<M>, usize) {
        (&self.breeding, self.population_size)
    }

    #[inline]
    fn configure(&mut self, parameters: &GeneticAlgorithmParameters) {
        parameters.validate();

        self.population_size = parameters.population_size;
        self.elite_size = interpolate_usize(0, parameters.population_size, parameters.elitism_rate);
        self.breeding.crossover_rate = parameters.crossover_rate;
        self.breeding.mutation_rate = parameters.mutation_rate;
        self.breeding.parent_selector = parameters.selection_strategy.create_selector();
    }

    #[inline]
    fn select_elites(
        &self,
        population: &Population<M::Individual>,
        elites: &mut Vec<M::Individual>,
    ) {
        if self.elite_size != 0 {
            let range_start = population.len().saturating_sub(self.elite_size);
            elites.extend(population[range_start..].iter().cloned());
        }
    }

    #[inline]
    fn prepare_for_next_population(&mut self, population: &Population<M::Individual>) {
        self.model.pre_generation(population, &mut self.breeding);
        self.breeding.parent_selector.prepare(population);
    }
}

impl<M: EvolutionModel> From<M> for GeneticAlgorithm<M> {
    fn from(model: M) -> Self {
        Self {
            model,
            population_size: Default::default(),
            elite_size: Default::default(),
            breeding: Default::default(),
        }
    }
}

impl<M: EvolutionModel + Default> Default for GeneticAlgorithm<M> {
    fn default() -> Self {
        Self::from(M::default())
    }
}

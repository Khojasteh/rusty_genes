use super::*;
use std::ops::Range;

/// Represents the parameters used in an adaptive genetic algorithm.
///
/// # Note
/// The `AdaptiveGeneticAlgorithmParameters` struct contains the various parameters and options that
/// control the behavior of a genetic algorithm. These settings include the maximum number of generations
/// to run the algorithm, the crossover rate and mutation rate used in the genetic operators, the size
/// of the population used in the algorithm, the elitism ratio used in the elitism process, and the
/// interval at which the parameters are adjusted.
pub struct AdaptiveGeneticAlgorithmParameters {
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

    /// The range of probabilities of two individuals combining their genetic information to produce offspring.
    /// This range is dynamically adjusted based on the diversity of the population.
    ///
    /// # Note
    /// `crossover_rate` is a range of floating-point values between 0.0 and 1.0, inclusive. A higher value
    /// increases the likelihood of crossover events occurring.
    pub crossover_rate_range: Range<f64>,

    /// The range of probabilities of an individual undergoing the mutation process.
    /// This range is dynamically adjusted based on the diversity of the population.
    ///
    /// # Note
    /// `mutation_rate` is a range of floating-point values between 0.0 and 1.0, inclusive. A higher value
    /// increases the likelihood of mutations occurring, which can introduce more genetic diversity
    /// in the population.
    pub mutation_rate_range: Range<f64>,

    /// The selection strategy used to select individuals from the population.
    ///
    /// # Note
    /// The selection strategy is an important aspect of an evolutionary algorithm. It determines how
    /// individuals are chosen from the population for reproduction and thereby, influences the diversity
    /// and convergence speed of the algorithm.
    pub selection_strategy: SelectionStrategy,

    /// The interval at which the parameters are adjusted based on the population's diversity.
    pub parameter_adjustment_interval: usize,
}

impl Default for AdaptiveGeneticAlgorithmParameters {
    /// Creates an instance of [`AdaptiveGeneticAlgorithmParameters`] with default values.
    ///
    /// # Default Values
    /// * population_size: 100
    /// * elitism_rate: 0.01
    /// * crossover_rate_range: 0.3..1.0
    /// * mutation_rate_range: 0.1..1.0,
    /// * selection_strategy: [`SelectionStrategy::RouletteWheel`]
    /// * parameter_adjustment_interval: 1
    fn default() -> Self {
        Self {
            population_size: 100,
            elitism_rate: 0.01,
            crossover_rate_range: 0.3..1.0,
            mutation_rate_range: 0.1..1.0,
            selection_strategy: SelectionStrategy::RouletteWheel,
            parameter_adjustment_interval: 1,
        }
    }
}

impl EvolutionParameters for AdaptiveGeneticAlgorithmParameters {
    fn initial_population_size(&self) -> usize {
        self.population_size
    }

    fn max_population_size(&self) -> usize {
        self.population_size
    }

    fn validate(&self) {
        assert!(self.population_size != 0);
        assert!((0.0..1.0).contains(&self.elitism_rate));
        assert!((0.0..=1.0).contains(&self.crossover_rate_range.start));
        assert!((0.0..=1.0).contains(&self.crossover_rate_range.end));
        assert!((0.0..=1.0).contains(&self.mutation_rate_range.start));
        assert!((0.0..=1.0).contains(&self.mutation_rate_range.end));
    }
}

/// An adaptive genetic algorithm for evolving populations towards better fitness values.
///
/// # Note
/// This struct represents a genetic algorithm that adaptively adjusts its parameters based on the diversity
/// of the population, with the goal of maintaining a balance between exploration and exploitation.
///
/// The crossover and mutation rates are dynamically adjusted based on the calculated diversity value. If
/// diversity is high, the algorithm increases crossover rate and decreases mutation rate. This promotes
/// the exploitation of existing diversity through crossover operations. Conversely, when diversity is low,
/// the algorithm increases mutation rate and decreases crossover rate to encourage exploration by introducing
/// and maintaining more diversity in the population.
///
/// These adaptive adjustments happen on a configured interval, making the algorithm dynamically responsive
/// to the state and diversity of the current population.
///
/// The algorithm operates on an [`EvolutionModel`] which defines the problem-specific components such
/// as crossover and mutation operators, as well as the fitness function for evaluating individuals in
/// the population.
///
/// By adapting its parameters throughout the iterations, the `AdaptiveGeneticAlgorithm` is better
/// suited to tackle complex optimization problems with varying search landscapes and diverse solution
/// spaces.
///
/// # Example
/// ```
/// use rusty_genes::*;
/// use rusty_genes::one_max::*;
///
/// const ONE_MAX_LENGTH: usize = 100;
///
/// let one_max = OneMaxEvolutionModel;
/// let adaptive_ga = AdaptiveGeneticAlgorithm::build_evolution(one_max);
/// let adaptive_ga_params = AdaptiveGeneticAlgorithmParameters {
///     selection_strategy: SelectionStrategy::Tournament(4),
///     ..Default::default()
/// };
///
/// let solution = adaptive_ga.run(&ONE_MAX_LENGTH, &adaptive_ga_params, |population| {
///     population.fittest().fitness() < 0.999999 && population.generation() <= 100
/// });
///
/// assert!(solution.genome().iter().all(|&one| one));
/// ```
pub struct AdaptiveGeneticAlgorithm<M: EvolutionModel> {
    model: M,
    population_size: usize,
    elite_size: usize,
    adjustment_interval: usize,
    crossover_rate_range: Range<f64>,
    mutation_rate_range: Range<f64>,
    breeding: ReproductionParameters<M>,
    reference_std: f64,
}

impl<M: EvolutionModel> EvolutionAlgorithm<M> for AdaptiveGeneticAlgorithm<M> {
    type Parameters = AdaptiveGeneticAlgorithmParameters;

    #[inline]
    fn model(&self) -> &M {
        &self.model
    }

    #[inline]
    fn reproduction_parameters(&self) -> (&ReproductionParameters<M>, usize) {
        (&self.breeding, self.population_size)
    }

    #[inline]
    fn configure(&mut self, parameters: &AdaptiveGeneticAlgorithmParameters) {
        parameters.validate();

        self.population_size = parameters.population_size;
        self.elite_size = interpolate_usize(0, parameters.population_size, parameters.elitism_rate);
        self.adjustment_interval = parameters.parameter_adjustment_interval;
        self.crossover_rate_range = parameters.crossover_rate_range.clone();
        self.mutation_rate_range = parameters.mutation_rate_range.clone();
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
        match population.generation() {
            0 => {
                self.reference_std = population.stats().std();
                self.breeding.crossover_rate = self.crossover_rate_range.end;
                self.breeding.mutation_rate = self.mutation_rate_range.start;
            }
            n if self.adjustment_interval != 0 && n % self.adjustment_interval == 0 => {
                let diversity = (population.stats().std() / self.reference_std).clamp(0.0, 1.0);

                self.breeding.crossover_rate = interpolate_f64(
                    self.crossover_rate_range.start,
                    self.crossover_rate_range.end,
                    diversity,
                );
                self.breeding.mutation_rate = interpolate_f64(
                    self.mutation_rate_range.start,
                    self.mutation_rate_range.end,
                    1.0 - diversity,
                );
            }
            _ => {}
        }

        self.model.pre_generation(population, &mut self.breeding);
        self.breeding.parent_selector.prepare(population);
    }
}

impl<M: EvolutionModel> From<M> for AdaptiveGeneticAlgorithm<M> {
    fn from(model: M) -> Self {
        Self {
            model,
            population_size: Default::default(),
            elite_size: Default::default(),
            breeding: Default::default(),
            reference_std: Default::default(),
            adjustment_interval: Default::default(),
            crossover_rate_range: Default::default(),
            mutation_rate_range: Default::default(),
        }
    }
}

impl<M: EvolutionModel + Default> Default for AdaptiveGeneticAlgorithm<M> {
    fn default() -> Self {
        Self::from(M::default())
    }
}

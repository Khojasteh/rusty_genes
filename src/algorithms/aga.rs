use super::*;
use std::ops::RangeInclusive;

/// Represents the parameters of an [`AdaptiveGeneticAlgorithm`].
///
/// Start with [`Default::default`] and modify fields before passing them to
/// [`AdaptiveGeneticAlgorithm::new`], which validates them. Stopping belongs to the model's goal
/// and the progress callback supplied to [`Evolution`].
///
/// New parameters may be added in the future, so the parameters are created by
/// [`Default::default`] and then changed as needed.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaptiveGeneticAlgorithmParameters {
    /// The number of individuals in a full generation.
    ///
    /// # Note
    /// `population_size` must be positive. A larger population can improve the diversity of the
    /// population and reduce the risk of premature convergence, but it also increases the
    /// computational cost of the algorithm.
    pub population_size: usize,

    /// The proportion of the fittest individuals in the current generation that are directly
    /// carried over to the next generation.
    ///
    /// # Note
    /// `elitism_rate` must be between 0.0 (inclusive) and 1.0 (exclusive). A value of 0.0 means
    /// no elitism, i.e. all individuals of the next generation are bred, while a value close to
    /// 1.0 means that most individuals of the next generation are the fittest ones of the
    /// current generation. The number of elites is rounded down.
    pub elitism_rate: f64,

    /// The range of the probability of two individuals combining their genetic information to
    /// produce an offspring.
    ///
    /// # Note
    /// Both bounds must be between 0.0 and 1.0, inclusive, and the start must not exceed the end.
    /// The crossover rate moves towards the end of the range as the diversity of the population
    /// increases.
    pub crossover_rate_range: RangeInclusive<f64>,

    /// The range of the probability of an offspring undergoing mutation.
    ///
    /// # Note
    /// Both bounds must be between 0.0 and 1.0, inclusive, and the start must not exceed the end.
    /// The mutation rate moves towards the end of the range as the diversity of the population
    /// decreases.
    pub mutation_rate_range: RangeInclusive<f64>,

    /// The strategy for selecting parents from the population.
    pub selection_strategy: SelectionStrategy,

    /// The number of generations between adjustments of the crossover and mutation rates.
    ///
    /// # Note
    /// `parameter_adjustment_interval` must be positive.
    pub parameter_adjustment_interval: usize,
}

impl Default for AdaptiveGeneticAlgorithmParameters {
    /// Creates an instance of [`AdaptiveGeneticAlgorithmParameters`] with default values.
    ///
    /// # Default Values
    /// * population_size: 100
    /// * elitism_rate: 0.01
    /// * crossover_rate_range: 0.3..=1.0
    /// * mutation_rate_range: 0.1..=1.0
    /// * selection_strategy: [`SelectionStrategy::RouletteWheel`]
    /// * parameter_adjustment_interval: 1
    fn default() -> Self {
        Self {
            population_size: 100,
            elitism_rate: 0.01,
            crossover_rate_range: 0.3..=1.0,
            mutation_rate_range: 0.1..=1.0,
            selection_strategy: SelectionStrategy::RouletteWheel,
            parameter_adjustment_interval: 1,
        }
    }
}

impl AdaptiveGeneticAlgorithmParameters {
    /// Validates the parameters.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] for the first parameter that is outside
    /// of its valid domain.
    pub fn validate(&self) -> Result<(), EvolutionError> {
        if self.population_size == 0 {
            return Err(EvolutionError::invalid_parameter(
                "population_size",
                "must be positive",
            ));
        }
        check_ratio("elitism_rate", self.elitism_rate)?;
        check_probability_range("crossover_rate_range", &self.crossover_rate_range)?;
        check_probability_range("mutation_rate_range", &self.mutation_rate_range)?;
        if self.parameter_adjustment_interval == 0 {
            return Err(EvolutionError::invalid_parameter(
                "parameter_adjustment_interval",
                "must be positive",
            ));
        }
        self.selection_strategy.validate()
    }
}

fn check_probability_range(
    name: &'static str,
    range: &RangeInclusive<f64>,
) -> Result<(), EvolutionError> {
    check_probability(name, *range.start())?;
    check_probability(name, *range.end())?;
    if range.start() > range.end() {
        return Err(EvolutionError::invalid_parameter(
            name,
            "the start must not exceed the end",
        ));
    }
    Ok(())
}

/// An adaptive genetic algorithm for evolving populations towards better score values.
///
/// # Note
/// This algorithm adjusts its crossover and mutation rates to the diversity of the population,
/// with the goal of balancing exploration and exploitation.
///
/// The diversity is the standard deviation of the scores of the current generation
/// relative to that of the initial generation, limited to the range 0.0 to 1.0. When the
/// initial generation has no spread at all, the first spread observed later becomes the
/// reference. When diversity is high, the algorithm increases the crossover rate and decreases
/// the mutation rate, which exploits the existing diversity through crossover. When diversity is
/// low, it decreases the crossover rate and increases the mutation rate, which encourages
/// exploration by introducing more diversity into the population.
/// The spread ratio is computed using scaled statistics so it remains usable even when the
/// standard deviations in the score's units would overflow.
///
/// The rates start at the highest crossover rate and the lowest mutation rate, and are adjusted
/// at the configured interval. The algorithm breeds its offspring by a [`Reproduction`], whose
/// parent selector and survival policy can be replaced through
/// [`reproduction_mut`](AdaptiveGeneticAlgorithm::reproduction_mut); the rates the algorithm
/// sets replace any rates set there.
///
/// The algorithm operates on an [`EvolutionModel`], which defines the problem-specific
/// components such as the crossover and mutation operators, and the score function.
///
/// # Example
/// Runs the algorithm on the model of the [OneMax example].
/// ```
/// # mod one_max { include!("../../examples/one_max.rs"); }
/// # use one_max::*;
/// use rusty_genes::*;
///
/// const ONE_MAX_LENGTH: usize = 100;
///
/// let mut aga_params = AdaptiveGeneticAlgorithmParameters::default();
/// aga_params.selection_strategy = SelectionStrategy::Tournament(4);
/// let mut aga = AdaptiveGeneticAlgorithm::new(OneMaxEvolutionModel, aga_params)?.into_evolution();
///
/// let limits = Limits::new().generations(200);
/// let solution = aga
///     .run(&ONE_MAX_LENGTH, |state| limits.check(state))?
///     .into_fittest_so_far();
///
/// println!("Best score: {} of {}", solution.score(), ONE_MAX_LENGTH);
/// # Ok::<(), EvolutionError>(())
/// ```
///
/// [OneMax example]: https://github.com/Khojasteh/rusty_genes/blob/master/examples/one_max.rs
pub struct AdaptiveGeneticAlgorithm<M: EvolutionModel> {
    model: M,
    parameters: AdaptiveGeneticAlgorithmParameters,
    elite_count: usize,
    reproduction: Reproduction<M::Individual>,
    reference_spread: Option<Spread>,
}

impl<M: EvolutionModel> AdaptiveGeneticAlgorithm<M> {
    /// Creates an adaptive genetic algorithm that operates on `model` with the given parameters.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if a parameter is invalid.
    pub fn new(
        model: M,
        parameters: AdaptiveGeneticAlgorithmParameters,
    ) -> Result<Self, EvolutionError> {
        parameters.validate()?;
        let reproduction = Reproduction::from_strategy(
            parameters.selection_strategy,
            *parameters.crossover_rate_range.end(),
            *parameters.mutation_rate_range.start(),
        )?;
        Ok(Self {
            model,
            elite_count: elite_count(parameters.population_size, parameters.elitism_rate),
            parameters,
            reproduction,
            reference_spread: None,
        })
    }

    /// Returns the parameters the algorithm was created with.
    #[inline]
    pub fn parameters(&self) -> &AdaptiveGeneticAlgorithmParameters {
        &self.parameters
    }

    /// Returns the reproduction that breeds the offspring, with the rates of the current
    /// generation.
    #[inline]
    pub fn reproduction(&self) -> &Reproduction<M::Individual> {
        &self.reproduction
    }

    /// Returns the reproduction that breeds the offspring, to replace its parent selector or
    /// survival policy.
    #[inline]
    pub fn reproduction_mut(&mut self) -> &mut Reproduction<M::Individual> {
        &mut self.reproduction
    }

    /// Sets the crossover rate to the point of its range given by `crossover_ratio`, and the
    /// mutation rate to the point of its range given by `mutation_ratio`.
    fn set_rates(&mut self, crossover_ratio: f64, mutation_ratio: f64) {
        let rate_in = |range: &RangeInclusive<f64>, ratio| {
            interpolate_f64(*range.start(), *range.end(), ratio).clamp(*range.start(), *range.end())
        };
        let crossover_rate = rate_in(&self.parameters.crossover_rate_range, crossover_ratio);
        let mutation_rate = rate_in(&self.parameters.mutation_rate_range, mutation_ratio);
        self.reproduction
            .set_rates(crossover_rate, mutation_rate)
            .expect("the rate ranges are validated on creation");
    }

    /// Returns the diversity of the population, between 0.0 and 1.0, relative to the reference
    /// standard deviation, which it adopts first if there is none yet.
    fn diversity(&mut self, population: &Population<M::Individual>) -> f64 {
        let Some(spread) = population.stats().spread() else {
            return 0.0;
        };
        let reference = self.reference_spread.get_or_insert(spread);
        spread.ratio(*reference).clamp(0.0, 1.0)
    }
}

impl<M: EvolutionModel> EvolutionAlgorithm for AdaptiveGeneticAlgorithm<M> {
    type Model = M;

    #[inline]
    fn model(&self) -> &M {
        &self.model
    }

    #[inline]
    fn population_size(&self) -> usize {
        self.parameters.population_size
    }

    fn start_run(&mut self) {
        self.reproduction.reset();
        self.model.start_run();
    }

    fn prepare_generation(&mut self, state: &RunState<M::Individual>) {
        let population = state.population();
        let generation = state.generation();
        if generation == 0 {
            self.reference_spread = population.stats().spread();
            self.set_rates(1.0, 0.0);
        } else if generation % self.parameters.parameter_adjustment_interval == 0 {
            let diversity = self.diversity(population);
            self.set_rates(diversity, 1.0 - diversity);
        }

        self.model.prepare(state);
        self.reproduction.prepare(population);
    }

    #[inline]
    fn select_elites(
        &self,
        population: &Population<M::Individual>,
        elites: &mut Vec<M::Individual>,
    ) {
        select_fittest(population, self.elite_count, elites);
    }

    #[inline]
    fn generate_offspring(
        &self,
        state: &RunState<M::Individual>,
        offspring: &mut Vec<M::Individual>,
        rng: &mut Rng,
    ) {
        let size = self.population_size();
        self.reproduction
            .breed_into(&self.model, state.population(), offspring, size, rng);
    }
}

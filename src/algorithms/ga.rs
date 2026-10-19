use super::*;

/// Represents the parameters of a [`GeneticAlgorithm`].
///
/// Start with [`Default::default`] and modify fields before passing them to
/// [`GeneticAlgorithm::new`], which validates them. Stopping belongs to the model's goal and
/// the progress callback supplied to [`Evolution`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GeneticAlgorithmParameters {
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
    ///
    /// Elitism preserves the fittest individuals from generation to generation. However, a high
    /// value may limit the diversity of the population and lead to premature convergence.
    pub elitism_rate: f64,

    /// The probability of two individuals combining their genetic information to produce an
    /// offspring.
    ///
    /// # Note
    /// `crossover_rate` must be between 0.0 and 1.0, inclusive.
    pub crossover_rate: f64,

    /// The probability of an offspring undergoing mutation.
    ///
    /// # Note
    /// `mutation_rate` must be between 0.0 and 1.0, inclusive. A higher value introduces more
    /// genetic diversity into the population.
    pub mutation_rate: f64,

    /// The strategy for selecting parents from the population.
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
    /// * selection_strategy: [`SelectionStrategy::Tournament(3)`](SelectionStrategy::Tournament)
    fn default() -> Self {
        Self {
            population_size: 100,
            elitism_rate: 0.01,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            selection_strategy: SelectionStrategy::Tournament(3),
        }
    }
}

impl GeneticAlgorithmParameters {
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
        check_probability("crossover_rate", self.crossover_rate)?;
        check_probability("mutation_rate", self.mutation_rate)?;
        self.selection_strategy.validate()
    }
}

/// A genetic algorithm with crossover and mutation probabilities fixed for the whole run.
///
/// The algorithm operates on an [`EvolutionModel`], which defines the problem-specific
/// components such as the crossover and mutation operators, and the score function.
///
/// The algorithm breeds its offspring by a [`Reproduction`], whose parent selector and survival
/// policy can be replaced through [`reproduction_mut`](GeneticAlgorithm::reproduction_mut).
///
/// While the `GeneticAlgorithm` can be effective in solving optimization problems, the
/// [`AdaptiveGeneticAlgorithm`], which adjusts its rates to the diversity of the population,
/// may be better suited to problems with varying search landscapes.
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
/// let mut ga_params = GeneticAlgorithmParameters::default();
/// ga_params.crossover_rate = 0.7;
/// ga_params.mutation_rate = 0.2;
/// ga_params.selection_strategy = SelectionStrategy::Linear(1.25);
/// let mut ga = GeneticAlgorithm::new(OneMaxEvolutionModel, ga_params)?.into_evolution();
///
/// let limits = Limits::new().generations(200);
/// let solution = ga
///     .run(&ONE_MAX_LENGTH, |state| limits.check(state))?
///     .into_fittest_so_far();
///
/// println!("Best score: {} of {}", solution.score(), ONE_MAX_LENGTH);
/// # Ok::<(), EvolutionError>(())
/// ```
///
/// [OneMax example]: https://github.com/Khojasteh/rusty_genes/blob/master/examples/one_max.rs
pub struct GeneticAlgorithm<M: EvolutionModel> {
    model: M,
    parameters: GeneticAlgorithmParameters,
    elite_count: usize,
    reproduction: Reproduction<M::Individual>,
}

impl<M: EvolutionModel> GeneticAlgorithm<M> {
    /// Creates a genetic algorithm that operates on `model` with the given parameters.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if a parameter is invalid.
    pub fn new(model: M, parameters: GeneticAlgorithmParameters) -> Result<Self, EvolutionError> {
        parameters.validate()?;
        let reproduction = Reproduction::from_strategy(
            parameters.selection_strategy,
            parameters.crossover_rate,
            parameters.mutation_rate,
        )?;
        Ok(Self {
            model,
            elite_count: elite_count(parameters.population_size, parameters.elitism_rate),
            parameters,
            reproduction,
        })
    }

    /// Returns the parameters the algorithm was created with.
    #[inline]
    pub fn parameters(&self) -> &GeneticAlgorithmParameters {
        &self.parameters
    }

    /// Returns the reproduction that breeds the offspring.
    #[inline]
    pub fn reproduction(&self) -> &Reproduction<M::Individual> {
        &self.reproduction
    }

    /// Returns the reproduction that breeds the offspring, to replace its parent selector or
    /// survival policy, or to change its rates.
    #[inline]
    pub fn reproduction_mut(&mut self) -> &mut Reproduction<M::Individual> {
        &mut self.reproduction
    }
}

impl<M: EvolutionModel> EvolutionAlgorithm for GeneticAlgorithm<M> {
    type Model = M;

    #[inline]
    fn model(&self) -> &M {
        &self.model
    }

    #[inline]
    fn population_size(&self) -> usize {
        self.parameters.population_size
    }

    #[inline]
    fn start_run(&mut self) {
        self.model.start_run();
    }

    #[inline]
    fn prepare_generation(&mut self, state: &RunState<M::Individual>) {
        self.model.prepare(state);
    }

    #[inline]
    fn select_elites(
        &self,
        population: &Population<M::Individual>,
        elites: &mut Generation<M::Individual>,
    ) {
        select_fittest(population, self.elite_count, elites);
    }

    #[inline]
    fn generate_offspring(
        &self,
        state: &RunState<M::Individual>,
        offspring: &mut Generation<M::Individual>,
        rng: &mut Rng,
    ) {
        self.reproduction
            .breed_into(&self.model, state, offspring, rng);
    }
}

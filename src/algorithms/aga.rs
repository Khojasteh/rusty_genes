use super::*;

/// Population size, elite carryover, parent-dependent rates, and selection for an adaptive GA.
///
/// Start with [`Default::default`] and modify fields before passing them to
/// [`AdaptiveGeneticAlgorithm::new`], which validates them. Stopping belongs to the model's goal
/// and the progress callback supplied to [`Evolution`].
///
/// The rates are the constants `k1` to `k4` of Srinivas and Patnaik, and their defaults are the
/// values they chose. See [`AdaptiveGeneticAlgorithm`] for how the rates of a candidate follow
/// from them.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AdaptiveGeneticAlgorithmParameters {
    /// Target number of individuals in a full generation; must be positive.
    ///
    /// A generation may be smaller when survival cannot supply enough valid, distinct genomes.
    pub population_size: usize,

    /// Fraction of `population_size` reserved for the fittest current individuals.
    ///
    /// Must be in `[0, 1)`; the elite count is rounded down. Zero disables explicit elite
    /// carryover, but the survival policy may still retain parents.
    pub elitism_rate: f64,

    /// The crossover rate of two parents the fitter of which has the average score (`k1`).
    ///
    /// `crossover_rate` must be between 0.0 and 1.0, inclusive. The rate of parents the fitter of
    /// which is fitter than the average decreases from it in proportion, to 0.0 for parents that
    /// include a fittest individual of the population.
    pub crossover_rate: f64,

    /// The crossover rate of two parents that are both less fit than the average (`k3`).
    ///
    /// `subaverage_crossover_rate` must be between 0.0 and 1.0, inclusive.
    pub subaverage_crossover_rate: f64,

    /// The mutation rate of a candidate whose fittest parent has the average score (`k2`).
    ///
    /// `mutation_rate` must be between 0.0 and 1.0, inclusive. The rate of a candidate whose
    /// fittest parent is fitter than the average decreases from it in proportion, to 0.0 for a
    /// candidate of a fittest individual of the population.
    pub mutation_rate: f64,

    /// The mutation rate of a candidate whose fittest parent is less fit than the average
    /// (`k4`).
    ///
    /// `subaverage_mutation_rate` must be between 0.0 and 1.0, inclusive.
    pub subaverage_mutation_rate: f64,

    /// The mutation rate of every candidate whose rate would otherwise be lower.
    ///
    /// `minimum_mutation_rate` must be between 0.0 and 1.0, inclusive. It is the default
    /// mutation rate of Srinivas and Patnaik.
    pub minimum_mutation_rate: f64,

    /// The strategy for selecting parents from the population.
    pub selection_strategy: SelectionStrategy,
}

impl Default for AdaptiveGeneticAlgorithmParameters {
    /// Creates an instance of [`AdaptiveGeneticAlgorithmParameters`] with default values.
    ///
    /// # Default Values
    /// * population_size: 100
    /// * elitism_rate: 0.01
    /// * crossover_rate: 1.0
    /// * subaverage_crossover_rate: 1.0
    /// * mutation_rate: 0.5
    /// * subaverage_mutation_rate: 0.5
    /// * minimum_mutation_rate: 0.005
    /// * selection_strategy: [`SelectionStrategy::RouletteWheel`]
    fn default() -> Self {
        Self {
            population_size: 100,
            elitism_rate: 0.01,
            crossover_rate: 1.0,
            subaverage_crossover_rate: 1.0,
            mutation_rate: 0.5,
            subaverage_mutation_rate: 0.5,
            minimum_mutation_rate: 0.005,
            selection_strategy: SelectionStrategy::RouletteWheel,
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
        check_probability("crossover_rate", self.crossover_rate)?;
        check_probability("subaverage_crossover_rate", self.subaverage_crossover_rate)?;
        check_probability("mutation_rate", self.mutation_rate)?;
        check_probability("subaverage_mutation_rate", self.subaverage_mutation_rate)?;
        check_probability("minimum_mutation_rate", self.minimum_mutation_rate)?;
        self.selection_strategy.validate()
    }
}

/// The adaptive genetic algorithm of Srinivas and Patnaik, in which each candidate is bred at
/// rates that depend on the scores of its parents.
///
/// For a candidate whose fitter parent has the score `f`, in a population whose fittest score is
/// `fmax` and average score is `favg`:
///
/// * Its crossover rate is `k1 * (fmax - f) / (fmax - favg)` if `f` is at least as fit as
///   `favg`, and `k3` otherwise.
/// * Its mutation rate is `k2 * (fmax - f) / (fmax - favg)` if `f` is at least as fit as `favg`,
///   and `k4` otherwise, but at least the minimum mutation rate.
///
/// The constants are the rates of [`AdaptiveGeneticAlgorithmParameters`]. Both rates are 0.0
/// for the fittest individuals, apart from the minimum mutation rate. If all scores are equal,
/// crossover is disabled and mutation uses that minimum. For minimization, the distances are
/// taken towards the lowest score instead.
///
/// A candidate that does not cross over is a copy of its first parent. The score that sets its
/// mutation rate is that of its fittest parent. Each probability applies to the model's operator
/// once per candidate; it does not prescribe a separate probability for every gene.
///
/// The algorithm breeds its offspring by a [`Reproduction`], whose parent selector and survival
/// policy can be replaced through
/// [`reproduction_mut`](AdaptiveGeneticAlgorithm::reproduction_mut).
///
/// The algorithm operates on an [`EvolutionModel`], which defines the problem-specific
/// components such as the crossover and mutation operators, and the score function.
///
/// # References
/// M. Srinivas and L. M. Patnaik, "Adaptive probabilities of crossover and mutation in genetic
/// algorithms", IEEE Transactions on Systems, Man, and Cybernetics, 24(4), pp. 656-667, 1994.
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
/// let mut parameters = AdaptiveGeneticAlgorithmParameters::default();
/// parameters.selection_strategy = SelectionStrategy::Tournament(4);
/// let mut evolution =
///     AdaptiveGeneticAlgorithm::new(OneMaxEvolutionModel, parameters)?.into_evolution();
///
/// let limits = Limits::new().generations(200);
/// let solution = evolution
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
        Ok(Self {
            model,
            elite_count: elite_count(parameters.population_size, parameters.elitism_rate),
            reproduction: Reproduction::from_strategy(parameters.selection_strategy)?,
            parameters,
        })
    }

    /// Returns the parameters the algorithm was created with.
    #[inline]
    pub fn parameters(&self) -> &AdaptiveGeneticAlgorithmParameters {
        &self.parameters
    }

    /// Returns the reproduction that breeds the offspring.
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

    fn generate_offspring(
        &self,
        state: &RunState<M::Individual>,
        offspring: &mut Generation<M::Individual>,
        rng: &mut Rng,
    ) {
        let rates = FitnessRates::new(&self.parameters, state.population());
        self.reproduction
            .breed_into(&self.model, state, offspring, &rates, rng);
    }
}

/// The rates at which the adaptive genetic algorithm breeds the candidates of a population.
struct FitnessRates {
    objective: Objective,
    fittest: f64,
    average: f64,
    crossover: f64,
    subaverage_crossover: f64,
    mutation: f64,
    subaverage_mutation: f64,
    minimum_mutation: f64,
}

impl FitnessRates {
    fn new<I: Individual>(
        parameters: &AdaptiveGeneticAlgorithmParameters,
        population: &Population<I>,
    ) -> Self {
        let stats = population.stats();
        Self {
            objective: population.objective(),
            fittest: population.fittest().score(),
            // When every score is equal, rounding could put the average apart from them.
            average: stats.avg().clamp(stats.min(), stats.max()),
            crossover: parameters.crossover_rate,
            subaverage_crossover: parameters.subaverage_crossover_rate,
            mutation: parameters.mutation_rate,
            subaverage_mutation: parameters.subaverage_mutation_rate,
            minimum_mutation: parameters.minimum_mutation_rate,
        }
    }

    /// Returns the rate of a parent with `score`: `subaverage_rate` if it is less fit than the
    /// average, and otherwise `rate` scaled by its distance to the fittest score, relative to the
    /// distance of the average score to it.
    fn rate(&self, score: f64, rate: f64, subaverage_rate: f64) -> f64 {
        if !self.objective.is_at_least_as_fit(score, self.average) {
            return subaverage_rate;
        }
        let distance = (self.fittest - score).abs();
        let average_distance = (self.fittest - self.average).abs();
        // Scale overflowing differences together, leaving subnormal differences intact.
        let (distance, average_distance) = if average_distance.is_infinite() {
            (
                (self.fittest / 2.0 - score / 2.0).abs(),
                (self.fittest / 2.0 - self.average / 2.0).abs(),
            )
        } else {
            (distance, average_distance)
        };
        if average_distance == 0.0 {
            // Every score is the fittest one.
            0.0
        } else {
            rate * (distance / average_distance).min(1.0)
        }
    }
}

impl<I: Individual> BreedingRates<I> for FitnessRates {
    fn crossover_rate(&self, parent1: &I, parent2: &I) -> f64 {
        let (score1, score2) = (parent1.score(), parent2.score());
        let fitter = if self.objective.is_fitter(score2, score1) {
            score2
        } else {
            score1
        };
        self.rate(fitter, self.crossover, self.subaverage_crossover)
    }

    fn mutation_rate(&self, parent: &I) -> f64 {
        self.rate(parent.score(), self.mutation, self.subaverage_mutation)
            .max(self.minimum_mutation)
    }
}

use super::*;

/// Population size, elite carryover, diversity thresholds, and parent selection for a DGGA.
///
/// Start with [`Default::default`] and modify fields before passing them to
/// [`DiversityGuidedGeneticAlgorithm::new`], which validates them. Stopping belongs to the
/// model's goal and the progress callback supplied to [`Evolution`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DiversityGuidedGeneticAlgorithmParameters {
    /// Target number of individuals in a full generation; must be positive.
    ///
    /// A generation may be smaller when survival cannot supply enough valid, distinct genomes.
    pub population_size: usize,

    /// Fraction of `population_size` reserved for the fittest current individuals.
    ///
    /// Must be in `[0, 1)`; the elite count is rounded down. Zero disables explicit elite
    /// carryover, but the survival policy may still retain parents.
    pub elitism_rate: f64,

    /// The diversity below which the algorithm switches from exploitation to exploration
    /// (`d_low`).
    ///
    /// `low_diversity` must be between 0.0 and 1.0, inclusive, and must not exceed
    /// `high_diversity`. The default suits real-valued genomes, which can converge on one
    /// point. Several distinct discrete genomes, such as bits or permutations, can remain
    /// above it even after convergence, preventing exploration; see
    /// [discrete genomes](DiversityGuidedGeneticAlgorithm#discrete-genomes).
    pub low_diversity: f64,

    /// The diversity above which the algorithm switches from exploration to exploitation
    /// (`d_high`).
    ///
    /// `high_diversity` must be between 0.0 and 1.0, inclusive.
    pub high_diversity: f64,

    /// The strategy for selecting parents from the population.
    pub selection_strategy: SelectionStrategy,
}

impl Default for DiversityGuidedGeneticAlgorithmParameters {
    /// Creates an instance of [`DiversityGuidedGeneticAlgorithmParameters`] with default values.
    ///
    /// The diversity thresholds are those of Ursem's experiments with the
    /// distance-to-average-point measure on real-valued genomes. For genomes of discrete genes,
    /// raise the low threshold; see
    /// [discrete genomes](DiversityGuidedGeneticAlgorithm#discrete-genomes).
    ///
    /// # Default Values
    /// * population_size: 100
    /// * elitism_rate: 0.01
    /// * low_diversity: 0.000005
    /// * high_diversity: 0.25
    /// * selection_strategy: [`SelectionStrategy::Tournament(2)`](SelectionStrategy::Tournament)
    fn default() -> Self {
        Self {
            population_size: 100,
            elitism_rate: 0.01,
            low_diversity: 5e-6,
            high_diversity: 0.25,
            selection_strategy: SelectionStrategy::Tournament(2),
        }
    }
}

impl DiversityGuidedGeneticAlgorithmParameters {
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
        check_probability("low_diversity", self.low_diversity)?;
        check_probability("high_diversity", self.high_diversity)?;
        if self.low_diversity > self.high_diversity {
            return Err(EvolutionError::invalid_parameter(
                "low_diversity",
                "must not exceed high_diversity",
            ));
        }
        self.selection_strategy.validate()
    }
}

/// A model that measures the genetic diversity of a population, which the
/// [`DiversityGuidedGeneticAlgorithm`] needs to switch between exploration and exploitation.
///
/// The model measures how far apart its genomes lie. The thresholds of the
/// [`DiversityGuidedGeneticAlgorithm`] are on the scale of the
/// distance-to-average-point measure, which [`distance_to_average_point`] computes for genomes
/// whose genes are coordinates, such as real values or bits.
pub trait DiversityModel: EvolutionModel {
    /// Returns the genetic diversity of `population`, from 0.0 for a population of equal
    /// genomes, normally at most 1.0.
    fn diversity(&self, population: &Population<Self::Individual>) -> f64;
}

/// Measures diversity as the average Euclidean distance to the average point, divided by the
/// search space's diagonal length.
///
/// Every point must have the same number of finite coordinates, and cloning `points` must
/// reproduce the same sequence: one pass finds the average and a second measures distances.
/// Supply a positive, finite `diagonal` for the bounding search space. For coordinate bounds
/// `low[j]..=high[j]`, it is `sqrt(sum((high[j] - low[j])^2))`.
///
/// The result is relative to that diagonal, normally in `[0, 1]`. An empty input or identical
/// points give zero. Coordinates are translated and normalized before averaging; squared
/// normalized deviations can still underflow at extremely small spreads relative to the
/// diagonal. This is the distance-to-average-point measure used by the diversity-guided GA.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// // Two opposite corners of the unit square are each half the diagonal from their middle.
/// let points = [[0.0, 0.0], [1.0, 1.0]];
/// let diversity = distance_to_average_point(points.iter().copied(), 2f64.sqrt());
/// assert!((diversity - 0.5).abs() < 1e-12);
/// ```
pub fn distance_to_average_point<P>(points: impl Iterator<Item = P> + Clone, diagonal: f64) -> f64
where
    P: IntoIterator<Item = f64>,
{
    let mut count = 0;
    let mut average = Vec::new();
    for point in points.clone() {
        if count == 0 {
            // Translate from the first point before normalizing, so large offsets and tiny
            // coordinate units do not overflow sums or round away the normalized mean.
            average.extend(point.into_iter().map(|x| (x, 0.0)));
        } else {
            for ((origin, sum), x) in average.iter_mut().zip(point) {
                *sum += (x - *origin) / diagonal;
            }
        }
        count += 1;
    }
    if count == 0 {
        return 0.0;
    }
    let count = count as f64;
    for (_, sum) in &mut average {
        *sum /= count;
    }
    // With a bounding diagonal, normalization bounds deviations before squaring. This avoids
    // overflow at large coordinate scales without the per-coordinate cost of a scaled norm.
    let distances = points
        .map(|point| {
            average
                .iter()
                .zip(point)
                .map(|((origin, mean), x)| ((x - origin) / diagonal - mean).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .sum::<f64>();
    distances / count
}

/// The diversity-guided genetic algorithm of Ursem, which alternates between exploiting the
/// population by crossover and exploring the search space by mutation, as its genetic diversity
/// falls and rises.
///
/// Each run resets to exploitation, then checks the initial population's diversity before
/// breeding. In exploitation, candidates are bred by crossover when two parents are available,
/// and none is mutated. When the diversity of a generation falls below the
/// [low threshold](DiversityGuidedGeneticAlgorithmParameters::low_diversity), it explores:
/// every candidate is a mutated copy of its parent, and none is crossed over. When the diversity
/// rises above the [high threshold](DiversityGuidedGeneticAlgorithmParameters::high_diversity),
/// it exploits again. Between the thresholds, it keeps its current mode.
///
/// The diversity is that of the genomes, which the model measures as a [`DiversityModel`]; the
/// default thresholds are those Ursem used with the distance-to-average-point measure of
/// [`distance_to_average_point`] on real-valued genomes.
///
/// The parent selector and survival policy of its [`Reproduction`], which can be replaced through
/// [`reproduction_mut`](DiversityGuidedGeneticAlgorithm::reproduction_mut), apply in both
/// phases, unlike in Ursem's algorithm, whose exploration applies no selection.
///
/// # Discrete genomes
/// Populations remove duplicate genomes. Several distinct real-valued genomes can lie
/// very close together; distinct bit strings or permutations remain separated by
/// discrete changes. Their measured diversity can therefore stay above the default low
/// threshold of 0.000005 even after the search has stopped improving. Crossover alone may then
/// fail to produce new solutions. A population reduced to a single genome has zero diversity.
///
/// Choose thresholds on the scale of the model's measure and genome length. For example,
/// [`distance_to_average_point`] for two `n`-bit genomes differing in one bit is
/// `1 / (2 * sqrt(n))`: 0.05 for 100 bits. Larger populations of nearby bit strings can have
/// greater diversity; the example below uses a low threshold of 0.15. Observe the diversity of
/// near-converged populations when choosing a threshold for another problem or measure.
///
/// # References
/// R. K. Ursem, "Diversity-guided evolutionary algorithms", Parallel Problem Solving from
/// Nature VII, Lecture Notes in Computer Science 2439, pp. 462-471, 2002.
///
/// # Example
/// Runs the algorithm on the model of the [OneMax example], measuring the diversity of its bit
/// strings by [`distance_to_average_point`].
/// ```
/// # mod one_max { include!("../../examples/one_max.rs"); }
/// # use one_max::*;
/// use rusty_genes::*;
///
/// // Each bit of a genome is a coordinate between 0 and 1.
/// impl DiversityModel for OneMaxEvolutionModel {
///     fn diversity(&self, population: &Population<OneMax>) -> f64 {
///         let length = population[0].genome().len();
///         let bits = population.iter().map(|individual| {
///             individual.genome().iter().map(|&bit| f64::from(u8::from(bit)))
///         });
///         distance_to_average_point(bits, (length as f64).sqrt())
///     }
/// }
///
/// const ONE_MAX_LENGTH: usize = 100;
///
/// let mut parameters = DiversityGuidedGeneticAlgorithmParameters::default();
/// parameters.low_diversity = 0.15;
/// parameters.selection_strategy = SelectionStrategy::Tournament(4);
/// let mut evolution =
///     DiversityGuidedGeneticAlgorithm::new(OneMaxEvolutionModel, parameters)?.into_evolution();
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
pub struct DiversityGuidedGeneticAlgorithm<M: DiversityModel> {
    model: M,
    parameters: DiversityGuidedGeneticAlgorithmParameters,
    elite_count: usize,
    exploring: bool,
    reproduction: Reproduction<M::Individual>,
}

/// The rates of the exploitation phase, which enables crossover and disables mutation.
const EXPLOITATION: FixedRates = FixedRates::constant(1.0, 0.0);

/// The rates of the exploration phase, which mutates every candidate.
const EXPLORATION: FixedRates = FixedRates::constant(0.0, 1.0);

impl<M: DiversityModel> DiversityGuidedGeneticAlgorithm<M> {
    /// Creates a diversity-guided genetic algorithm that operates on `model` with the given
    /// parameters.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if a parameter is invalid.
    pub fn new(
        model: M,
        parameters: DiversityGuidedGeneticAlgorithmParameters,
    ) -> Result<Self, EvolutionError> {
        parameters.validate()?;
        Ok(Self {
            model,
            elite_count: elite_count(parameters.population_size, parameters.elitism_rate),
            exploring: false,
            reproduction: Reproduction::from_strategy(parameters.selection_strategy)?,
            parameters,
        })
    }

    /// Returns the parameters the algorithm was created with.
    #[inline]
    pub fn parameters(&self) -> &DiversityGuidedGeneticAlgorithmParameters {
        &self.parameters
    }

    /// Returns whether the algorithm explores the search space by mutation in the current
    /// generation, rather than exploiting the population by crossover.
    #[inline]
    pub fn is_exploring(&self) -> bool {
        self.exploring
    }

    /// Returns the rates at which the offspring of the current generation are bred.
    #[inline]
    pub fn rates(&self) -> &FixedRates {
        if self.exploring {
            &EXPLORATION
        } else {
            &EXPLOITATION
        }
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

impl<M: DiversityModel> EvolutionAlgorithm for DiversityGuidedGeneticAlgorithm<M> {
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
        self.exploring = false;
        self.model.start_run();
    }

    fn prepare_generation(&mut self, state: &RunState<M::Individual>) {
        let diversity = self.model.diversity(state.population());
        if self.exploring && diversity > self.parameters.high_diversity {
            self.exploring = false;
        } else if !self.exploring && diversity < self.parameters.low_diversity {
            self.exploring = true;
        }

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
            .breed_into(&self.model, state, offspring, self.rates(), rng);
    }
}

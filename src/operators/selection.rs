use super::*;

/// Represents various selection strategies for selecting an individual from the population in
/// an evolutionary algorithm.
///
/// # Note
/// The selection strategy is an important aspect of an evolutionary algorithm. It determines how
/// individuals are chosen from the population for reproduction and thereby, influences the diversity
/// and convergence speed of the algorithm.
///
/// Each selection strategy has its own strengths and weaknesses, and the choice of strategy can
/// significantly impact the performance of the evolutionary algorithm.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SelectionStrategy {
    /// The tournament selection strategy runs a "tournament" among a few individuals chosen at
    /// random from the population and selects the winner (the one with the best score). The
    /// parameter is the number of individuals in a tournament. See [`TournamentSelection`].
    Tournament(usize /* size */),

    /// Selects proportionally to score-derived weights, accounting for negative scores and
    /// minimization. See [`RouletteWheelSelection`].
    RouletteWheel,

    /// The Boltzmann selection strategy selects individuals with a probability that is a function
    /// of their score and a decreasing "temperature". The first parameter is the initial
    /// temperature, and the second one is the cooling rate. See [`BoltzmannSelection`].
    Boltzmann(f64 /* temperature */, f64 /* cooling rate */),

    /// The rank selection strategy selects individuals with a probability proportional to their
    /// rank in the population sorted from worst to best. See [`RankSelection`].
    Rank,

    /// The linear selection strategy selects individuals with a probability that is a linear
    /// function of their rank. The parameter is the selective pressure. See [`LinearSelection`].
    Linear(f64 /* selective pressure */),

    /// The elitist selection strategy selects uniformly among a top portion of the population.
    /// The parameter is the ratio of the population that is excluded from the selection. See
    /// [`ElitistSelection`].
    Elitist(f64 /* ratio */),
}

impl SelectionStrategy {
    /// Validates the parameters of the strategy.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if a parameter of the strategy is invalid.
    pub fn validate(&self) -> Result<(), EvolutionError> {
        match *self {
            Self::Tournament(size) => TournamentSelection::new(size).map(drop),
            Self::RouletteWheel | Self::Rank => Ok(()),
            Self::Boltzmann(temperature, cooling_rate) => {
                BoltzmannSelection::new(temperature, cooling_rate).map(drop)
            }
            Self::Linear(pressure) => LinearSelection::new(pressure).map(drop),
            Self::Elitist(ratio) => ElitistSelection::new(ratio).map(drop),
        }
    }

    /// Creates the selection mechanism of the strategy.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if a parameter of the strategy is invalid.
    pub fn create_selector<I: Individual>(
        &self,
    ) -> Result<Box<dyn SelectionMechanism<I>>, EvolutionError> {
        Ok(match *self {
            Self::Tournament(size) => Box::new(TournamentSelection::new(size)?),
            Self::RouletteWheel => Box::new(RouletteWheelSelection::new()),
            Self::Rank => Box::new(RankSelection::new()),
            Self::Linear(pressure) => Box::new(LinearSelection::new(pressure)?),
            Self::Elitist(ratio) => Box::new(ElitistSelection::new(ratio)?),
            Self::Boltzmann(temperature, cooling_rate) => {
                Box::new(BoltzmannSelection::new(temperature, cooling_rate)?)
            }
        })
    }
}

/// Defines the mechanism for selecting an individual from the population in an evolutionary
/// algorithm.
///
/// The mechanism is shared by the threads that breed offspring in parallel, and it is part of
/// algorithms that may be moved to other threads, so it must be [`Send`] and [`Sync`].
/// Populations are ordered from worst to best. A custom mechanism comparing numerical scores
/// directly must respect [`Population::objective`].
pub trait SelectionMechanism<I: Individual>: Send + Sync {
    /// Restores the state the mechanism adapts during a run, for a new run.
    ///
    /// # Note
    /// The algorithms of this library call this method when a run starts. The default
    /// implementation performs no operation.
    fn reset(&mut self) {}

    /// Prepares the mechanism for selecting individuals from a population.
    ///
    /// # Arguments
    /// * `population` - A reference to the population from which individuals will be selected.
    ///
    /// # Note
    /// This method must be called once for each new population before selecting individuals
    /// from it. It can be used to perform any necessary pre-processing.
    fn prepare(&mut self, population: &Population<I>);

    /// Selects an individual from the population.
    ///
    /// # Arguments
    /// * `population` - A reference to the population from which an individual will be
    ///   selected. It must be the population the mechanism was last prepared for.
    /// * `rng` - A random number generator for stochastic selection strategies.
    ///
    /// # Returns
    /// The index and a reference of the selected individual.
    fn select<'a>(&self, population: &'a Population<I>, rng: &mut Rng) -> (usize, &'a I);

    /// Selects an individual from the population, excluding a specific individual.
    /// The selection attempt will be repeated for a specified number of tries (`max_tries`),
    /// or until a different individual is found.
    ///
    /// # Arguments
    /// * `population` - A reference to the population from which an individual will be selected.
    /// * `rng` - A random number generator for stochastic selection strategies.
    /// * `max_tries` - The maximum number of selection attempts before giving up.
    /// * `excluded` - The individual to be excluded from selection.
    ///
    /// # Returns
    /// Returns a tuple containing the index and a reference to the selected individual distinct
    /// from the `excluded` one. If no different individual is found after `max_tries` attempts,
    /// the method returns [`None`].
    #[inline]
    fn select_distinct<'a>(
        &self,
        population: &'a Population<I>,
        rng: &mut Rng,
        max_tries: usize,
        excluded: &I,
    ) -> Option<(usize, &'a I)> {
        for _ in 0..max_tries {
            let (index, candidate) = self.select(population, rng);
            if candidate != excluded {
                return Some((index, candidate));
            }
        }
        None
    }
}

/// Samples indices of a population uniformly from a range of indices.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct UniformIndex {
    population_len: usize,
    indices: Option<UniformIndices>,
}

impl UniformIndex {
    /// Prepares for sampling indices from `start..population_len`.
    fn prepare(&mut self, start: usize, population_len: usize) {
        self.population_len = population_len;
        self.indices = UniformIndices::new(start..population_len);
    }

    #[inline]
    fn sample(&self, population_len: usize, rng: &mut Rng) -> usize {
        match self.indices {
            Some(indices) if population_len == self.population_len => indices.sample(rng),
            _ => panic!("the selection mechanism is not prepared for the population"),
        }
    }
}

/// Samples indices with probabilities proportional to non-negative weights.
///
/// Weights that are negative or not finite count as zero. If no weight is positive, every index
/// is equally likely.
#[derive(Debug, Clone, Default, PartialEq)]
struct WeightedIndex {
    cumulative_weights: Vec<f64>,
}

impl WeightedIndex {
    fn rebuild(&mut self, weights: impl Iterator<Item = f64>) {
        self.cumulative_weights.clear();
        let mut largest: f64 = 0.0;
        for weight in weights {
            let weight = if weight > 0.0 && weight.is_finite() {
                weight
            } else {
                0.0
            };
            largest = largest.max(weight);
            self.cumulative_weights.push(weight);
        }
        if largest > 0.0 {
            let mut total = 0.0;
            for weight in &mut self.cumulative_weights {
                total += *weight / largest;
                *weight = total;
            }
        }
    }

    fn sample(&self, len: usize, rng: &mut Rng) -> usize {
        assert!(
            self.cumulative_weights.len() == len,
            "the selection mechanism is not prepared for the population"
        );

        let total = self.cumulative_weights.last().copied().unwrap_or(0.0);
        if total > 0.0 && total.is_finite() {
            let target = rng.random_f64() * total;
            let index = self.cumulative_weights.partition_point(|&c| c <= target);
            index.min(len - 1)
        } else {
            rng.random_range(0..len)
        }
    }
}

/// Represents the tournament selection algorithm.
///
/// # Note
/// The tournament selection method draws a fixed number of random candidates from the
/// population, with replacement, and selects the best candidate under its objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TournamentSelection {
    size: usize,
    indices: UniformIndex,
}

impl TournamentSelection {
    /// Creates a new instance with the given tournament size.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if `size` is zero.
    pub fn new(size: usize) -> Result<Self, EvolutionError> {
        Self::check_size(size)?;
        Ok(Self {
            size,
            indices: UniformIndex::default(),
        })
    }

    /// Gets the tournament size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Sets the tournament size.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if `size` is zero.
    pub fn set_size(&mut self, size: usize) -> Result<(), EvolutionError> {
        Self::check_size(size)?;
        self.size = size;
        Ok(())
    }

    fn check_size(size: usize) -> Result<(), EvolutionError> {
        if size == 0 {
            return Err(EvolutionError::invalid_parameter(
                "tournament size",
                "must be positive",
            ));
        }
        Ok(())
    }
}

impl<I: Individual> SelectionMechanism<I> for TournamentSelection {
    #[inline]
    fn prepare(&mut self, population: &Population<I>) {
        self.indices.prepare(0, population.len());
    }

    #[inline]
    fn select<'a>(&self, population: &'a Population<I>, rng: &mut Rng) -> (usize, &'a I) {
        // The population is sorted from worst to best, so the highest index wins the tournament.
        let index = (0..self.size)
            .map(|_| self.indices.sample(population.len(), rng))
            .max()
            .unwrap();
        (index, &population[index])
    }
}

/// Selects proportionally to non-negative weights derived from scores and the objective.
///
/// Score-proportionate selection is defined for non-negative scores only. If the
/// population has a negative score, the scores are shifted by the lowest one, so the
/// probabilities become proportional to the distance from the least fit individual, which is
/// never selected unless all individuals have the same score. If all weights are zero, every
/// individual is equally likely to be selected.
/// Differences are formed before scaling to preserve nearby scores. When their range would
/// overflow, scores are halved first. Weights are scaled before summing to avoid overflow.
/// For minimization the same rule is applied to the negated scores internally: weights are
/// proportional to `max(0, maximum score) - score`. Stored scores are unchanged.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct RouletteWheelSelection {
    weights: WeightedIndex,
}

impl RouletteWheelSelection {
    /// Creates a new instance.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<I: Individual> SelectionMechanism<I> for RouletteWheelSelection {
    #[inline]
    fn prepare(&mut self, population: &Population<I>) {
        let objective = population.objective();
        let offset = objective.as_maximized(population[0].score()).min(0.0);
        let best = objective.as_maximized(population.fittest().score());
        let factor = if (best - offset).is_infinite() {
            0.5
        } else {
            1.0
        };
        self.weights.rebuild(
            population
                .iter()
                .map(|i| objective.as_maximized(i.score()) * factor - offset * factor),
        );
    }

    #[inline]
    fn select<'a>(&self, population: &'a Population<I>, rng: &mut Rng) -> (usize, &'a I) {
        let index = self.weights.sample(population.len(), rng);
        (index, &population[index])
    }
}

/// Represents the Boltzmann selection algorithm used in genetic algorithms.
///
/// # Note
/// In Boltzmann selection, the probability of selecting an individual with score `f` is
/// proportional to `exp(f / T)`, where `T` is the temperature. A high temperature makes the
/// probabilities more uniform, which encourages exploration, while a low temperature favors the
/// fittest individuals, which encourages exploitation.
/// For minimization probabilities are proportional to `exp(-f / T)`, using the score in its
/// original units. The temperature therefore has the same units as the score.
///
/// The first population of a run is selected from at the initial temperature, and the
/// temperature is multiplied by the cooling rate after each population, bounded to the
/// positive finite `f64` range. The weights are computed relative to the fittest
/// individual, so they cannot overflow at any temperature.
#[derive(Debug, Clone, PartialEq)]
pub struct BoltzmannSelection {
    initial_temperature: f64,
    temperature: f64,
    cooling_rate: f64,
    weights: WeightedIndex,
}

impl BoltzmannSelection {
    /// Creates a new instance.
    ///
    /// # Arguments
    /// * `temperature` - The initial temperature, which must be positive and finite.
    /// * `cooling_rate` - The factor the temperature is multiplied by after each population,
    ///   which must be positive and finite. A value less than 1.0 cools the temperature down.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if a parameter is invalid.
    pub fn new(temperature: f64, cooling_rate: f64) -> Result<Self, EvolutionError> {
        Self::check_positive("temperature", temperature)?;
        Self::check_positive("cooling rate", cooling_rate)?;
        Ok(Self {
            initial_temperature: temperature,
            temperature,
            cooling_rate,
            weights: WeightedIndex::default(),
        })
    }

    /// Gets the initial temperature, which [`reset`](SelectionMechanism::reset) restores for a
    /// new run.
    pub fn initial_temperature(&self) -> f64 {
        self.initial_temperature
    }

    /// Gets the temperature for the next population.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Sets the temperature for the next population, without changing the initial temperature.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if `temperature` is not positive and finite.
    pub fn set_temperature(&mut self, temperature: f64) -> Result<(), EvolutionError> {
        Self::check_positive("temperature", temperature)?;
        self.temperature = temperature;
        Ok(())
    }

    /// Gets the cooling rate.
    pub fn cooling_rate(&self) -> f64 {
        self.cooling_rate
    }

    /// Sets the cooling rate.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if `cooling_rate` is not positive and finite.
    pub fn set_cooling_rate(&mut self, cooling_rate: f64) -> Result<(), EvolutionError> {
        Self::check_positive("cooling rate", cooling_rate)?;
        self.cooling_rate = cooling_rate;
        Ok(())
    }

    fn check_positive(name: &'static str, value: f64) -> Result<(), EvolutionError> {
        if value > 0.0 && value.is_finite() {
            Ok(())
        } else {
            Err(EvolutionError::invalid_parameter(
                name,
                "must be positive and finite",
            ))
        }
    }
}

impl<I: Individual> SelectionMechanism<I> for BoltzmannSelection {
    #[inline]
    fn reset(&mut self) {
        self.temperature = self.initial_temperature;
    }

    #[inline]
    fn prepare(&mut self, population: &Population<I>) {
        // Subtracting the best score gives non-positive exponents in either direction.
        let temperature = self.temperature;
        let objective = population.objective();
        let best = objective.as_maximized(population.fittest().score());
        self.weights.rebuild(population.iter().map(|i| {
            let score = objective.as_maximized(i.score());
            let difference = score - best;
            let exponent = if difference.is_finite() {
                difference / temperature
            } else {
                score / temperature - best / temperature
            };
            exponent.exp()
        }));
        self.temperature =
            (self.temperature * self.cooling_rate).clamp(f64::from_bits(1), f64::MAX);
    }

    #[inline]
    fn select<'a>(&self, population: &'a Population<I>, rng: &mut Rng) -> (usize, &'a I) {
        let index = self.weights.sample(population.len(), rng);
        (index, &population[index])
    }
}

/// Represents the rank selection algorithm used in genetic algorithms.
///
/// # Note
/// In rank selection, the probability of selecting an individual is proportional to its rank
/// in the population sorted by score, where the least fit individual has rank 1 and the
/// fittest has rank `n`.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct RankSelection {
    weights: WeightedIndex,
}

impl RankSelection {
    /// Creates a new instance.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<I: Individual> SelectionMechanism<I> for RankSelection {
    #[inline]
    fn prepare(&mut self, population: &Population<I>) {
        // The population is sorted from worst to best.
        self.weights
            .rebuild((1..=population.len()).map(|rank| rank as f64));
    }

    #[inline]
    fn select<'a>(&self, population: &'a Population<I>, rng: &mut Rng) -> (usize, &'a I) {
        let index = self.weights.sample(population.len(), rng);
        (index, &population[index])
    }
}

/// Represents the elitist selection algorithm used in genetic algorithms.
///
/// # Note
/// Elitist selection excludes the least fit portion of the population, given by the ratio, and
/// selects uniformly among the remaining, fittest individuals. At least the fittest individual
/// always remains selectable.
///
/// Note that using a high ratio in elitist selection may lead to premature convergence and a
/// loss of population diversity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElitistSelection {
    ratio: f64,
    indices: UniformIndex,
}

impl ElitistSelection {
    /// Creates a new instance.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if `ratio` is not between 0.0 (inclusive)
    /// and 1.0 (exclusive).
    pub fn new(ratio: f64) -> Result<Self, EvolutionError> {
        check_ratio("elitist selection ratio", ratio)?;
        Ok(Self {
            ratio,
            indices: UniformIndex::default(),
        })
    }

    /// Gets the ratio of the population that is excluded from the selection.
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// Sets the ratio of the population that is excluded from the selection, which takes
    /// effect from the next population.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if `ratio` is not between 0.0 (inclusive)
    /// and 1.0 (exclusive).
    pub fn set_ratio(&mut self, ratio: f64) -> Result<(), EvolutionError> {
        check_ratio("elitist selection ratio", ratio)?;
        self.ratio = ratio;
        Ok(())
    }
}

impl<I: Individual> SelectionMechanism<I> for ElitistSelection {
    #[inline]
    fn prepare(&mut self, population: &Population<I>) {
        let selection_bound = interpolate_usize(0, population.len(), self.ratio);
        self.indices.prepare(selection_bound, population.len());
    }

    #[inline]
    fn select<'a>(&self, population: &'a Population<I>, rng: &mut Rng) -> (usize, &'a I) {
        let index = self.indices.sample(population.len(), rng);
        (index, &population[index])
    }
}

/// Represents the linear ranking selection algorithm used in genetic algorithms.
///
/// # Note
/// Linear selection is a variant of rank selection in which the selection probability is a
/// linear function of the rank, controlled by the selective pressure `s` between 1.0 and 2.0,
/// inclusive. The fittest individual is `s` times as likely to be selected as an average
/// individual, and the least fit one is `2 - s` times as likely. A pressure of 1.0 selects
/// uniformly, and higher values result in stronger selection pressure.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearSelection {
    pressure: f64,
    weights: WeightedIndex,
}

impl LinearSelection {
    /// Creates a new instance with the given selective pressure.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if `pressure` is not between 1.0 and 2.0,
    /// inclusive.
    pub fn new(pressure: f64) -> Result<Self, EvolutionError> {
        Self::check_pressure(pressure)?;
        Ok(Self {
            pressure,
            weights: WeightedIndex::default(),
        })
    }

    /// Gets the selective pressure.
    pub fn pressure(&self) -> f64 {
        self.pressure
    }

    /// Sets the selective pressure, which takes effect from the next population.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if `pressure` is not between 1.0 and 2.0,
    /// inclusive.
    pub fn set_pressure(&mut self, pressure: f64) -> Result<(), EvolutionError> {
        Self::check_pressure(pressure)?;
        self.pressure = pressure;
        Ok(())
    }

    fn check_pressure(pressure: f64) -> Result<(), EvolutionError> {
        if (1.0..=2.0).contains(&pressure) {
            Ok(())
        } else {
            Err(EvolutionError::invalid_parameter(
                "selective pressure",
                "must be between 1.0 and 2.0, inclusive",
            ))
        }
    }
}

impl<I: Individual> SelectionMechanism<I> for LinearSelection {
    #[inline]
    fn prepare(&mut self, population: &Population<I>) {
        // The population is sorted from worst to best.
        let n = population.len();
        let last = n.saturating_sub(1).max(1) as f64;
        let s = self.pressure;
        self.weights
            .rebuild((0..n).map(|i| (2.0 - s) + 2.0 * (s - 1.0) * i as f64 / last));
    }

    #[inline]
    fn select<'a>(&self, population: &'a Population<I>, rng: &mut Rng) -> (usize, &'a I) {
        let index = self.weights.sample(population.len(), rng);
        (index, &population[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct MockIndividual {
        score: f64,
    }

    impl Hash for MockIndividual {
        fn hash<H: Hasher>(&self, state: &mut H) {
            hash_f64(self.score, state);
        }
    }

    impl Individual for MockIndividual {
        fn score(&self) -> f64 {
            self.score
        }
    }

    const POPULATION_SIZE: usize = 10;
    const ITERATIONS: usize = 100000;
    const CONFIDENCE: f64 = 0.01;

    fn create_population(score: impl Fn(usize) -> f64) -> Population<MockIndividual> {
        (0..POPULATION_SIZE)
            .map(|i| MockIndividual { score: score(i) })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn create_test_population() -> Population<MockIndividual> {
        create_population(|i| i as f64 / POPULATION_SIZE as f64)
    }

    fn observed_probabilities<S>(
        strategy: &mut S,
        population: &Population<MockIndividual>,
    ) -> Vec<f64>
    where
        S: SelectionMechanism<MockIndividual>,
    {
        let mut rng = Rng::from_seed(0);
        let mut counts = vec![0; population.len()];

        strategy.prepare(population);
        for _ in 0..ITERATIONS {
            let (index, _) = strategy.select(population, &mut rng);
            counts[index] += 1;
        }
        counts
            .into_iter()
            .map(|count| count as f64 / ITERATIONS as f64)
            .collect()
    }

    fn assert_probabilities(observed: &[f64], expected: impl Fn(usize) -> f64) {
        for (index, &observed) in observed.iter().enumerate() {
            let expected = expected(index);
            assert!(
                (observed - expected).abs() < CONFIDENCE,
                "index {index}: observed {observed}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_tournament_selection_probabilities() {
        let mut tournament = TournamentSelection::new(2).unwrap();
        let population = create_test_population();
        let observed = observed_probabilities(&mut tournament, &population);

        let n = population.len() as f64;
        let size = tournament.size() as i32;
        assert_probabilities(&observed, |index| {
            (1.0 - (index as f64 / n).powi(size)) - (1.0 - ((index as f64 + 1.0) / n).powi(size))
        });
    }

    #[test]
    fn test_roulette_wheel_selection_probabilities() {
        let mut roulette_wheel = RouletteWheelSelection::new();
        let population = create_test_population();
        let observed = observed_probabilities(&mut roulette_wheel, &population);

        let total_score = population.iter().map(|i| i.score).sum::<f64>();
        assert_probabilities(&observed, |index| population[index].score / total_score);
    }

    #[test]
    fn test_roulette_wheel_selection_with_negative_score() {
        let mut roulette_wheel = RouletteWheelSelection::new();
        let population = create_population(|i| i as f64 - 20.0);
        let observed = observed_probabilities(&mut roulette_wheel, &population);

        // The scores are shifted by the lowest one, -20.
        let total_shifted = (0..POPULATION_SIZE).sum::<usize>() as f64;
        assert_probabilities(&observed, |index| index as f64 / total_shifted);
    }

    #[test]
    fn test_roulette_wheel_selection_with_zero_score() {
        let mut roulette_wheel = RouletteWheelSelection::new();
        let population: Population<MockIndividual> =
            vec![MockIndividual { score: 0.0 }].try_into().unwrap();
        let observed = observed_probabilities(&mut roulette_wheel, &population);

        assert_probabilities(&observed, |_| 1.0);
    }

    #[test]
    fn roulette_preserves_probabilities_at_large_score_scales() {
        for (scores, expected) in [
            (vec![-1e308, 0.0, 1e308], vec![0.0, 1.0 / 3.0, 2.0 / 3.0]),
            (vec![8e307, 1.6e308], vec![1.0 / 3.0, 2.0 / 3.0]),
            (vec![0.0, 1e-308, 2e-308], vec![0.0, 1.0 / 3.0, 2.0 / 3.0]),
        ] {
            let population = Population::new(
                scores
                    .into_iter()
                    .map(|score| MockIndividual { score })
                    .collect(),
            )
            .unwrap();
            let observed = observed_probabilities(&mut RouletteWheelSelection::new(), &population);
            assert_probabilities(&observed, |index| expected[index]);
        }
    }

    #[test]
    fn test_rank_selection_probabilities() {
        let mut rank = RankSelection::new();
        let population = create_test_population();
        let observed = observed_probabilities(&mut rank, &population);

        let total_ranks = (population.len() * (population.len() + 1) / 2) as f64;
        assert_probabilities(&observed, |index| (index + 1) as f64 / total_ranks);
    }

    #[test]
    fn test_elitist_selection_probabilities() {
        let mut elitist = ElitistSelection::new(0.7).unwrap();
        let population = create_test_population();
        let observed = observed_probabilities(&mut elitist, &population);

        let selection_bound = (population.len() as f64 * elitist.ratio()).floor() as usize;
        assert_probabilities(&observed, |index| {
            if index >= selection_bound {
                1.0 / (population.len() - selection_bound) as f64
            } else {
                0.0
            }
        });
    }

    #[test]
    fn test_boltzmann_selection_probabilities() {
        // We keep the temperature constant for this test (cooling_rate = 1.0)
        let mut boltzmann = BoltzmannSelection::new(5.0, 1.0).unwrap();
        let population = create_test_population();
        let observed = observed_probabilities(&mut boltzmann, &population);

        let scaled = |index: usize| (population[index].score / 5.0).exp();
        let total_scaled = (0..population.len()).map(scaled).sum::<f64>();
        assert_probabilities(&observed, |index| scaled(index) / total_scaled);
    }

    #[test]
    fn test_boltzmann_selection_favors_fittest_after_cooling() {
        let mut boltzmann = BoltzmannSelection::new(5.0, 0.9).unwrap();
        let population = create_test_population();
        for _ in 0..100 {
            SelectionMechanism::<MockIndividual>::prepare(&mut boltzmann, &population);
        }
        let observed = observed_probabilities(&mut boltzmann, &population);

        // After 100 coolings, the temperature is about 1.3e-4, and the fittest individual is
        // the only one with a non-negligible weight.
        assert_probabilities(&observed, |index| {
            if index == population.len() - 1 {
                1.0
            } else {
                0.0
            }
        });
    }

    #[test]
    fn test_boltzmann_selection_reset_restores_the_initial_temperature() {
        let mut boltzmann = BoltzmannSelection::new(5.0, 0.5).unwrap();
        let population = create_test_population();
        SelectionMechanism::<MockIndividual>::prepare(&mut boltzmann, &population);
        assert_eq!(boltzmann.temperature(), 2.5);

        SelectionMechanism::<MockIndividual>::reset(&mut boltzmann);
        assert_eq!(boltzmann.temperature(), 5.0);
    }

    #[test]
    fn test_linear_selection_probabilities() {
        let n = POPULATION_SIZE as f64;
        for pressure in [1.0, 1.25, 1.5, 2.0] {
            let mut linear = LinearSelection::new(pressure).unwrap();
            let population = create_test_population();
            let observed = observed_probabilities(&mut linear, &population);

            assert_probabilities(&observed, |index| {
                ((2.0 - pressure) + 2.0 * (pressure - 1.0) * index as f64 / (n - 1.0)) / n
            });
        }
    }

    #[test]
    fn test_selection_parameters_are_validated() {
        assert!(TournamentSelection::new(0).is_err());
        assert!(BoltzmannSelection::new(0.0, 0.9).is_err());
        assert!(BoltzmannSelection::new(1.0, f64::INFINITY).is_err());
        assert!(LinearSelection::new(0.9).is_err());
        assert!(LinearSelection::new(2.1).is_err());
        assert!(ElitistSelection::new(1.0).is_err());
        assert!(SelectionStrategy::Linear(f64::NAN).validate().is_err());
        assert!(SelectionStrategy::Rank.validate().is_ok());
    }

    #[test]
    fn roulette_preserves_adjacent_score_differences() {
        let center = 1.5_f64;
        let scores = [
            -center,
            -f64::from_bits(center.to_bits() - 1),
            -f64::from_bits(center.to_bits() - 2),
        ];
        let population = Population::new(
            scores
                .into_iter()
                .map(|score| MockIndividual { score })
                .collect(),
        )
        .unwrap();
        let observed = observed_probabilities(&mut RouletteWheelSelection::new(), &population);
        assert_probabilities(&observed, |index| index as f64 / 3.0);
    }

    #[test]
    fn boltzmann_scales_overflowing_differences_before_exponentiation() {
        let population = Population::new(vec![
            MockIndividual { score: -1e308 },
            MockIndividual { score: 1e308 },
        ])
        .unwrap();
        let observed = observed_probabilities(
            &mut BoltzmannSelection::new(1e308, 1.0).unwrap(),
            &population,
        );
        let lower = (-2.0_f64).exp();
        assert_probabilities(&observed, |index| {
            if index == 0 {
                lower / (1.0 + lower)
            } else {
                1.0 / (1.0 + lower)
            }
        });
    }

    #[test]
    fn boltzmann_preserves_probabilities_in_subnormal_score_units() {
        let unit = f64::from_bits(1);
        let population = Population::new(
            [0.0, unit, 2.0 * unit]
                .into_iter()
                .map(|score| MockIndividual { score })
                .collect(),
        )
        .unwrap();
        let mut boltzmann = BoltzmannSelection::new(unit, 0.5).unwrap();
        for _ in 0..2 {
            let observed = observed_probabilities(&mut boltzmann, &population);
            let total = (-2.0_f64).exp() + (-1.0_f64).exp() + 1.0;
            assert_probabilities(&observed, |index| ((index as f64) - 2.0).exp() / total);
            assert_eq!(boltzmann.temperature(), unit);
        }
    }
}

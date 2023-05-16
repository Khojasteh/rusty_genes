use super::*;
use rand::distributions::Uniform;
use rayon::prelude::*;

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
pub enum SelectionStrategy {
    /// The tournament selection strategy involves running a "tournament" among a few individuals
    /// chosen at random from the population and selecting the winner (the one with the best fitness).
    /// The size parameter represents the number of individuals to be selected for the tournament.
    Tournament(usize /* size */),

    /// The roulette wheel selection strategy involves choosing individuals based on their fitness
    /// proportionate to the total fitness of the population, like spinning a roulette wheel.
    RouletteWheel,

    /// The Boltzmann selection strategy involves selecting individuals with a probability that is
    /// a function of their fitness and a decreasing "temperature" parameter. The first parameter
    /// represents the initial temperature, and the second one represents the cooling rate.
    Boltzmann(f64 /* temperature */, f64 /* cooling rate */),

    /// The rank selection strategy involves selecting individuals based on their rank in the population
    /// when sorted by fitness, rather than their actual fitness values.
    Rank,

    /// The linear selection strategy involves selecting individuals with a probability that is a
    /// linear function of their rank. The bias parameter determines the slope of the function.
    Linear(f64 /* bias */),

    /// The elitist selection strategy involves always selecting a certain ratio of the best individuals
    /// in the population. The ratio parameter determines the proportion of individuals to be selected.
    Elitist(f64 /* ratio */),
}

impl SelectionStrategy {
    /// Creates the selector for the the selected selection strategy.
    pub fn create_selector<I: Individual, R: Rng>(&self) -> Box<dyn SelectionMechanism<I, R>> {
        match self {
            SelectionStrategy::Tournament(size) => Box::new(TournamentSelection::new(*size)),
            SelectionStrategy::RouletteWheel => Box::new(RouletteWheelSelection::new()),
            SelectionStrategy::Rank => Box::new(RankSelection::new()),
            SelectionStrategy::Linear(bias) => Box::new(LinearSelection::new(*bias)),
            SelectionStrategy::Elitist(ratio) => Box::new(ElitistSelection::new(*ratio)),
            SelectionStrategy::Boltzmann(temperature, cooling_rate) => {
                Box::new(BoltzmannSelection::new(*temperature, *cooling_rate))
            }
        }
    }
}

/// Defines the mechanism for selecting an individual from the population in an evolutionary
/// algorithm.
pub trait SelectionMechanism<I, R>: Sync
where
    I: Individual,
    R: Rng,
{
    /// Prepares the mechanism for a new population.
    ///
    /// # Arguments
    /// * `population` - A reference to the population from which an individual will be selected.
    ///
    /// # Note
    /// This method is called once before selecting individuals from a new population. It
    /// can be used to perform any necessary setup or pre-processing steps before selecting
    /// individuals.
    fn prepare(&mut self, population: &Population<I>);

    /// Selects an individual from the population.
    ///
    /// # Arguments
    /// * `population` - A reference to the population from which an individual will be
    ///   selected.
    /// * `rng` - A random number generator for stochastic selection strategies.
    ///
    /// # Note
    /// The method should return a reference to the selected individual. The implementation
    /// should take into account the fitness of the individuals in the population, as well as
    /// any other selection criteria that are part of the selection strategy.
    fn select<'a>(&self, population: &'a Population<I>, rng: &mut R) -> (usize, &'a I);

    /// Selects an individual from the population, excluding a specific individual.
    /// The selection attempt will be repeated for a specified number of tries (`max_tries`),
    /// or until a different individual is found.
    ///
    /// # Arguments
    /// * `population` - A reference to the population from which an individual will be selected.
    /// * `rng` - A random number generator for stochastic selection strategies.
    /// * `max_tries` - The maximum number of selection attempts before giving up.
    ///   This parameter prevents potential infinite loops in cases where all individuals
    ///   are the same.
    /// * `excluded` - The individual to be excluded from selection.
    ///
    /// # Returns
    /// Returns a tuple containing the index and a reference to the selected individual distinct
    /// from the `excluded` one. If no different individual is found after `max_tries` attempts,
    /// the method returns [`None`].
    ///
    /// # Note
    /// This method is especially useful in genetic operations like crossover, where typically
    /// two distinct individuals need to be selected from the population.
    #[inline]
    fn select_distinct<'a>(
        &self,
        population: &'a Population<I>,
        rng: &mut R,
        max_tries: usize,
        excluded: &I,
    ) -> Option<(usize, &'a I)>
    where
        I: Individual,
        R: Rng,
    {
        for _ in 0..max_tries {
            let (index, candidate) = self.select(population, rng);
            if candidate != excluded {
                return Some((index, candidate));
            }
        }
        None
    }
}

/// Represents the tournament selection algorithm.
///
/// # Note
/// The tournament selection method involves selecting a fixed number of random
/// candidates from the population (determined by tournament's size) and choosing
/// the candidate with the highest fitness.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TournamentSelection {
    size: usize,
}

impl TournamentSelection {
    /// Creates a new instance.
    ///
    /// # Panics
    /// The method will panic if `size` is zero.
    pub fn new(size: usize) -> Self {
        assert!(size != 0);
        Self { size }
    }

    /// Gets the tournament size.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Sets the tournament size.
    ///
    /// # Panics
    /// The method will panic if `size` is zero.
    pub fn set_size(&mut self, size: usize) {
        assert!(size != 0);
        self.size = size
    }
}

impl<I, R> SelectionMechanism<I, R> for TournamentSelection
where
    I: Individual,
    R: Rng,
{
    #[inline]
    fn prepare(&mut self, _population: &Population<I>) {}

    #[inline]
    fn select<'a>(&self, population: &'a Population<I>, rng: &mut R) -> (usize, &'a I) {
        rng.sample_iter(Uniform::new(0, population.len()))
            .take(self.size)
            .max() // The population is already sorted by the fitness of individuals
            .map(|i| (i, &population[i]))
            .unwrap()
    }
}

/// Represents the roulette-wheel selection algorithm used in genetic algorithms.
///
/// # Note
/// The roulette wheel selection method calculates the total fitness of the
/// population and chooses a random target value. It iterates over the population,
/// summing up the fitness values, and selects the first individual whose cumulative
/// fitness is greater than or equal to the target value.
#[derive(Default, Clone, Copy, PartialEq)]
pub struct RouletteWheelSelection {
    total_fitness: f64,
}

impl RouletteWheelSelection {
    /// Creates a new instance.
    pub fn new() -> Self {
        Self { total_fitness: 0.0 }
    }
}

impl<I, R> SelectionMechanism<I, R> for RouletteWheelSelection
where
    I: Individual,
    R: Rng,
{
    #[inline]
    fn prepare(&mut self, population: &Population<I>) {
        self.total_fitness = population.par_iter().map(|i| i.fitness()).sum();
    }

    #[inline]
    fn select<'a>(&self, population: &'a Population<I>, rng: &mut R) -> (usize, &'a I) {
        let target_fitness = rng.gen::<f64>() * self.total_fitness;
        let mut cumulative_fitness = 0.0;
        for (index, individual) in population.iter().enumerate() {
            cumulative_fitness += individual.fitness();
            if cumulative_fitness >= target_fitness {
                return (index, individual);
            }
        }
        (population.len() - 1, population.last().unwrap())
    }
}

/// Represents the Boltzmann selection algorithm used in genetic algorithms.
///
/// # Note
/// In Boltzmann selection, the selection probability of an individual is determined by
/// its fitness relative to other individuals in the population and the algorithm's
/// temperature parameter. The Boltzmann distribution from statistical mechanics is used
/// to create a probability distribution for selection.
///
/// This strategy encourages exploration in the early stages and exploitation in the
/// later stages of the algorithm by adjusting the temperature and cooling rate.
#[derive(Clone, Copy, PartialEq)]
pub struct BoltzmannSelection {
    temperature: f64,
    cooling_rate: f64,
    total_scaled_fitness: f64,
}

impl BoltzmannSelection {
    /// Creates a new instance.
    ///
    /// # Panic
    /// The method will panic if `cooling_rate` is zero.
    pub fn new(temperature: f64, cooling_rate: f64) -> Self {
        assert!(cooling_rate != 0.0);
        Self {
            temperature,
            cooling_rate,
            total_scaled_fitness: 0.0,
        }
    }

    /// Gets the current temperature.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Sets the current temperature.
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature
    }

    /// Gets the cooling rate.
    pub fn cooling_rate(&self) -> f64 {
        self.cooling_rate
    }

    /// Sets the cooling rate.
    ///
    /// # Panic
    /// The method will panic if `cooling_rate` is zero.
    pub fn set_cooling_rate(&mut self, cooling_rate: f64) {
        assert!(cooling_rate != 0.0);
        self.cooling_rate = cooling_rate
    }

    /// Calculates the Boltzmann scaled fitness of an individual.
    #[inline]
    fn scaled_fitness<I: Individual>(&self, individual: &I) -> f64 {
        (individual.fitness() / self.temperature).exp()
    }
}

impl<I, R> SelectionMechanism<I, R> for BoltzmannSelection
where
    I: Individual,
    R: Rng,
{
    #[inline]
    fn prepare(&mut self, population: &Population<I>) {
        self.temperature *= self.cooling_rate;
        self.total_scaled_fitness = population
            .par_iter()
            .map(|i| self.scaled_fitness(i))
            .sum();
    }

    #[inline]
    fn select<'a>(&self, population: &'a Population<I>, rng: &mut R) -> (usize, &'a I) {
        let target_fitness = rng.gen::<f64>() * self.total_scaled_fitness;
        let mut cumulative_fitness = 0.0;
        for (index, individual) in population.iter().enumerate() {
            cumulative_fitness += self.scaled_fitness(individual);
            if cumulative_fitness >= target_fitness {
                return (index, individual);
            }
        }
        (population.len() - 1, population.last().unwrap())
    }
}

/// Represents the rank selection algorithm used in genetic algorithms.
///
/// # Note
/// In rank selection, the population is first sorted by fitness. A random target rank is
/// chosen, based on the total ranks of the population. The algorithm iterates over the
/// sorted population, summing up the ranks, and returns the first individual whose
/// cumulative rank is greater than or equal to the target rank.
#[derive(Default, Clone, Copy, PartialEq, Eq)]
pub struct RankSelection {}

impl RankSelection {
    /// Creates a new instance.
    pub fn new() -> Self {
        Self {}
    }
}

impl<I, R> SelectionMechanism<I, R> for RankSelection
where
    I: Individual,
    R: Rng,
{
    #[inline]
    fn prepare(&mut self, _population: &Population<I>) {
        // The population is already sorted by the fitness of individuals
    }

    #[inline]
    fn select<'a>(&self, population: &'a Population<I>, rng: &mut R) -> (usize, &'a I) {
        let n = population.len();
        let total_ranks = n * (n + 1) / 2;
        let target_rank = rng.gen_range(1..=total_ranks);

        let mut cumulative_rank = 0;
        for (index, individual) in population.iter().enumerate() {
            cumulative_rank += index + 1;
            if cumulative_rank >= target_rank {
                return (index, individual);
            }
        }
        (0, population.first().unwrap())
    }
}

/// Represents the elitist selection algorithm used in genetic algorithms.
///
/// # Note
/// Elitist selection directly favors the best-performing individuals in a population,
/// ensuring that top performers have a chance to pass their genetic information to the
/// next generation. This promotes convergence to an optimal solution.
///
/// The algorithm works by selecting a fixed percentage of the best individuals based
/// on their fitness values. This typically involves sorting the population by fitness
/// and selecting the top-ranked individuals.
///
/// Note that using a high selection ratio in elitist selection may lead to premature
/// convergence and a loss of population diversity.
pub struct ElitistSelection {
    ratio: f64,
}

impl ElitistSelection {
    /// Creates a new instance.
    ///
    /// # Panics
    /// The method will panic if `ratio` is not between 0.0 (inclusive)
    /// and 1.0 (exclusive).
    pub fn new(ratio: f64) -> Self {
        assert!((0.0..1.0).contains(&ratio));
        Self { ratio }
    }

    /// Gets the selection ratio.
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// Sets the selection ratio.
    ///
    /// # Panics
    /// The method will panic if `ratio` is not between 0.0 (inclusive)
    /// and 1.0 (exclusive).
    pub fn set_ratio(&mut self, ratio: f64) {
        assert!((0.0..1.0).contains(&ratio));
        self.ratio = ratio
    }
}

impl<I, R> SelectionMechanism<I, R> for ElitistSelection
where
    I: Individual,
    R: Rng,
{
    #[inline]
    fn prepare(&mut self, _population: &Population<I>) {
        // The population is already sorted by the fitness of individuals
    }

    #[inline]
    fn select<'a>(&self, population: &'a Population<I>, rng: &mut R) -> (usize, &'a I) {
        let selection_bound = (population.len() as f64 * self.ratio).floor() as usize;
        let index = rng.gen_range(selection_bound..population.len());
        (index, &population[index])
    }
}

/// Represents the linear selection algorithm used in genetic algorithms.
///
/// # Note
/// Linear selection is a variant of rank-based selection that introduces a bias factor
/// to adjust the selection pressure. Higher bias values result in stronger selection
/// pressure, which means individuals with higher fitness values have a higher chance
/// of being selected.
#[derive(Clone, Copy, PartialEq)]
pub struct LinearSelection {
    bias: f64,
}

impl LinearSelection {
    /// Creates a new instance.
    pub fn new(bias: f64) -> Self {
        Self { bias }
    }

    /// Gets the bias factor.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Sets the bias factor.
    pub fn set_bias(&mut self, bias: f64) {
        self.bias = bias
    }
}

impl<I, R> SelectionMechanism<I, R> for LinearSelection
where
    I: Individual,
    R: Rng,
{
    #[inline]
    fn prepare(&mut self, _population: &Population<I>) {
        // The population is already sorted by the fitness of individuals
    }

    #[inline]
    fn select<'a>(&self, population: &'a Population<I>, rng: &mut R) -> (usize, &'a I) {
        let index = population.len()
            - ((population.len() as f64)
                * (self.bias
                    - ((self.bias * self.bias - 4.0 * (self.bias - 1.0) * rng.gen::<f64>())
                        .sqrt()))
                / 2.0
                / (self.bias - 1.0)) as usize
            - 1;
        (index, &population[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct MockIndividual {
        fitness: f64,
    }

    impl Individual for MockIndividual {
        fn fitness(&self) -> f64 {
            self.fitness
        }
    }

    const POPULATION_SIZE: usize = 10;
    const ITERATIONS: usize = 100000;
    const CONFIDENCE: f64 = 0.01;

    fn create_test_population() -> Population<MockIndividual> {
        (0..POPULATION_SIZE)
            .map(|i| MockIndividual {
                fitness: i as f64 / POPULATION_SIZE as f64,
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn collect_samples<S>(strategy: &mut S) -> (Vec<MockIndividual>, Vec<u32>)
    where
        S: SelectionMechanism<MockIndividual, SmallRng>,
    {
        let mut rng = SmallRng::seed_from_u64(0);
        let population = create_test_population();
        let mut counts = vec![0; POPULATION_SIZE];

        strategy.prepare(&population);
        for _ in 0..ITERATIONS {
            let (index, _) = strategy.select(&population, &mut rng);
            counts[index] += 1;
        }
        (population.into(), counts)
    }

    #[test]
    fn test_tournament_selection_probabilities() {
        let mut tournament = TournamentSelection::new(2);
        let (population, observed_counts) = collect_samples(&mut tournament);

        for (index, count) in observed_counts.into_iter().enumerate() {
            let observed_probability = count as f64 / ITERATIONS as f64;
            let expected_probability = (1.0
                - (index as f64 / population.len() as f64).powi(tournament.size as i32))
                - (1.0
                    - ((index as f64 + 1.0) / population.len() as f64)
                        .powi(tournament.size as i32));

            assert!((observed_probability - expected_probability).abs() < CONFIDENCE);
        }
    }

    #[test]
    fn test_roulette_wheel_selection_probabilities() {
        let mut roulette_wheel = RouletteWheelSelection::new();
        let (population, observed_counts) = collect_samples(&mut roulette_wheel);

        let total_fitness = population.iter().map(|i| i.fitness).sum::<f64>();
        for (index, count) in observed_counts.into_iter().enumerate() {
            let observed_probability = count as f64 / ITERATIONS as f64;
            let expected_probability = population[index].fitness / total_fitness;

            assert!((observed_probability - expected_probability).abs() < CONFIDENCE);
        }
    }

    #[test]
    fn test_rank_selection_probabilities() {
        let mut rank = RankSelection::new();
        let (population, observed_counts) = collect_samples(&mut rank);

        let total_ranks = population.len() * (population.len() + 1) / 2;
        for (index, count) in observed_counts.into_iter().enumerate() {
            let observed_probability = count as f64 / ITERATIONS as f64;
            let expected_probability = (index + 1) as f64 / total_ranks as f64;

            assert!((observed_probability - expected_probability).abs() < CONFIDENCE);
        }
    }

    #[test]
    fn test_elitist_selection_probabilities() {
        let mut elitist = ElitistSelection::new(0.7);
        let (population, observed_counts) = collect_samples(&mut elitist);

        let selection_bound = (population.len() as f64 * elitist.ratio).floor() as usize;
        for (index, count) in observed_counts.into_iter().enumerate() {
            let observed_probability = count as f64 / ITERATIONS as f64;
            let expected_probability = if index >= selection_bound {
                1.0 / (population.len() - selection_bound) as f64
            } else {
                0.0
            };

            assert!((observed_probability - expected_probability).abs() < CONFIDENCE);
        }
    }

    #[test]
    fn test_boltzmann_selection_probabilities() {
        // We keep the temperature constant for this test (cooling_rate = 1.0)
        let mut boltzmann = BoltzmannSelection::new(5.0, 1.0);
        let (population, observed_counts) = collect_samples(&mut boltzmann);

        let total_scaled_fitness = population
            .iter()
            .map(|i| (i.fitness / boltzmann.temperature).exp())
            .sum::<f64>();
        for (index, count) in observed_counts.into_iter().enumerate() {
            let observed_probability = count as f64 / ITERATIONS as f64;
            let expected_probability =
                (population[index].fitness / boltzmann.temperature).exp() / total_scaled_fitness;

            assert!((observed_probability - expected_probability).abs() < CONFIDENCE);
        }
    }

    #[test]
    fn test_linear_selection_probabilities() {
        let mut linear = LinearSelection::new(1.25);
        let (_, observed_counts) = collect_samples(&mut linear);

        let mut last_observed_probability: Option<f64> = None;
        for count in observed_counts {
            let observed_probability = count as f64 / ITERATIONS as f64;
            if let Some(previous_observed_probability) = last_observed_probability {
                assert!(observed_probability + CONFIDENCE > previous_observed_probability);
            }
            last_observed_probability = Some(observed_probability);
        }
    }
}

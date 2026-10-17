use super::*;
use rayon::prelude::*;

/// The `Evolution` struct is the primary user-facing interface for running an evolutionary
/// algorithm.
///
/// It owns an implementation of the [`EvolutionAlgorithm`] trait and runs it generation by
/// generation, starting from a random initial population, from a single individual, or from
/// a given initial population.
///
/// # Reproducibility
/// Without a seed, every run is seeded from the operating system. With a seed, set by
/// [`with_seed`](Evolution::with_seed), every run starts from the same random state, so runs of
/// the same algorithm and model from the same input produce the same result with the same
/// version of this library, on any number of threads. This holds as long as the model and the
/// algorithm are deterministic apart from the random number generators they are given.
///
/// # Type parameters
/// * `A` - The evolutionary algorithm to run.
///
/// # Example
/// ```
/// use rusty_genes::*;
/// use rusty_genes::one_max::*;
///
/// let mut evolution = GeneticAlgorithm::new(OneMaxEvolutionModel, Default::default())?
///     .into_evolution()
///     .with_seed(42);
///
/// let best = evolution.run(&50, |population| population.generation() < 10)?;
/// assert_eq!(best.genome().len(), 50);
/// # Ok::<(), EvolutionError>(())
/// ```
pub struct Evolution<A: EvolutionAlgorithm> {
    algorithm: A,
    seed: Option<u64>,
}

impl<A: EvolutionAlgorithm> Evolution<A> {
    /// Creates an `Evolution` that runs `algorithm`, without a seed.
    #[inline]
    pub fn new(algorithm: A) -> Self {
        Self {
            algorithm,
            seed: None,
        }
    }

    /// Sets the seed of the runs, and returns the `Evolution`.
    #[inline]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Returns the seed of the runs, if any.
    #[inline]
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Sets the seed of the runs, or removes it with [`None`], so that runs are seeded from the
    /// operating system.
    #[inline]
    pub fn set_seed(&mut self, seed: Option<u64>) {
        self.seed = seed;
    }

    /// Returns a reference to the algorithm.
    #[inline]
    pub fn algorithm(&self) -> &A {
        &self.algorithm
    }

    /// Returns a mutable reference to the algorithm.
    #[inline]
    pub fn algorithm_mut(&mut self) -> &mut A {
        &mut self.algorithm
    }

    /// Consumes the `Evolution` and returns the algorithm.
    #[inline]
    pub fn into_algorithm(self) -> A {
        self.algorithm
    }

    /// Runs the evolutionary algorithm with a random initial population.
    ///
    /// # Arguments
    /// * `args` - A reference to the domain-specific arguments for generating the random
    ///   individuals by [`RandomIndividual::new_random`].
    /// * `progress` - A closure that is called before each generation is bred. It receives the
    ///   current population, which provides the generation number, the number of generations
    ///   without improvement (stagnation), and the statistics of the fitness scores. The closure
    ///   returns whether the algorithm should continue ([`true`]) or stop ([`false`]).
    ///
    /// # Returns
    /// The fittest individual of the last generation.
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if no valid individual could be generated.
    pub fn run<T, F>(&mut self, args: &T, progress: F) -> Result<IndividualOf<A>, EvolutionError>
    where
        T: Sync,
        IndividualOf<A>: RandomIndividual<T>,
        F: FnMut(&Population<IndividualOf<A>>) -> bool,
    {
        let mut rng = self.new_rng();
        let streams = rng.split();
        let individuals = (0..self.algorithm.population_size())
            .into_par_iter()
            .map(|index| IndividualOf::<A>::new_random(args, &mut streams.rng(index as u64)))
            .collect();
        self.evolve(individuals, progress, &mut rng)
            .map(Population::take_fittest)
    }

    /// Runs the evolutionary algorithm with an initial population made of `individual` and its
    /// mutated copies.
    ///
    /// # Arguments
    /// * `individual` - The individual to build the initial population from.
    /// * `progress` - A closure that is called before each generation is bred. See
    ///   [`run`](Evolution::run) for details.
    ///
    /// # Returns
    /// The fittest individual of the last generation.
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if the initial population has no valid
    /// individual.
    ///
    /// # Note
    /// This method is intended to be used when a good individual has already been found, so its
    /// genetic information is likely to be valuable in generating high-quality offspring. The
    /// copies are mutated by [`EvolutionModel::mutate`].
    pub fn evolve_individual<F>(
        &mut self,
        individual: IndividualOf<A>,
        progress: F,
    ) -> Result<IndividualOf<A>, EvolutionError>
    where
        F: FnMut(&Population<IndividualOf<A>>) -> bool,
    {
        let mut rng = self.new_rng();
        let streams = rng.split();
        let model = self.algorithm.model();
        let copies = self.algorithm.population_size().saturating_sub(1);
        let mut individuals = Vec::with_capacity(copies + 1);
        individuals.par_extend((0..copies).into_par_iter().map(|index| {
            let mut copy = individual.clone();
            model.mutate(&mut copy, &mut streams.rng(index as u64));
            copy
        }));
        individuals.push(individual);
        self.evolve(individuals, progress, &mut rng)
            .map(Population::take_fittest)
    }

    /// Runs the evolutionary algorithm on a given initial population.
    ///
    /// # Arguments
    /// * `initial_population` - The individuals to start the algorithm with. They are evaluated
    ///   by the algorithm, so they do not need to be evaluated beforehand.
    /// * `progress` - A closure that is called before each generation is bred. See
    ///   [`run`](Evolution::run) for details.
    ///
    /// # Returns
    /// The population of the last generation.
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if the initial population has no valid
    /// individual, or if a generation ends up with no valid individual.
    pub fn evolve_population<F>(
        &mut self,
        initial_population: Vec<IndividualOf<A>>,
        progress: F,
    ) -> Result<Population<IndividualOf<A>>, EvolutionError>
    where
        F: FnMut(&Population<IndividualOf<A>>) -> bool,
    {
        let mut rng = self.new_rng();
        self.evolve(initial_population, progress, &mut rng)
    }

    fn new_rng(&self) -> Rng {
        self.seed.map_or_else(Rng::from_entropy, Rng::from_seed)
    }

    fn evolve<F>(
        &mut self,
        initial_population: Vec<IndividualOf<A>>,
        mut progress: F,
        rng: &mut Rng,
    ) -> Result<Population<IndividualOf<A>>, EvolutionError>
    where
        F: FnMut(&Population<IndividualOf<A>>) -> bool,
    {
        let algorithm = &mut self.algorithm;
        let mut population = algorithm.initialize_population(initial_population)?;
        let mut offspring = Vec::with_capacity(algorithm.population_size());

        while progress(&population) {
            offspring.clear();
            algorithm.prepare_generation(&population);
            algorithm.select_elites(&population, &mut offspring);
            algorithm.generate_offspring(&population, &mut offspring, rng);
            algorithm.replace_population(&mut population, &mut offspring)?;
        }

        Ok(population)
    }
}

impl<A: EvolutionAlgorithm> From<A> for Evolution<A> {
    #[inline]
    fn from(algorithm: A) -> Self {
        Self::new(algorithm)
    }
}

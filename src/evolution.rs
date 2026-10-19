use super::*;
use rayon::prelude::*;
use std::{ops::ControlFlow, time::Instant};

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
/// algorithm are deterministic apart from the random number generators they are given, reset
/// the state they adapt during a run in [`EvolutionAlgorithm::start_run`] and
/// [`EvolutionModel::start_run`], which every run calls before it creates its starting
/// individuals, and the progress callback does not stop the run by its
/// [elapsed time](RunState::elapsed).
///
/// # Type parameters
/// * `A` - The evolutionary algorithm to run.
///
/// # Example
/// Runs the genetic algorithm on the model of the [OneMax example].
/// ```
/// # mod one_max { include!("../examples/one_max.rs"); }
/// # use one_max::*;
/// use rusty_genes::*;
///
/// let mut evolution = GeneticAlgorithm::new(OneMaxEvolutionModel, Default::default())?
///     .into_evolution()
///     .with_seed(42);
///
/// let limits = Limits::new().generations(10);
/// let best = evolution.run(&50, |state| limits.check(state))?.into_fittest_so_far();
/// assert_eq!(best.genome().len(), 50);
/// # Ok::<(), EvolutionError>(())
/// ```
///
/// [OneMax example]: https://github.com/Khojasteh/rusty_genes/blob/master/examples/one_max.rs
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
    ///   [`RunState`] of the run, which provides the current population, the generation number,
    ///   the number of generations without improvement (stagnation), the fittest individual of
    ///   any generation, and the time the run has taken. The closure returns
    ///   [`ControlFlow::Continue`] to continue, or [`ControlFlow::Break`] with the reason to
    ///   stop, for example by checking [`Limits`].
    ///
    /// The model's [`goal_score`](EvolutionModel::goal_score) is read once after the initial
    /// population has been evaluated. If present, the run stops with
    /// [`StopReason::GoalReached`] as soon as its best score is at least that goal for
    /// maximization or at most that goal for minimization, without calling `progress` for the
    /// generation that reaches it.
    ///
    /// # Returns
    /// The [`Outcome`] of the run, which holds its final state, including the fittest
    /// individual of any generation, and the reason it stopped.
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if no valid individual could be generated.
    /// Returns [`EvolutionError::InvalidParameter`] if the model's goal score is not finite.
    pub fn run<T, F>(
        &mut self,
        args: &T,
        progress: F,
    ) -> Result<Outcome<IndividualOf<A>>, EvolutionError>
    where
        T: Sync,
        IndividualOf<A>: RandomIndividual<T>,
        F: FnMut(&RunState<IndividualOf<A>>) -> ControlFlow<StopReason>,
    {
        let starting_individuals = |algorithm: &A, rng: &mut Rng| {
            let streams = rng.split();
            (0..algorithm.population_size())
                .into_par_iter()
                .map(|index| IndividualOf::<A>::new_random(args, &mut streams.rng(index as u64)))
                .collect()
        };
        self.evolve(starting_individuals, progress)
    }

    /// Runs the evolutionary algorithm with an initial population made of `individual` and its
    /// mutated copies.
    ///
    /// The original is retained in the initial set. Copies are varied by
    /// [`EvolutionModel::mutate`], and the algorithm evaluates every starting individual,
    /// including the original, rather than trusting its cached score. `progress`, goal
    /// stopping, and the outcome work as in [`run`](Self::run).
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if the initial population has no valid
    /// individual.
    /// Returns [`EvolutionError::InvalidParameter`] if the model's goal score is not finite.
    ///
    /// # Note
    /// This method is intended to be used when a good individual has already been found, so its
    /// genetic information is likely to be valuable in generating high-quality offspring.
    pub fn evolve_individual<F>(
        &mut self,
        individual: IndividualOf<A>,
        progress: F,
    ) -> Result<Outcome<IndividualOf<A>>, EvolutionError>
    where
        F: FnMut(&RunState<IndividualOf<A>>) -> ControlFlow<StopReason>,
    {
        let starting_individuals = |algorithm: &A, rng: &mut Rng| {
            let streams = rng.split();
            let model = algorithm.model();
            let copies = algorithm.population_size().saturating_sub(1);
            let mut individuals = Vec::with_capacity(copies + 1);
            individuals.par_extend((0..copies).into_par_iter().map(|index| {
                let mut copy = individual.clone();
                model.mutate(&mut copy, &mut streams.rng(index as u64));
                copy
            }));
            individuals.push(individual);
            individuals
        };
        self.evolve(starting_individuals, progress)
    }

    /// Runs the evolutionary algorithm on a given initial population.
    ///
    /// The algorithm evaluates the starting individuals, so cached scores are not required.
    /// `progress`, goal stopping, and the outcome work as in [`run`](Self::run).
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if the initial population has no valid
    /// individual, or if a generation ends up with no valid individual.
    /// Returns [`EvolutionError::InvalidParameter`] if the model's goal score is not finite.
    pub fn evolve_population<F>(
        &mut self,
        initial_population: Vec<IndividualOf<A>>,
        progress: F,
    ) -> Result<Outcome<IndividualOf<A>>, EvolutionError>
    where
        F: FnMut(&RunState<IndividualOf<A>>) -> ControlFlow<StopReason>,
    {
        self.evolve(|_, _| initial_population, progress)
    }

    fn new_rng(&self) -> Rng {
        self.seed.map_or_else(Rng::from_entropy, Rng::from_seed)
    }

    /// Runs the algorithm from the individuals that `starting_individuals` creates with the
    /// generator of the run, once the algorithm has started the run.
    fn evolve<S, F>(
        &mut self,
        starting_individuals: S,
        mut progress: F,
    ) -> Result<Outcome<IndividualOf<A>>, EvolutionError>
    where
        S: FnOnce(&A, &mut Rng) -> Vec<IndividualOf<A>>,
        F: FnMut(&RunState<IndividualOf<A>>) -> ControlFlow<StopReason>,
    {
        let started = Instant::now();
        let mut rng = self.new_rng();
        let algorithm = &mut self.algorithm;
        algorithm.start_run();
        let mut initial_population = starting_individuals(algorithm, &mut rng);
        algorithm.evaluate_initial(&mut initial_population);
        let population =
            Population::with_objective(initial_population, algorithm.model().objective())?;
        let goal_score = algorithm.model().goal_score(&population);
        if goal_score.is_some_and(|goal| !goal.is_finite()) {
            return Err(EvolutionError::invalid_parameter(
                "goal_score",
                "must be finite",
            ));
        }
        let mut state = RunState::started_at(population, started);
        let mut offspring = Vec::with_capacity(algorithm.population_size());

        let reason = loop {
            if goal_score.is_some_and(|goal| {
                state
                    .population()
                    .objective()
                    .is_goal_reached(state.fittest_so_far().score(), goal)
            }) {
                break StopReason::GoalReached;
            }
            if let ControlFlow::Break(reason) = progress(&state) {
                break reason;
            }
            offspring.clear();
            algorithm.prepare_generation(&state);
            algorithm.select_elites(state.population(), &mut offspring);
            algorithm.generate_offspring(&state, &mut offspring, &mut rng);
            state.advance(|population| algorithm.replace_population(population, &mut offspring))?;
        };

        Ok(Outcome { state, reason })
    }
}

impl<A: EvolutionAlgorithm> From<A> for Evolution<A> {
    #[inline]
    fn from(algorithm: A) -> Self {
        Self::new(algorithm)
    }
}

/// The result of a run of an [`Evolution`]: its final state, and why it stopped.
pub struct Outcome<I: Individual> {
    state: RunState<I>,
    reason: StopReason,
}

impl<I: Individual> Outcome<I> {
    /// Returns the state of the run when it stopped.
    #[inline]
    pub fn state(&self) -> &RunState<I> {
        &self.state
    }

    /// Returns why the run stopped.
    #[inline]
    pub fn reason(&self) -> &StopReason {
        &self.reason
    }

    /// Returns the state of the run when it stopped while consuming the outcome.
    #[inline]
    pub fn into_state(self) -> RunState<I> {
        self.state
    }

    /// Returns the fittest individual of any generation of the run while consuming the outcome.
    #[inline]
    pub fn into_fittest_so_far(self) -> I {
        self.state.into_fittest_so_far()
    }
}

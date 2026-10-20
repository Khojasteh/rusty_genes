use super::*;

/// The problem-specific representation, operators, and scoring rule used by an algorithm.
///
/// The model defines the problem-specific parts of an evolutionary algorithm: the individuals,
/// and the operators that combine (crossover), vary (mutation), refine (local search), and
/// evaluate (score) them. How and when these operators are applied is decided by the algorithm,
/// for example by a [`Reproduction`].
///
/// The model is shared by the threads that breed new individuals in parallel, so it must be
/// [`Sync`].
pub trait EvolutionModel: Sync {
    /// The type of individual used by the model.
    type Individual: Individual;

    /// Whether lower or higher scores are better for this problem.
    ///
    /// The default is [`Objective::Maximize`]. Return [`Objective::Minimize`] to evaluate
    /// individuals directly in units such as cost, distance, or error, without negating or
    /// normalizing their scores. [`Evolution`] reads this when forming the initial population;
    /// the objective stays fixed for that run, including after [`prepare`](Self::prepare).
    fn objective(&self) -> Objective {
        Objective::Maximize
    }

    /// Returns the optional score at which a run has achieved its goal.
    ///
    /// Called once with the evaluated initial population, so goals may depend on the problem
    /// instance, such as a OneMax genome's length. The returned goal stays fixed for that run.
    /// The default is [`None`], leaving termination entirely to the progress callback.
    ///
    /// A goal must be finite. [`Evolution`] stops with [`StopReason::GoalReached`] before
    /// breeding another generation when the best score is at least the goal for maximization
    /// or at most the goal for minimization. The progress callback does not observe the
    /// generation that reaches the goal; the [`Outcome`] of the run holds it.
    fn goal_score(&self, _initial_population: &Population<Self::Individual>) -> Option<f64> {
        None
    }

    /// Combines two parents into a new candidate, which need not have an evaluated score.
    ///
    /// [`Reproduction`] calls this with distinct parents, then improves and evaluates the
    /// candidate before its survival policy considers it. Use only `rng` for random choices
    /// when the run must be reproducible from its seed.
    fn crossover(
        &self,
        parent1: &Self::Individual,
        parent2: &Self::Individual,
        rng: &mut Rng,
    ) -> Self::Individual;

    /// Varies a candidate in place to explore solutions that crossover alone may not produce.
    ///
    /// The engine evaluates it afterward; its cached score may be stale during mutation.
    fn mutate(&self, individual: &mut Self::Individual, rng: &mut Rng);

    /// Optionally refines a candidate by local search before its final evaluation.
    ///
    /// [`Reproduction`] calls this on every candidate, including copied parents, after any
    /// mutation. Its score may be stale or invalid at this point; a search that uses scores
    /// must evaluate its own trial solutions. The default leaves the candidate unchanged.
    fn improve(&self, _individual: &mut Self::Individual, _rng: &mut Rng) {}

    /// Computes and caches the score that [`Individual::score`] will return.
    ///
    /// Store the score in the problem's own units. Whether lower or higher scores are better
    /// is determined by [`objective`](Self::objective). An individual that is not a valid
    /// solution should be evaluated in a way that makes
    /// [`Individual::is_valid`] return [`false`].
    fn evaluate(&self, individual: &mut Self::Individual);

    /// Starts a new run, before its starting individuals are created.
    ///
    /// A model that adapts its own state during a run, such as in [`prepare`](Self::prepare),
    /// resets it here, so that every run starts as if the model were new. The algorithms of this
    /// library call this method from [`EvolutionAlgorithm::start_run`]. The default
    /// implementation performs no operation.
    fn start_run(&mut self) {}

    /// Prepares the model for breeding the next generation from the population of `state`.
    ///
    /// The algorithms of this library call this method once per generation, before breeding.
    /// A model may use it to adapt its own state, such as the strength of its mutation, to the
    /// population or the progress of the run. The default implementation performs no
    /// operation.
    fn prepare(&mut self, _state: &RunState<Self::Individual>) {}
}

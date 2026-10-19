use super::*;
use std::{fmt, ops::ControlFlow, time::Duration};

/// Why a run of an [`Evolution`] stopped.
///
/// The run stops with [`GoalReached`](StopReason::GoalReached) when it reaches the model's
/// [goal](EvolutionModel::goal_score), and otherwise with the reason its progress callback
/// gives, such as one of [`Limits`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StopReason {
    /// The fittest individual so far reached the model's
    /// [goal score](EvolutionModel::goal_score).
    GoalReached,

    /// The run reached its limit on the number of generations.
    GenerationLimit,

    /// The run reached its limit on the number of generations without improvement.
    StagnationLimit,

    /// The run reached its time limit.
    TimeLimit,

    /// The run was interrupted, such as by its user.
    Interrupted,

    /// Another reason, described by the progress callback that stopped the run.
    Custom(String),
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GoalReached => f.write_str("goal reached"),
            Self::GenerationLimit => f.write_str("generation limit reached"),
            Self::StagnationLimit => f.write_str("stagnation limit reached"),
            Self::TimeLimit => f.write_str("time limit reached"),
            Self::Interrupted => f.write_str("interrupted"),
            Self::Custom(reason) => f.write_str(reason),
        }
    }
}

/// Limits on a run, which stop it as soon as one of them is reached.
///
/// A limit that is not set does not stop the run. [`check`](Limits::check) decides for a
/// progress callback whether the run continues, and why it stops.
///
/// # Example
/// Runs the genetic algorithm on the OneMax problem for at most 500 generations, 50 of them
/// without improvement.
/// ```
/// use rusty_genes::one_max::*;
/// use rusty_genes::*;
///
/// let limits = Limits::new().generations(500).stagnation(50);
/// let mut evolution = GeneticAlgorithm::new(OneMaxEvolutionModel, Default::default())?
///     .into_evolution()
///     .with_seed(42);
///
/// let outcome = evolution.run(&50, |state| limits.check(state))?;
/// assert!(outcome.state().generation() <= 500);
/// println!("Stopped after {} generations: {}", outcome.state().generation(), outcome.reason());
/// # Ok::<(), EvolutionError>(())
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    generations: Option<usize>,
    stagnation: Option<usize>,
    time: Option<Duration>,
}

impl Limits {
    /// Creates limits of which none is set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stops the run at generation `count`, so that at most `count` generations are bred after
    /// the initial population.
    pub fn generations(mut self, count: usize) -> Self {
        self.generations = Some(count);
        self
    }

    /// Stops the run once `count` generations have passed without improving the
    /// [fittest individual so far](RunState::fittest_so_far).
    pub fn stagnation(mut self, count: usize) -> Self {
        self.stagnation = Some(count);
        self
    }

    /// Stops the run once it has taken at least `duration`, as measured by
    /// [`RunState::elapsed`].
    ///
    /// How many generations fit in the time depends on the speed of the machine, so a run that
    /// this limit stops is not reproducible from its seed.
    pub fn time(mut self, duration: Duration) -> Self {
        self.time = Some(duration);
        self
    }

    /// Returns [`ControlFlow::Break`] with the reason of the limit that `state` reaches, or
    /// [`ControlFlow::Continue`] if it reaches none.
    ///
    /// When `state` reaches several limits, the generation limit is reported before the
    /// stagnation limit, and that before the time limit.
    pub fn check<I: Individual>(&self, state: &RunState<I>) -> ControlFlow<StopReason> {
        if self
            .generations
            .is_some_and(|count| state.generation() >= count)
        {
            ControlFlow::Break(StopReason::GenerationLimit)
        } else if self
            .stagnation
            .is_some_and(|count| state.stagnation() >= count)
        {
            ControlFlow::Break(StopReason::StagnationLimit)
        } else if self
            .time
            .is_some_and(|duration| state.elapsed() >= duration)
        {
            ControlFlow::Break(StopReason::TimeLimit)
        } else {
            ControlFlow::Continue(())
        }
    }
}

use super::*;
use std::time::{Duration, Instant};

/// The state of a run: the current population and the history of the generations before it.
///
/// An [`Evolution`] starts the state from the initial population, presents it to its progress
/// callback and to its algorithm before breeding each generation, and returns it in the
/// [`Outcome`] of the run.
pub struct RunState<I: Individual> {
    population: Population<I>,
    generation: usize,
    stagnation: usize,
    fittest_so_far: I,
    started: Instant,
    elapsed: Duration,
}

impl<I: Individual> RunState<I> {
    /// Starts a run from `population` as its generation zero, timing the run from now.
    pub fn new(population: Population<I>) -> Self {
        Self::started_at(population, Instant::now())
    }

    /// Starts a run from `population` as its generation zero, timing the run from `started`.
    pub(crate) fn started_at(population: Population<I>, started: Instant) -> Self {
        let fittest_so_far = population.fittest().clone();
        Self {
            population,
            generation: 0,
            stagnation: 0,
            fittest_so_far,
            started,
            elapsed: started.elapsed(),
        }
    }

    /// Replaces the population by `replace` and, if it succeeds, records the new generation.
    ///
    /// On success, the generation counter is incremented, and the stagnation counter is reset
    /// if the new population is fitter than the
    /// [fittest individual so far](RunState::fittest_so_far), or incremented otherwise.
    ///
    /// # Errors
    /// Returns the error of `replace`, such as [`EvolutionError::EmptyPopulation`] from
    /// [`Population::replace`]. In that case, the counters and the fittest individual so far
    /// are left unchanged.
    pub fn advance<F>(&mut self, replace: F) -> Result<(), EvolutionError>
    where
        F: FnOnce(&mut Population<I>) -> Result<(), EvolutionError>,
    {
        replace(&mut self.population)?;
        self.generation += 1;
        let fittest = self.population.fittest();
        if self
            .population
            .objective()
            .is_fitter(fittest.score(), self.fittest_so_far.score())
        {
            self.fittest_so_far = fittest.clone();
            self.stagnation = 0;
        } else {
            self.stagnation += 1;
        }
        self.elapsed = self.started.elapsed();
        Ok(())
    }

    /// Returns the population of the current generation.
    #[inline]
    pub fn population(&self) -> &Population<I> {
        &self.population
    }

    /// Returns the generation number of the current population, starting at zero for the
    /// initial population and increasing with each [advance](RunState::advance).
    #[inline]
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Returns the number of generations since the score of the
    /// [fittest individual so far](RunState::fittest_so_far) last improved.
    #[inline]
    pub fn stagnation(&self) -> usize {
        self.stagnation
    }

    /// Returns the fittest individual of the current and every earlier generation of the run.
    /// An equally fit individual of a later generation does not replace it.
    ///
    /// The fittest individual so far may be fitter than the
    /// [fittest](Population::fittest) individual of the current population when a generation
    /// can lose the fittest individual of the one before, such as without elites.
    #[inline]
    pub fn fittest_so_far(&self) -> &I {
        &self.fittest_so_far
    }

    /// Returns the time from the start of the run until the current population was formed.
    ///
    /// The run of an [`Evolution`] starts when one of its methods is called, before it creates
    /// and evaluates the initial individuals. The time does not advance until the next
    /// generation is formed.
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Returns the population of the current generation while consuming the state.
    #[inline]
    pub fn into_population(self) -> Population<I> {
        self.population
    }

    /// Returns the [fittest individual so far](RunState::fittest_so_far) while consuming the
    /// state.
    #[inline]
    pub fn into_fittest_so_far(self) -> I {
        self.fittest_so_far
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        hash::{Hash, Hasher},
        ops::ControlFlow,
        time::Duration,
    };

    #[derive(Debug, Clone, PartialEq)]
    struct Mock {
        id: u32,
        score: f64,
    }

    impl Hash for Mock {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.id.hash(state);
        }
    }

    impl Individual for Mock {
        fn score(&self) -> f64 {
            self.score
        }
    }

    fn mock(id: u32, score: f64) -> Mock {
        Mock { id, score }
    }

    fn state(individuals: Vec<Mock>) -> RunState<Mock> {
        RunState::new(Population::new(individuals).unwrap())
    }

    fn advance(state: &mut RunState<Mock>, individuals: Vec<Mock>) {
        state
            .advance(|population| population.replace(&mut individuals.clone()))
            .unwrap();
    }

    #[test]
    fn a_new_state_starts_at_generation_zero_with_its_fittest_individual() {
        let state = state(vec![mock(1, 2.0), mock(2, 1.0)]);
        assert_eq!((state.generation(), state.stagnation()), (0, 0));
        assert_eq!(state.fittest_so_far().id, 1);
    }

    #[test]
    fn advance_tracks_generation_and_stagnation() {
        let mut state = state(vec![mock(1, 1.0)]);

        advance(&mut state, vec![mock(2, 1.0)]);
        assert_eq!((state.generation(), state.stagnation()), (1, 1));

        advance(&mut state, vec![mock(3, 2.0)]);
        assert_eq!((state.generation(), state.stagnation()), (2, 0));
    }

    #[test]
    fn advance_counts_stagnation_from_the_best_score_ever_reached() {
        let mut state = state(vec![mock(1, 2.0)]);

        // The fittest individual is lost, and then an equally fit one is found again.
        advance(&mut state, vec![mock(2, 1.0)]);
        advance(&mut state, vec![mock(3, 2.0)]);
        assert_eq!(state.stagnation(), 2);

        advance(&mut state, vec![mock(4, 3.0)]);
        assert_eq!(state.stagnation(), 0);
    }

    #[test]
    fn fittest_so_far_is_the_fittest_individual_of_any_generation() {
        let mut state = state(vec![mock(1, 2.0), mock(2, 1.0)]);

        advance(&mut state, vec![mock(3, 1.5)]);
        assert_eq!(state.population().fittest().id, 3);
        assert_eq!(state.fittest_so_far().id, 1);

        // An equally fit individual does not replace the one found first.
        advance(&mut state, vec![mock(4, 2.0)]);
        assert_eq!(state.fittest_so_far().id, 1);

        advance(&mut state, vec![mock(5, 2.5)]);
        assert_eq!(state.into_fittest_so_far().id, 5);
    }

    #[test]
    fn improvement_follows_the_objective_of_the_population() {
        let population = Population::with_objective(vec![mock(1, 10.0)], Objective::Minimize);
        let mut state = RunState::new(population.unwrap());

        advance(&mut state, vec![mock(2, 15.0)]);
        assert_eq!((state.stagnation(), state.fittest_so_far().id), (1, 1));

        advance(&mut state, vec![mock(3, -5.0)]);
        assert_eq!((state.stagnation(), state.fittest_so_far().id), (0, 3));
    }

    #[test]
    fn a_failed_advance_leaves_the_state_unchanged() {
        let mut state = state(vec![mock(1, 1.0)]);
        let elapsed = state.elapsed();

        let result = state.advance(|population| population.replace(&mut vec![mock(2, f64::NAN)]));

        assert_eq!(result, Err(EvolutionError::EmptyPopulation));
        assert_eq!((state.generation(), state.stagnation()), (0, 0));
        assert_eq!(state.elapsed(), elapsed);
        assert_eq!(state.population().fittest().id, 1);
    }

    #[test]
    fn elapsed_time_is_that_of_the_current_generation() {
        let mut state = state(vec![mock(1, 1.0)]);
        let first = state.elapsed();
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(state.elapsed(), first);

        advance(&mut state, vec![mock(2, 1.0)]);
        assert!(state.elapsed() >= first + Duration::from_millis(5));
    }

    #[test]
    fn limits_stop_once_their_counts_are_reached() {
        let mut state = state(vec![mock(1, 1.0)]);
        let generations = Limits::new().generations(2);
        let stagnation = Limits::new().stagnation(2);

        advance(&mut state, vec![mock(2, 1.0)]);
        assert_eq!(generations.check(&state), ControlFlow::Continue(()));
        assert_eq!(stagnation.check(&state), ControlFlow::Continue(()));

        advance(&mut state, vec![mock(3, 1.0)]);
        assert_eq!(
            generations.check(&state),
            ControlFlow::Break(StopReason::GenerationLimit)
        );
        assert_eq!(
            stagnation.check(&state),
            ControlFlow::Break(StopReason::StagnationLimit)
        );
    }

    #[test]
    fn limits_report_the_generation_limit_before_the_stagnation_and_time_limits() {
        let mut state = state(vec![mock(1, 1.0)]);
        advance(&mut state, vec![mock(2, 1.0)]);
        let all = Limits::new()
            .generations(1)
            .stagnation(1)
            .time(Duration::ZERO);
        assert_eq!(
            all.check(&state),
            ControlFlow::Break(StopReason::GenerationLimit)
        );
        assert_eq!(
            all.generations(2).check(&state),
            ControlFlow::Break(StopReason::StagnationLimit)
        );
        assert_eq!(
            all.generations(2).stagnation(2).check(&state),
            ControlFlow::Break(StopReason::TimeLimit)
        );
    }

    #[test]
    fn unset_limits_never_stop_a_run() {
        let state = state(vec![mock(1, 1.0)]);
        assert_eq!(Limits::new().check(&state), ControlFlow::Continue(()));
        assert_eq!(
            Limits::new().time(Duration::from_secs(3600)).check(&state),
            ControlFlow::Continue(())
        );
    }
}

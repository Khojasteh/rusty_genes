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

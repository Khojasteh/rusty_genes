use super::*;
use rayon::prelude::*;
use std::ops::{Deref, DerefMut};

/// Represents a population of individuals in an evolutionary algorithm.
///
/// # Note
/// The individuals in the population are unique and they are sorted by their fitness value
/// in ascending order.
pub struct Population<I: Individual> {
    individuals: Vec<I>,
    generation: usize,
    stagnation: usize,
    stats: PopulatedStats,
}

impl<I: Individual> Population<I> {
    /// Creates a new instance of the `Population` struct with a provided initial population.
    ///
    /// # Arguments
    /// * `initial_population` - A vector of individuals that make up the initial population.
    ///
    /// # Returns
    /// A new Population instance containing the provided individuals, with statistical data
    /// collected and ready for use.
    ///
    /// # Panics
    /// This method will panic if `initial_population` is empty, as an evolutionary algorithm
    /// requires at least one individual to operate.
    pub fn new(mut initial_population: Vec<I>) -> Self {
        let stats = sort_dedup_and_collect_stats(&mut initial_population);
        Self {
            individuals: initial_population,
            generation: 0,
            stagnation: 0,
            stats,
        }
    }

    /// Replaces the current population with a new generation.
    ///
    /// # Arguments
    /// * `new_population` - A mutable reference to a vector of individuals that will replace
    ///   the current population.
    ///
    /// # Behavior
    /// This method swaps the current population with the `new_population` argument, effectively
    /// replacing the old generation. It also updates statistical data about the population,
    /// increments the generation counter, and tracks the stagnation state. The stagnation counter
    /// is reset if the fitness of the best individual has improved, otherwise it is incremented.
    ///
    /// Upon return, the `new_population` argument will contain the old population.
    ///
    /// # Panics
    /// This method will panic if `new_population` is empty, as an evolutionary algorithm requires
    /// at least one individual to operate.
    #[inline]
    pub fn replace(&mut self, new_population: &mut Vec<I>) {
        std::mem::swap(&mut self.individuals, new_population);
        let previous_best = self.stats.max();
        self.stats = sort_dedup_and_collect_stats(&mut self.individuals);
        self.generation += 1;
        if self.stats.max() > previous_best {
            self.stagnation = 0;
        } else {
            self.stagnation += 1;
        }
    }

    /// Returns the current generation number of the population.
    #[inline]
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Returns the number of generations since the last improvement.
    #[inline]
    pub fn stagnation(&self) -> usize {
        self.stagnation
    }

    /// Returns the statistical information regarding the fitness score of the individuals
    /// in the current population.
    #[inline]
    pub fn stats(&self) -> &PopulatedStats {
        &self.stats
    }

    /// Returns the fittest individual in the population.
    #[inline]
    pub fn fittest(&self) -> &I {
        &self.individuals[self.stats.arg_max()]
    }

    /// Returns the fittest individual in the population while consuming the population.
    #[inline]
    pub fn take_fittest(mut self) -> I {
        self.individuals.swap_remove(self.stats.arg_max())
    }
}

impl<I: Individual> Deref for Population<I> {
    type Target = Vec<I>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.individuals
    }
}

impl<I: Individual> DerefMut for Population<I> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.individuals
    }
}

impl<I: Individual> TryFrom<Vec<I>> for Population<I> {
    type Error = Vec<I>;

    fn try_from(individuals: Vec<I>) -> Result<Self, Self::Error> {
        if !individuals.is_empty() {
            Ok(Self::new(individuals))
        } else {
            Err(individuals)
        }
    }
}

impl<I: Individual> From<Population<I>> for Vec<I> {
    fn from(population: Population<I>) -> Self {
        population.individuals
    }
}

#[inline]
fn sort_dedup_and_collect_stats<I: Individual>(individuals: &mut Vec<I>) -> PopulatedStats {
    individuals.par_sort_unstable_by(compare_fitness);
    individuals.dedup();
    individuals
        .iter()
        .map(|i| i.fitness())
        .collect::<Stats>()
        .try_into()
        .unwrap()
}

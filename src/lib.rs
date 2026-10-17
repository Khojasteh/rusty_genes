//! RustyGenes is a library for building and running evolutionary algorithms, with a focus on
//! permutation-based problems.
//!
//! # Overview
//!
//! A problem is described by an [`Individual`], which holds a candidate solution and its fitness,
//! and an [`EvolutionModel`], which combines, varies, and evaluates individuals. The library
//! maximizes fitness.
//!
//! An algorithm, such as [`GeneticAlgorithm`] or [`AdaptiveGeneticAlgorithm`], evolves a
//! [`Population`] of individuals generation by generation, and an [`Evolution`] runs it. The
//! algorithms breed offspring through a [`Reproduction`], which selects parents by a
//! [`SelectionMechanism`] and keeps an offspring only if its [`SurvivalPolicy`] accepts it. Each
//! of these can be replaced by an implementation of its trait, and [`EvolutionAlgorithm`] defines
//! the steps of a custom algorithm.
//!
//! The crossover, mutation, and selection strategies are available as building blocks for models
//! and algorithms: see [`CrossoverStrategy`], [`MutationStrategy`], and [`SelectionStrategy`].
//!
//! Randomness comes from the library's own [`Rng`]. A run seeded by [`Evolution::with_seed`]
//! gives the same result on any number of threads.
//!
//! # Example
//!
//! ```
//! use rusty_genes::one_max::*;
//! use rusty_genes::*;
//!
//! let mut evolution = GeneticAlgorithm::new(OneMaxEvolutionModel, Default::default())?
//!     .into_evolution()
//!     .with_seed(42);
//!
//! // Evolve 64-bit strings towards all '1' bits for at most 100 generations.
//! let solution = evolution.run(&64, |population| {
//!     population.fittest().fitness() < 1.0 && population.generation() < 100
//! })?;
//! assert_eq!(solution.genome().len(), 64);
//! # Ok::<(), EvolutionError>(())
//! ```
//!
//! # Features
//!
//! * `rand-0_10` - Implements the random number generator traits of `rand` 0.10 for [`Rng`], so
//!   it can be used with the distributions and other functions of that crate.

#![deny(unsafe_code)]

mod algorithms;
mod error;
mod evolution;
mod individual;
mod model;
mod operators;
mod population;
mod reproduction;
mod rng;
mod stats;

pub use algorithms::*;
pub use error::*;
pub use evolution::*;
pub use individual::*;
pub use model::*;
pub use operators::*;
pub use population::*;
pub use reproduction::*;
pub use rng::*;
pub use stats::*;

/// Returns the value at `ratio` of the way from `min` to `max`, where a ratio of 0.0 gives `min`
/// and 1.0 gives `max`.
#[inline]
pub(crate) fn interpolate_f64(min: f64, max: f64, ratio: f64) -> f64 {
    min + (max - min) * ratio
}

/// Returns the value at `ratio` of the way from `min` to `max`, rounded down. `max` must not be
/// less than `min`.
#[inline]
pub(crate) fn interpolate_usize(min: usize, max: usize, ratio: f64) -> usize {
    (min as f64 + (max - min) as f64 * ratio) as usize
}

/// This module provides an implementation of the OneMax problem for use with the evolutionary
/// algorithm framework. The OneMax problem is a simple problem often used in the field of
/// genetic algorithms to test new techniques. The problem consists of maximizing the number
/// of '1' bits in a binary string.
pub mod one_max {
    use super::*;

    /// Represents an individual in the OneMax problem.
    ///
    /// The `genome` is a binary string represented as a vector of booleans. Each boolean
    /// represents a bit, and the fitness of an individual is the ratio of 'true' bits to
    /// the total number of bits in this binary string.
    #[derive(Debug, Clone, PartialEq)]
    pub struct OneMax {
        genome: Vec<bool>,
        fitness: f64,
    }

    impl OneMax {
        /// Returns the binary string of the individual.
        pub fn genome(&self) -> &[bool] {
            &self.genome
        }
    }

    /// Conversion from a `Vec<bool>` to a `OneMax` individual.
    ///
    /// This allows a binary string to be directly converted into an individual. The individual
    /// is not evaluated.
    impl From<Vec<bool>> for OneMax {
        fn from(genome: Vec<bool>) -> Self {
            Self {
                genome,
                fitness: 0.0,
            }
        }
    }

    /// Implements the method required by the [`Individual`] trait for the OneMax problem.
    impl Individual for OneMax {
        /// Returns the fitness of the individual.
        fn fitness(&self) -> f64 {
            self.fitness
        }
    }

    /// Allows the creation of a new random individual for the OneMax problem.
    impl RandomIndividual<usize> for OneMax {
        /// Generates a new random individual for the OneMax problem.
        ///
        /// The `length` parameter determines the length of the binary string (i.e., the genome),
        /// and the `rng` parameter is a random number generator used to generate the binary string.
        fn new_random(length: &usize, rng: &mut Rng) -> Self {
            let genome = (0..*length)
                .map(|_| rng.random_bool(0.5))
                .collect::<Vec<_>>();
            Self::from(genome)
        }
    }

    /// The evolutionary model for the OneMax problem.
    ///
    /// This struct provides methods for crossover, mutation, and evaluation operations specific
    /// to the OneMax problem.
    #[derive(Debug, Default)]
    pub struct OneMaxEvolutionModel;

    /// Implements the methods required by the [`EvolutionModel`] trait for the OneMax problem.
    impl EvolutionModel for OneMaxEvolutionModel {
        type Individual = OneMax;

        /// Performs a single point crossover on the parents' genomes to create a new individual.
        fn crossover(
            &self,
            parent1: &Self::Individual,
            parent2: &Self::Individual,
            rng: &mut Rng,
        ) -> Self::Individual {
            CrossoverStrategy::SinglePoint
                .crossover(&parent1.genome, &parent2.genome, rng)
                .into()
        }

        /// Mutates the individual's genome by using the duplicate mutation strategy.
        fn mutate(&self, individual: &mut Self::Individual, rng: &mut Rng) {
            MutationStrategy::Duplicate.mutate(&mut individual.genome, rng);
        }

        /// Evaluates the individual's fitness by calculating the ratio of `true` bits to the total
        /// number of bits in its genome. An empty genome is invalid, since its fitness is NaN.
        fn evaluate(&self, individual: &mut Self::Individual) {
            let one_bits = individual.genome.iter().filter(|&&one| one).count();
            let total_bits = individual.genome.len();
            individual.fitness = one_bits as f64 / total_bits as f64;
        }
    }
}

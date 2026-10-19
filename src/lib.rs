//! RustyGenes is a library for building and running evolutionary algorithms, with a focus on
//! permutation-based problems.
//!
//! # Overview
//!
//! A problem is described by an [`Individual`], which holds a candidate solution and its score,
//! and an [`EvolutionModel`], which combines, varies, and evaluates individuals. The library
//! uses scores in the model's own units and maximizes them by default. A model can return
//! [`Objective::Minimize`] from [`EvolutionModel::objective`] to minimize costs or errors.
//!
//! An algorithm, such as [`GeneticAlgorithm`] or [`AdaptiveGeneticAlgorithm`], evolves a
//! [`Population`] of individuals generation by generation, and an [`Evolution`] runs it. The
//! [`RunState`] of a run holds its population and history, such as its generation number and
//! the fittest individual of any generation, and a progress callback observes it to decide when
//! the run stops, for example by [`Limits`]. The algorithms breed offspring through a
//! [`Reproduction`], which selects parents by a [`SelectionMechanism`] and keeps an offspring
//! only if its [`SurvivalPolicy`] accepts it. Each of these can be replaced by an
//! implementation of its trait, and [`EvolutionAlgorithm`] defines the steps of a custom
//! algorithm.
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
//! const GENOME_LENGTH: usize = 64;
//!
//! let mut evolution = GeneticAlgorithm::new(OneMaxEvolutionModel, Default::default())?
//!     .into_evolution()
//!     .with_seed(42);
//!
//! // Evolve 64-bit strings towards all '1' bits for at most 100 generations.
//! let limits = Limits::new().generations(100);
//! let outcome = evolution.run(&GENOME_LENGTH, |state| {
//!     limits.check(state)
//! })?;
//! println!("Stopped at generation {}: {}", outcome.state().generation(), outcome.reason());
//! let solution = outcome.into_fittest_so_far();
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
mod objective;
mod operators;
mod population;
mod reproduction;
mod rng;
mod run_state;
mod stats;
mod stopping;

pub use algorithms::*;
pub use error::*;
pub use evolution::*;
pub use individual::*;
pub use model::*;
pub use objective::*;
pub use operators::*;
pub use population::*;
pub use reproduction::*;
pub use rng::*;
pub use run_state::*;
pub use stats::*;
pub use stopping::*;

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

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
    use std::hash::{Hash, Hasher};

    /// Represents an individual in the OneMax problem.
    ///
    /// The `genome` is a binary string represented as a vector of booleans. Each boolean
    /// represents a bit, and the score of an evaluated individual is the number of `true`
    /// bits, without normalization. An optimal genome of length `n` has a score of `n`.
    ///
    /// # Example
    /// Equality and hashing use only the genome, so an individual equals its evaluated copy.
    /// ```
    /// use rusty_genes::one_max::*;
    /// use rusty_genes::*;
    /// use std::hash::{BuildHasher, RandomState};
    ///
    /// let unevaluated = OneMax::from(vec![true, false]);
    /// assert!(!unevaluated.is_valid());
    ///
    /// let mut evaluated = unevaluated.clone();
    /// OneMaxEvolutionModel.evaluate(&mut evaluated);
    /// assert!(evaluated.is_valid());
    ///
    /// assert_eq!(unevaluated, evaluated);
    /// let hasher = RandomState::new();
    /// assert_eq!(hasher.hash_one(&unevaluated), hasher.hash_one(&evaluated));
    /// ```
    #[derive(Debug, Clone)]
    pub struct OneMax {
        genome: Vec<bool>,
        score: f64,
    }

    impl PartialEq for OneMax {
        fn eq(&self, other: &Self) -> bool {
            self.genome == other.genome
        }
    }

    impl Eq for OneMax {}

    impl Hash for OneMax {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.genome.hash(state);
        }
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
    /// is not evaluated and remains invalid until the model evaluates it.
    impl From<Vec<bool>> for OneMax {
        fn from(genome: Vec<bool>) -> Self {
            Self {
                genome,
                score: f64::NAN,
            }
        }
    }

    /// Implements the method required by the [`Individual`] trait for the OneMax problem.
    impl Individual for OneMax {
        /// Returns the score of the individual.
        fn score(&self) -> f64 {
            self.score
        }

        /// A valid OneMax individual has a non-empty genome and a finite score.
        fn is_valid(&self) -> bool {
            !self.genome.is_empty() && self.score.is_finite()
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

        /// The goal is to set every bit, so it equals the genome length of this run.
        /// All individuals in a OneMax run must have the same genome length.
        ///
        /// # Example
        /// A run stops as soon as every bit is set, even in its initial population. Each run
        /// takes its goal from its own initial population, so the goal follows the genome length.
        /// ```
        /// use rusty_genes::one_max::*;
        /// use rusty_genes::*;
        ///
        /// let mut evolution = GeneticAlgorithm::new(OneMaxEvolutionModel, Default::default())?
        ///     .into_evolution()
        ///     .with_seed(42);
        ///
        /// let outcome = evolution.evolve_population(vec![OneMax::from(vec![true; 8])], |s| {
        ///     Limits::new().generations(5).check(s)
        /// })?;
        /// assert_eq!(outcome.reason(), &StopReason::GoalReached);
        /// assert_eq!(outcome.state().generation(), 0);
        /// assert_eq!(outcome.state().population().fittest().score(), 8.0);
        ///
        /// let limits = Limits::new().generations(2);
        /// let first = evolution
        ///     .evolve_population(vec![OneMax::from(vec![true])], |s| limits.check(s))?;
        /// assert_eq!(first.state().generation(), 0);
        ///
        /// let second = evolution.evolve_population(
        ///     vec![OneMax::from(vec![true, false, true, false])],
        ///     |s| limits.check(s),
        /// )?;
        /// assert!(second.state().generation() > 0);
        /// assert_eq!(
        ///     OneMaxEvolutionModel.goal_score(second.state().population()),
        ///     Some(4.0)
        /// );
        /// # Ok::<(), EvolutionError>(())
        /// ```
        fn goal_score(&self, initial_population: &Population<OneMax>) -> Option<f64> {
            Some(initial_population.fittest().genome.len() as f64)
        }

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

        /// Flips one randomly chosen bit, allowing new set bits even in an all-zero genome.
        ///
        /// # Example
        /// The mutated individual remains invalid until it is evaluated again.
        /// ```
        /// use rusty_genes::one_max::*;
        /// use rusty_genes::*;
        ///
        /// let mut individual = OneMax::from(vec![false]);
        /// OneMaxEvolutionModel.evaluate(&mut individual);
        /// assert!(individual.is_valid());
        ///
        /// OneMaxEvolutionModel.mutate(&mut individual, &mut Rng::from_seed(0));
        /// assert!(!individual.is_valid());
        ///
        /// OneMaxEvolutionModel.evaluate(&mut individual);
        /// assert_eq!(individual.genome(), &[true]);
        /// assert_eq!(individual.score(), 1.0);
        /// ```
        fn mutate(&self, individual: &mut Self::Individual, rng: &mut Rng) {
            if !individual.genome.is_empty() {
                let index = rng.random_range(0..individual.genome.len());
                individual.genome[index] = !individual.genome[index];
                individual.score = f64::NAN;
            }
        }

        /// Evaluates the individual's score as the number of `true` bits in its genome.
        /// Empty genomes remain invalid according to [`Individual::is_valid`].
        ///
        /// # Example
        /// ```
        /// use rusty_genes::one_max::*;
        /// use rusty_genes::*;
        ///
        /// for (genome, expected) in [
        ///     (vec![false], 0.0),
        ///     (vec![true], 1.0),
        ///     (vec![true, false, true, false], 2.0),
        ///     (vec![true, true, false, false, false, false], 2.0),
        ///     (vec![true; 8], 8.0),
        /// ] {
        ///     let mut individual = OneMax::from(genome);
        ///     OneMaxEvolutionModel.evaluate(&mut individual);
        ///     assert_eq!(individual.score(), expected);
        ///     assert!(individual.is_valid());
        /// }
        ///
        /// // An empty genome remains invalid, so a population cannot hold it.
        /// let mut empty = OneMax::from(vec![]);
        /// OneMaxEvolutionModel.evaluate(&mut empty);
        /// assert!(!empty.is_valid());
        /// assert!(matches!(
        ///     Population::new(vec![empty]),
        ///     Err(EvolutionError::EmptyPopulation)
        /// ));
        /// ```
        fn evaluate(&self, individual: &mut Self::Individual) {
            let one_bits = individual.genome.iter().filter(|&&one| one).count();
            individual.score = one_bits as f64;
        }
    }
}

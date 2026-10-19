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
//! the run stops, for example by [`Limits`]. The algorithms breed each generation through a
//! [`Reproduction`], which selects parents by a [`SelectionMechanism`] to breed candidates, and
//! lets a [`SurvivalPolicy`] choose the next generation from the population and the candidates.
//! Each of these can be replaced by an implementation of its trait, and [`EvolutionAlgorithm`]
//! defines the steps of a custom algorithm.
//!
//! The crossover, mutation, and selection strategies are available as building blocks for models
//! and algorithms: see [`CrossoverStrategy`], [`MutationStrategy`], and [`SelectionStrategy`].
//!
//! Randomness comes from the library's own [`Rng`]. A run seeded by [`Evolution::with_seed`]
//! gives the same result on any number of threads.
//!
//! # Example
//!
//! This runs the genetic algorithm on the model of the [OneMax example], which evolves strings
//! of bits towards one in which every bit is `1`. The example shows how its individual and model
//! are defined.
//!
//! ```
//! # mod one_max { include!("../examples/one_max.rs"); }
//! # use one_max::*;
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
//! [OneMax example]: https://github.com/Khojasteh/rusty_genes/blob/master/examples/one_max.rs
//!
//! # Features
//!
//! * `rand-0_10` - Implements the random number generator traits of `rand` 0.10 for [`Rng`], so
//!   it can be used with the distributions and other functions of that crate.

#![deny(unsafe_code)]

mod algorithms;
mod error;
mod evolution;
mod generation;
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
pub use generation::*;
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

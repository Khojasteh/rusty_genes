#![deny(unsafe_code)]

mod algorithms;
mod evolution;
mod individual;
mod model;
mod operators;
mod population;
mod stats;

pub use algorithms::*;
pub use evolution::*;
pub use individual::*;
pub use model::*;
pub use operators::*;
pub use population::*;
pub use stats::*;

/// Calculates the linear interpolation value between two `f64` values based on a given ratio.
///
/// # Arguments
/// * `min` - The minimum value to be interpolated.
/// * `max` - The maximum value to be interpolated.
/// * `ratio` - An `f64` value representing the interpolation ratio. The ratio should be between
///   0.0 and 1.0, where 0.0 corresponds to the minimum value and 1.0 corresponds to the maximum
///   value.
///
/// # Returns
/// Returns an `f64` value representing the interpolated value.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let interpolated_value = interpolate_f64(0.0, 10.0, 0.5);
/// assert!((interpolated_value - 5.0).abs() < 1e-6);
/// ```
#[inline]
pub fn interpolate_f64(min: f64, max: f64, ratio: f64) -> f64 {
    min + (max - min) * ratio
}

/// Calculates the linear interpolation value between two `usize` values based on a given ratio.
///
/// # Arguments
/// * `min` - The minimum value to be interpolated.
/// * `max` - The maximum value to be interpolated.
/// * `ratio` - An `f64` value representing the interpolation ratio. The ratio should be between
///   0.0 and 1.0, where 0.0 corresponds to the minimum value and 1.0 corresponds to the maximum
///   value.
///
/// # Returns
/// Returns a `usize` value representing the interpolated value.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let interpolated_value = interpolate_usize(0, 10, 0.5);
/// assert_eq!(interpolated_value, 5);
/// ```
#[inline]
pub fn interpolate_usize(min: usize, max: usize, ratio: f64) -> usize {
    (min as f64 + (max - min) as f64 * ratio) as usize
}

/// This module provides an implementation of the OneMax problem for use with the evolutionary
/// algorithm framework. The OneMax problem is a simple problem often used in the field of
/// genetic algorithms to test new techniques. The problem consists of maximizing the number
/// of '1' bits in a binary string.
pub mod one_max {
    use super::*;
    use rand::{distributions::Standard, prelude::*};

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
        // Returns the binary string of the individual.
        pub fn genome(&self) -> &[bool] {
            &self.genome
        }
    }

    /// Conversion from a `Vec<bool>` to a `OneMax` individual.
    ///
    /// This allows a binary string to be directly converted into an individual.
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
        fn new_random<R: Rng>(length: &usize, rng: &mut R) -> Self {
            let bits = rng.sample_iter(Standard);
            let genome = Vec::from_iter(bits.take(*length));
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
        type Rng = SmallRng;

        /// Creates a new random number generator from entropy.
        fn rng() -> Self::Rng {
            SmallRng::from_entropy()
        }

        /// Performs a single point crossover on the parents' genomes to create a new individual.
        fn crossover(
            &self,
            parent1: &Self::Individual,
            parent2: &Self::Individual,
            rng: &mut Self::Rng,
        ) -> Self::Individual {
            CrossoverStrategy::SinglePoint
                .crossover(&parent1.genome, &parent2.genome, rng)
                .into()
        }

        /// Mutates the individual's genome by using the duplicate mutation strategy.
        fn mutate(&self, individual: &mut Self::Individual, rng: &mut Self::Rng) {
            MutationStrategy::Duplicate.mutate(&mut individual.genome, rng);
        }

        /// Evaluates the individual's fitness by calculating the ratio of `true` bits to the total
        /// number of bits in its genome.
        fn evaluate(&self, individual: &mut Self::Individual) {
            let one_bits = individual.genome.iter().filter(|&&one| one).count();
            let total_bits = individual.genome.len();
            individual.fitness = one_bits as f64 / total_bits as f64;
        }
    }
}

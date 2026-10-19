use crate::Rng;
use std::hash::{Hash, Hasher};

/// Represents an individual in a population of an evolutionary algorithm.
///
/// The model's [`objective`](crate::EvolutionModel::objective) determines whether lower or
/// higher scores are better. The engine evaluates candidates before admitting them to a
/// population, so new or changed genomes may carry an invalid score until then.
///
/// Equality determines whether individuals are duplicates in a [`Population`](crate::Population).
/// It should compare only the solution itself, excluding a cached score, so that re-evaluated
/// copies of a solution are recognized as duplicates.
///
/// [`Hash`] must agree with equality: equal individuals must have equal hashes, so the hash
/// covers the same solution data and excludes a cached score. A population compares only
/// individuals with equal hashes, which takes `O(n log n)` time for `n` individuals when few
/// hashes are equal. `f64` has no [`Hash`]; hash floating-point genes with [`hash_f64`] or
/// [`hash_f64_slice`]. A constant hash is valid when a solution cannot be hashed, but then every
/// pair of individuals is compared, which takes quadratic time.
///
/// # Example
/// A knapsack packing whose score is the value of the packed items. A packing that exceeds the
/// capacity is invalid, so it is evaluated as NaN. A population discards invalid packings and
/// keeps one copy of each packing.
/// ```
/// use rusty_genes::*;
/// use std::hash::{Hash, Hasher};
///
/// const VALUES: [f64; 3] = [6.0, 10.0, 12.0];
/// const WEIGHTS: [f64; 3] = [1.0, 2.0, 3.0];
/// const CAPACITY: f64 = 5.0;
///
/// #[derive(Clone)]
/// struct Packing {
///     packed: Vec<bool>,
///     value: f64,
/// }
///
/// impl Packing {
///     fn evaluated(packed: Vec<bool>) -> Self {
///         let total = |amounts: [f64; 3]| -> f64 {
///             packed.iter().zip(amounts).filter_map(|(&p, a)| p.then_some(a)).sum()
///         };
///         let value = if total(WEIGHTS) <= CAPACITY { total(VALUES) } else { f64::NAN };
///         Self { packed, value }
///     }
/// }
///
/// // A packing is identified by the items it packs, not by its cached value.
/// impl PartialEq for Packing {
///     fn eq(&self, other: &Self) -> bool {
///         self.packed == other.packed
///     }
/// }
///
/// impl Hash for Packing {
///     fn hash<H: Hasher>(&self, state: &mut H) {
///         self.packed.hash(state);
///     }
/// }
///
/// impl Individual for Packing {
///     fn score(&self) -> f64 {
///         self.value
///     }
/// }
///
/// let population = Population::new(vec![
///     Packing::evaluated(vec![true, true, false]),
///     Packing::evaluated(vec![false, true, true]),
///     Packing::evaluated(vec![true, true, true]), // Too heavy.
///     Packing::evaluated(vec![true, true, false]), // A duplicate.
/// ])?;
///
/// let scores = population.iter().map(Individual::score).collect::<Vec<_>>();
/// assert_eq!(scores, [16.0, 22.0]);
/// assert_eq!(population.fittest().packed, [false, true, true]);
/// # Ok::<(), EvolutionError>(())
/// ```
pub trait Individual: Clone + PartialEq + Hash + Send + Sync {
    /// Returns the cached evaluation in the model's own units, without evaluating the genome.
    ///
    /// The score may be negative or outside `[0, 1]`.
    /// It must be finite for a valid individual. The model's [`Objective`](crate::Objective)
    /// determines whether lower or higher scores are better.
    fn score(&self) -> f64;

    /// Returns [`true`] if the individual is a valid solution. Invalid individuals never become
    /// part of a [`Population`](crate::Population).
    ///
    /// # Note
    /// The default implementation considers an individual valid when its score is finite,
    /// i.e. neither infinite nor NaN.
    ///
    /// Implementations of this trait may override this method with a more specific definition of
    /// validity, but an individual whose score is not finite must never be reported as valid.
    #[inline]
    fn is_valid(&self) -> bool {
        self.score().is_finite()
    }
}

/// Represents an individual in a population of an evolutionary algorithm that can be generated
/// randomly.
///
/// The trait is parameterized over a type `A`, which represents domain-specific arguments for
/// generating the individual, such as the length of its genome.
pub trait RandomIndividual<A>: Individual {
    /// Generates a new individual with random characteristics.
    ///
    /// # Arguments
    /// * `args` - A reference to domain-specific arguments for generating the individual.
    /// * `rng` - A mutable reference to random number generator.
    ///
    /// # Returns
    /// A new individual whose characteristics are determined by the random number generator and
    /// the domain-specific arguments.
    fn new_random(args: &A, rng: &mut Rng) -> Self;
}

/// Feeds an `f64` into `state` consistently with `==`, for implementing [`Hash`] on
/// an [`Individual`] with floating-point genes.
///
/// `0.0` and `-0.0` are equal, so they hash equally. NaN is not equal to itself, so its hash
/// is unconstrained.
///
/// # Example
/// Points are equal when their coordinates are, and `0.0 == -0.0`, so a population keeps one of
/// two points that differ only in the sign of zero.
/// ```
/// use rusty_genes::*;
/// use std::hash::{Hash, Hasher};
///
/// #[derive(Clone)]
/// struct Point {
///     x: f64,
///     y: f64,
///     score: f64,
/// }
///
/// impl PartialEq for Point {
///     fn eq(&self, other: &Self) -> bool {
///         (self.x, self.y) == (other.x, other.y)
///     }
/// }
///
/// impl Hash for Point {
///     fn hash<H: Hasher>(&self, state: &mut H) {
///         hash_f64(self.x, state);
///         hash_f64(self.y, state);
///     }
/// }
///
/// impl Individual for Point {
///     fn score(&self) -> f64 {
///         self.score
///     }
/// }
///
/// let population = Population::new(vec![
///     Point { x: 0.0, y: 1.0, score: 1.0 },
///     Point { x: -0.0, y: 1.0, score: 1.0 },
/// ])?;
/// assert_eq!(population.len(), 1);
/// # Ok::<(), EvolutionError>(())
/// ```
#[inline]
pub fn hash_f64<H: Hasher>(value: f64, state: &mut H) {
    let value = if value == 0.0 { 0.0 } else { value };
    state.write_u64(value.to_bits());
}

/// Feeds a slice of `f64` into `state` consistently with `==`, for implementing [`Hash`] on
/// an [`Individual`] whose genome is a vector of floating-point genes.
///
/// Like the [`Hash`] implementation of slices, it includes the length, so consecutive slices
/// hash differently when their elements are split differently between them. Each element is
/// hashed as by [`hash_f64`].
///
/// # Example
/// A genome of real-valued weights, scored by their sum. Equal genomes, including those that
/// differ only in the sign of zero, are one solution in a population.
/// ```
/// use rusty_genes::*;
/// use std::hash::{Hash, Hasher};
///
/// #[derive(Clone)]
/// struct Weights {
///     genes: Vec<f64>,
///     score: f64,
/// }
///
/// impl Weights {
///     fn evaluated(genes: Vec<f64>) -> Self {
///         let score = genes.iter().sum();
///         Self { genes, score }
///     }
/// }
///
/// impl PartialEq for Weights {
///     fn eq(&self, other: &Self) -> bool {
///         self.genes == other.genes
///     }
/// }
///
/// impl Hash for Weights {
///     fn hash<H: Hasher>(&self, state: &mut H) {
///         hash_f64_slice(&self.genes, state);
///     }
/// }
///
/// impl Individual for Weights {
///     fn score(&self) -> f64 {
///         self.score
///     }
/// }
///
/// let population = Population::new(vec![
///     Weights::evaluated(vec![0.0, 0.5]),
///     Weights::evaluated(vec![0.5, 0.0]),
///     Weights::evaluated(vec![-0.0, 0.5]),
/// ])?;
/// assert_eq!(population.len(), 2);
/// # Ok::<(), EvolutionError>(())
/// ```
#[inline]
pub fn hash_f64_slice<H: Hasher>(values: &[f64], state: &mut H) {
    state.write_usize(values.len());
    for &value in values {
        hash_f64(value, state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, RandomState};

    #[test]
    fn floating_point_hashes_agree_with_equality() {
        let hasher = RandomState::new();
        let hash = |value| hasher.hash_one(HashedF64(value));
        assert_eq!(hash(0.0), hash(-0.0));
        assert_eq!(hash(1.5), hash(1.5));

        let hash_slices = |values: &[&[f64]]| {
            let mut state = hasher.build_hasher();
            for values in values {
                hash_f64_slice(values, &mut state);
            }
            state.finish()
        };
        assert_eq!(hash_slices(&[&[0.0, 1.0]]), hash_slices(&[&[-0.0, 1.0]]));
        assert_ne!(
            hash_slices(&[&[1.0, 2.0], &[3.0]]),
            hash_slices(&[&[1.0], &[2.0, 3.0]])
        );
    }

    struct HashedF64(f64);

    impl Hash for HashedF64 {
        fn hash<H: Hasher>(&self, state: &mut H) {
            hash_f64(self.0, state);
        }
    }
}

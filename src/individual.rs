use crate::Rng;
use std::hash::Hash;

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
/// hashes are equal. A constant hash is valid when a solution cannot be hashed, but then every
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

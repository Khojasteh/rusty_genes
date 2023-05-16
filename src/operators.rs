use super::*;
use rand::prelude::*;
use std::{collections::HashSet, hash::Hash};

mod mutation;
pub use mutation::*;

mod crossover;
pub use crossover::*;

mod selection;
pub use selection::*;

/// Compares two individuals by their fitness scores.
#[inline]
pub fn compare_fitness<I: Individual>(a: &I, b: &I) -> std::cmp::Ordering {
    a.fitness()
        .partial_cmp(&b.fitness())
        .unwrap_or(std::cmp::Ordering::Equal)
}

/// Returns a distinct random pair of numbers within the range `[0, n)`.
///
/// # Arguments
/// * `n` - The upper limit of the range (exclusive).
/// * `rng` - A mutable reference to a random number generator implementing the `Rng` trait.
///
/// # Panics
/// The function will panic if `n` is less than 2.
/// # Example
/// ```
/// use rusty_genes::*;
/// use rand::prelude::*;
///
/// let mut rng = SmallRng::seed_from_u64(0);
///
/// let pair = random_pair(10, &mut rng);
/// assert_eq!(pair, (4, 5));
/// ```
#[inline]
pub fn random_pair<R: Rng>(n: usize, rng: &mut R) -> (usize, usize) {
    assert!(n >= 2);

    let p1 = rng.gen_range(0..n);
    let mut p2 = rng.gen_range(0..n - 1);
    if p2 >= p1 {
        p2 += 1;
    }
    (p1, p2)
}

/// Returns a random range as a subset of `0..n`. The range is guaranteed to be non-empty
/// and uniformly distributed.
///
/// # Arguments
/// * `n` - The exclusive upper bound for the random range (0..n)
/// * `rng` - A mutable reference to a random number generator implementing the `Rng` trait
///
/// # Panics
/// * The function will panic if `n` is zero.
///
/// # Example
/// ```
/// use rusty_genes::*;
/// use rand::prelude::*;
///
/// let mut rng = SmallRng::seed_from_u64(0);
///
/// let range = random_range(10, &mut rng);
/// assert_eq!(range, 2..7);
#[inline]
pub fn random_range<R: Rng>(n: usize, rng: &mut R) -> std::ops::Range<usize> {
    assert!(n != 0);
    bounded_random_range(n, 1, n, rng)
}

/// Returns a random range as a subset of `0..n`, with the additional constraint
/// that the length of the range should be between `min_len` and `max_len` (inclusive).
/// The range is guaranteed to be uniformly distributed.
///
/// # Arguments
/// * `n` - The exclusive upper bound for the random range (0..n)
/// * `min_len` - The minimum length (inclusive) for the random range
/// * `max_len` - The maximum length (inclusive) for the random range
/// * `rng` - A mutable reference to a random number generator implementing the `Rng` trait
///
/// # Panics
/// The function will panic if `min_len` is larger than `max_len` and it may panic if `max_len`
/// is larger than `n`.
///
/// # Example
/// ```
/// use rusty_genes::*;
/// use rand::prelude::*;
///
/// let mut rng = SmallRng::seed_from_u64(0);
///
/// let range = bounded_random_range(10, 3, 5, &mut rng);
/// assert_eq!(range, 3..7);
/// ```
#[inline]
pub fn bounded_random_range<R: Rng>(
    n: usize,
    min_len: usize,
    max_len: usize,
    rng: &mut R,
) -> std::ops::Range<usize> {
    let length = rng.gen_range(min_len..=max_len);
    let start = rng.gen_range(0..=(n - length));
    let end = start + length;
    start..end
}

/// Identifies and returns the common neighboring genes between two genomes.
///
/// For each genome, the function identifies neighboring genes, i.e., pairs of genes that are adjacent
/// in the genome sequence. Then, it returns those neighbors that are exist in both genomes.
///
/// # Arguments
/// * `genome1` - A reference to the first  genome.
/// * `genome2` - A reference to the second genome.
///
/// # Returns
/// A `Vec<(G, G)>` where each tuple represents a pair of neighboring genes common to both genomes.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let genome1 = vec![1, 2, 3, 4, 5];
/// let genome2 = vec![1, 3, 2, 4, 5];
///
/// let common = common_neighbors(&genome1, &genome2);
/// assert_eq!(common, vec![(2, 3), (4, 5), (1, 5)]);
/// ```
pub fn common_neighbors<G>(genome1: &[G], genome2: &[G]) -> Vec<(G, G)>
where
    G: Copy + PartialOrd + Eq + Hash,
{
    if genome1.len() < 2 || genome2.len() <= 2 {
        return Vec::new();
    }

    let (genome1, genome2) = if genome1.len() <= genome2.len() {
        (genome1, genome2)
    } else {
        (genome2, genome1)
    };

    let mut neighbors1 = Vec::with_capacity(genome1.len());
    let mut neighbors2 = HashSet::with_capacity(genome2.len());

    let neighbor = |a: &G, b: &G| if a < b { (*a, *b) } else { (*b, *a) };

    neighbors1.extend(
        genome1
            .windows(2)
            .map(|w| neighbor(&w[0], &w[1]))
            .chain(std::iter::once(neighbor(
                genome1.last().unwrap(),
                genome1.first().unwrap(),
            ))),
    );
    neighbors2.extend(
        genome2
            .windows(2)
            .map(|w| neighbor(&w[0], &w[1]))
            .chain(std::iter::once(neighbor(
                genome2.last().unwrap(),
                genome2.first().unwrap(),
            ))),
    );

    neighbors1.retain(|neighbor| neighbors2.contains(neighbor));
    neighbors1
}

/// Identifies and returns the common partitions between two genomes.
///
/// # Note
/// A partition is a sequence of genes that is common to both genomes and where
/// each gene is a neighbor of the next gene in the sequence in both genomes.
///
/// # Arguments
/// * `genome1` - A reference to the first genome.
/// * `genome2` - A reference to the second genome.
///
/// # Returns
/// A `Vec<Vec<G>>` representing the common partitions found in both parent genomes.
/// Each partition is a vector of genes. The order of genes in a partition is the
/// same as their order in the parent genomes.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let genome1 = vec![1, 2, 3, 4, 5];
/// let genome2 = vec![1, 3, 2, 4, 5];
///
/// let partitions = common_partitions(&genome1, &genome2);
/// assert_eq!(partitions, vec![vec![1, 5, 4], vec![2, 3]]);
/// ```
pub fn common_partitions<G>(genome1: &[G], genome2: &[G]) -> Vec<Vec<G>>
where
    G: Copy + PartialOrd + Eq + Hash,
{
    let mut partitions = Vec::with_capacity(genome1.len());
    let mut visited = HashSet::with_capacity(genome1.len());
    let common_neighbors = common_neighbors(genome1, genome2);

    for &node in genome1.iter() {
        if visited.contains(&node) {
            continue;
        }

        let mut partition = Vec::with_capacity(common_neighbors.len());
        partition.push(node);
        visited.insert(node);

        while let Some(next_node) = partition.last().and_then(|&last| {
            common_neighbors.iter().find_map(|&(a, b)| {
                if a == last && !visited.contains(&b) {
                    Some(b)
                } else if b == last && !visited.contains(&a) {
                    Some(a)
                } else {
                    None
                }
            })
        }) {
            partition.push(next_node);
            visited.insert(next_node);
        }

        partitions.push(partition);
    }

    partitions
}

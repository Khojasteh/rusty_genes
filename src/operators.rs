use super::*;
use std::{
    collections::{HashMap, HashSet},
    hash::{BuildHasherDefault, Hash, Hasher},
};

mod mutation;
pub use mutation::*;

mod crossover;
pub use crossover::*;

mod selection;
pub use selection::*;

mod survival;
pub use survival::*;

/// A hash map keyed by genes, using [`GeneHasher`].
type GeneMap<K, V> = HashMap<K, V, BuildHasherDefault<GeneHasher>>;

/// A hash set of genes, using [`GeneHasher`].
type GeneSet<K> = HashSet<K, BuildHasherDefault<GeneHasher>>;

/// A fast, non-cryptographic hasher for the gene lookups of the genetic operators.
///
/// The operators hash the genes of every genome they process, so the hasher is on their hot
/// path. Unlike the standard library's default hasher, it is neither randomly seeded nor
/// resistant to collision attacks, which is acceptable because the genes come from the caller's
/// own model rather than from an adversary. Collections whose iteration order affects a result
/// must not use it, since its order is the same on every run.
#[derive(Debug, Default, Clone, Copy)]
struct GeneHasher(u64);

impl GeneHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        // The multiply-rotate step of the FxHash function.
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for GeneHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i.into());
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(i.into());
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i.into());
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
}

/// Compares two individuals by their fitness scores.
///
/// The comparison is a total order that follows [`f64::total_cmp`].
#[inline]
pub fn compare_fitness<I: Individual>(a: &I, b: &I) -> std::cmp::Ordering {
    a.fitness().total_cmp(&b.fitness())
}

/// Returns a distinct random pair of numbers within the range `[0, n)`.
///
/// # Arguments
/// * `n` - The upper limit of the range (exclusive).
/// * `rng` - A mutable reference to a random number generator.
///
/// # Panics
/// The function will panic if `n` is less than 2.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let (p1, p2) = random_pair(10, &mut Rng::from_seed(1));
/// assert!(p1 < 10 && p2 < 10 && p1 != p2);
/// ```
#[inline]
pub fn random_pair(n: usize, rng: &mut Rng) -> (usize, usize) {
    assert!(n >= 2);

    let p1 = rng.random_range(0..n);
    let mut p2 = rng.random_range(0..n - 1);
    if p2 >= p1 {
        p2 += 1;
    }
    (p1, p2)
}

/// Returns a random non-empty range as a subset of `0..n`.
///
/// The length of the range is uniformly distributed between 1 and `n`, and for a given length,
/// the start of the range is uniformly distributed over the positions the length allows.
///
/// # Arguments
/// * `n` - The exclusive upper bound for the random range (0..n)
/// * `rng` - A mutable reference to a random number generator
///
/// # Panics
/// The function will panic if `n` is zero.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let range = random_range(10, &mut Rng::from_seed(1));
/// assert!(!range.is_empty() && range.end <= 10);
/// ```
#[inline]
pub fn random_range(n: usize, rng: &mut Rng) -> std::ops::Range<usize> {
    assert!(n != 0);
    bounded_random_range(n, 1, n, rng)
}

/// Returns a random range as a subset of `0..n`, with the additional constraint that the length
/// of the range is between `min_len` and `max_len` (inclusive).
///
/// The length of the range is uniformly distributed between `min_len` and `max_len`, and for
/// a given length, the start of the range is uniformly distributed over the positions the length
/// allows.
///
/// # Arguments
/// * `n` - The exclusive upper bound for the random range (0..n)
/// * `min_len` - The minimum length (inclusive) for the random range
/// * `max_len` - The maximum length (inclusive) for the random range
/// * `rng` - A mutable reference to a random number generator
///
/// # Panics
/// The function will panic if `min_len` is larger than `max_len`, or if `max_len` is larger
/// than `n`.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let range = bounded_random_range(10, 3, 5, &mut Rng::from_seed(1));
/// assert!((3..=5).contains(&range.len()) && range.end <= 10);
/// ```
#[inline]
pub fn bounded_random_range(
    n: usize,
    min_len: usize,
    max_len: usize,
    rng: &mut Rng,
) -> std::ops::Range<usize> {
    assert!(min_len <= max_len && max_len <= n);

    let length = rng.random_range(min_len..max_len + 1);
    let start = rng.random_range(0..n - length + 1);
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
    let mut neighbors2 = GeneSet::with_capacity_and_hasher(genome2.len(), Default::default());

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
/// Each partition is a vector of genes, and each gene of `genome1` is in exactly one
/// partition. A partition extends as far as the common neighbors reach, and its genes are
/// in the order of that chain, which can run backwards or wrap around the ends of the
/// genomes, as in the example below.
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
    let mut visited = GeneSet::with_capacity_and_hasher(genome1.len(), Default::default());

    // The common neighbors of each gene, in the order of the common neighbor pairs.
    let mut neighbors =
        GeneMap::<G, Vec<G>>::with_capacity_and_hasher(genome1.len(), Default::default());
    for (a, b) in common_neighbors(genome1, genome2) {
        neighbors.entry(a).or_default().push(b);
        neighbors.entry(b).or_default().push(a);
    }

    for &node in genome1.iter() {
        if visited.contains(&node) {
            continue;
        }

        let mut partition = vec![node];
        visited.insert(node);
        extend_chain(&mut partition, node, &neighbors, &mut visited);

        // The gene may be in the middle of a chain, so the chain can also continue from its
        // other neighbor; that part precedes the gene in the partition.
        let mut head = Vec::new();
        extend_chain(&mut head, node, &neighbors, &mut visited);
        if !head.is_empty() {
            head.reverse();
            head.append(&mut partition);
            partition = head;
        }

        partitions.push(partition);
    }

    partitions
}

/// Follows the chain of common neighbors from `start` while it reaches unvisited genes, and
/// appends those genes to `chain` in the order they are reached.
fn extend_chain<G>(
    chain: &mut Vec<G>,
    start: G,
    neighbors: &GeneMap<G, Vec<G>>,
    visited: &mut GeneSet<G>,
) where
    G: Copy + Eq + Hash,
{
    let mut last = start;
    while let Some(next) = neighbors.get(&last).and_then(|neighbors| {
        neighbors
            .iter()
            .copied()
            .find(|neighbor| !visited.contains(neighbor))
    }) {
        chain.push(next);
        visited.insert(next);
        last = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn common_partitions_keeps_a_chain_through_the_first_gene_whole() {
        // The common neighbors 4-5, 5-0, and 0-1 form one chain, which passes through the first
        // gene of the first genome.
        let genome1 = [0, 1, 2, 3, 4, 5];
        let genome2 = [5, 0, 1, 3, 2, 4];

        let mut partitions = common_partitions(&genome1, &genome2);
        for partition in &mut partitions {
            partition.sort_unstable();
        }
        partitions.sort_unstable();

        assert_eq!(partitions, vec![vec![0, 1, 4, 5], vec![2, 3]]);
    }
}

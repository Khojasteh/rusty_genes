use super::*;

/// In-place variation of a genome without changing its length.
///
/// All strategies except [`Duplicate`](Self::Duplicate) preserve the number of copies of each
/// gene, so they can vary permutations without dropping or repeating an item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MutationStrategy {
    /// Exchanges two distinct, randomly chosen positions.
    Swap,

    /// Shuffles a contiguous range of at least two genes; the order can remain unchanged.
    Scramble,

    /// Reverses a contiguous range of at least two genes.
    Inversion,

    /// Moves a randomly chosen gene to a distinct position, shifting the intervening genes.
    Insertion,

    /// Copies one position's value over another, potentially changing gene counts.
    /// This does not preserve a permutation.
    Duplicate,
}

impl MutationStrategy {
    /// Varies `genome` in place using this strategy; fewer than two genes are left unchanged.
    #[inline]
    pub fn mutate<G: Copy>(&self, genome: &mut [G], rng: &mut Rng) {
        match self {
            Self::Swap => swap_mutation(genome, rng),
            Self::Scramble => scramble_mutation(genome, rng),
            Self::Inversion => inversion_mutation(genome, rng),
            Self::Insertion => insertion_mutation(genome, rng),
            Self::Duplicate => duplicate_mutation(genome, rng),
        }
    }
}

/// Exchanges two distinct positions chosen uniformly from `genome`.
///
/// Gene counts are preserved. Genomes with fewer than two genes are unchanged; equal values
/// at the chosen positions can also leave the genome unchanged.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let original = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let mut genome = original.clone();
///
/// swap_mutation(&mut genome, &mut Rng::from_seed(1));
///
/// let changed = (0..10).filter(|&i| genome[i] != original[i]).collect::<Vec<_>>();
/// assert_eq!(changed.len(), 2);
/// assert_eq!(genome[changed[0]], original[changed[1]]);
/// assert_eq!(genome[changed[1]], original[changed[0]]);
/// ```
#[inline]
pub fn swap_mutation<G>(genome: &mut [G], rng: &mut Rng) {
    let n = genome.len();
    if n < 2 {
        return;
    }

    let (p1, p2) = random_pair(n, rng);
    genome.swap(p1, p2);
}

/// Shuffles a random contiguous range of at least two genes, preserving gene counts.
///
/// Range lengths are uniform, then starts are uniform for the chosen length. Fewer than two
/// genes are left unchanged, and shuffling can also reproduce the original order.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let mut genome = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
///
/// scramble_mutation(&mut genome, &mut Rng::from_seed(1));
///
/// let mut genes = genome.clone();
/// genes.sort_unstable();
/// assert_eq!(genes, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
/// ```
#[inline]
pub fn scramble_mutation<G>(genome: &mut [G], rng: &mut Rng) {
    let n = genome.len();
    if n < 2 {
        return;
    }

    let range = bounded_random_range(n, 2, n, rng);
    rng.shuffle(&mut genome[range]);
}

/// Reverses a random contiguous range of at least two genes, preserving gene counts.
///
/// Range lengths are uniform, then starts are uniform for the chosen length. Genomes with
/// fewer than two genes are unchanged.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let original = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let mut genome = original.clone();
///
/// inversion_mutation(&mut genome, &mut Rng::from_seed(1));
///
/// let start = (0..10).position(|i| genome[i] != original[i]).unwrap();
/// let end = (0..10).rposition(|i| genome[i] != original[i]).unwrap() + 1;
/// assert!(genome[start..end].iter().eq(original[start..end].iter().rev()));
/// ```
#[inline]
pub fn inversion_mutation<G>(genome: &mut [G], rng: &mut Rng) {
    let n = genome.len();
    if n < 2 {
        return;
    }

    let range = bounded_random_range(n, 2, n, rng);
    genome[range].reverse();
}

/// Moves one gene to a distinct position before or after it, shifting the intervening genes.
///
/// Source and destination form a uniformly chosen ordered pair. Gene counts are preserved,
/// and genomes with fewer than two genes are unchanged.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let original = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let mut genome = original.clone();
///
/// insertion_mutation(&mut genome, &mut Rng::from_seed(1));
///
/// // The genes from the old to the new position of the moved gene shift by one position.
/// let start = (0..10).position(|i| genome[i] != original[i]).unwrap();
/// let end = (0..10).rposition(|i| genome[i] != original[i]).unwrap() + 1;
/// let mut moved_left = original[start..end].to_vec();
/// moved_left.rotate_right(1);
/// let mut moved_right = original[start..end].to_vec();
/// moved_right.rotate_left(1);
/// assert!(genome[start..end] == moved_left || genome[start..end] == moved_right);
/// ```
#[inline]
pub fn insertion_mutation<G>(genome: &mut [G], rng: &mut Rng) {
    let n = genome.len();
    if n < 2 {
        return;
    }

    let (from, to) = random_pair(n, rng);
    if from > to {
        genome[to..=from].rotate_right(1);
    } else {
        genome[from..=to].rotate_left(1);
    }
}

/// Copies one gene's value over another, using uniformly chosen distinct positions.
///
/// This can drop one gene and repeat another, so it does not preserve permutations. Genomes
/// with fewer than two genes are unchanged.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let original = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let mut genome = original.clone();
///
/// duplicate_mutation(&mut genome, &mut Rng::from_seed(1));
///
/// let changed = (0..10).filter(|&i| genome[i] != original[i]).collect::<Vec<_>>();
/// assert_eq!(changed.len(), 1);
/// assert_eq!(genome.iter().filter(|&&gene| gene == genome[changed[0]]).count(), 2);
/// ```
#[inline]
pub fn duplicate_mutation<G: Copy>(genome: &mut [G], rng: &mut Rng) {
    let n = genome.len();
    if n < 2 {
        return;
    }

    let (p1, p2) = random_pair(n, rng);
    genome[p2] = genome[p1];
}

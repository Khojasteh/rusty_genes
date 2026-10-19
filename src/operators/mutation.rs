use super::*;

/// Represents various mutation strategies for modifying an individual's genetic material.
///
/// In genetic algorithms, mutation is used to prevent the population from stagnating at a local
/// minima or maxima and introduces novel gene sequences that could potentially lead to better
/// solutions. By introducing small, random tweaks to the genetic material, mutation can help
/// explore new areas of the solution space that may not have been reachable via crossover alone.
///
/// Depending on the nature of the specific problem and the algorithm used, different mutation
/// strategies can be more effective. Therefore, it is often beneficial to have several mutation
/// strategies available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MutationStrategy {
    /// Swap Mutation exchanges the positions of two randomly chosen elements in the genetic material.
    /// This mutation operator is effective at making small changes in the solution structure,
    /// allowing for local search and fine-tuning of the solution.
    Swap,

    /// Scramble Mutation picks a random subset of elements in the genetic material and shuffles their
    /// positions within the selected subset. This mutation operator is useful for introducing
    /// larger-scale changes in the solution, promoting exploration and diversity in the population.
    Scramble,

    /// Inversion Mutation selects a random subset of elements in the genetic material and reverses their
    /// order within the chosen subset. This mutation operator can introduce both small and
    /// large changes in the solution, balancing exploration and exploitation in the search process.
    Inversion,

    /// Insertion Mutation selects a gene randomly from the genome and moves it to another position.
    /// This mutation operator can introduce a small local rearrangement of genes, and is particularly
    /// useful in permutation-based problems where the ordering of genes is important.
    Insertion,

    /// Duplicate Mutation selects a gene randomly from the genome and copies its value to another
    /// position, replacing the gene that was previously there. This mutation operator can introduce
    /// variation into the genome, and can be useful in problems where duplications of certain genes
    /// may lead to a better solution. Note that this mutation is not applicable to permutation-based
    /// problems where each gene should be unique.
    Duplicate,
}

impl MutationStrategy {
    /// Performs a mutation operation on an individual's gene sequence according to the chosen strategy.
    ///
    /// # Arguments
    /// * `genome` - The sequence of genes to be mutated. This could represent a solution
    ///   in the problem space.
    /// * `rng` - A random number generator for use in stochastic processes.
    ///
    /// # Note
    /// The mutation operation directly modifies the `genome` in place. A genome with fewer than
    /// two genes is left unchanged.
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

/// Exchanges the positions of two randomly chosen genes of the given genome.
///
/// This mutation operator is effective at making small changes in the solution structure,
/// allowing for local search and fine-tuning of the solution.
///
/// # Arguments
/// * `genome` - The sequence of genes to be mutated. This could represent a solution in the
///   problem space.
/// * `rng` - A mutable reference to a random number generator
///
/// # Note
/// Swap mutation is applicable to permutation-based problems.
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

/// Shuffles the genes within a randomly selected range, of at least two genes, of the given
/// genome.
///
/// This mutation operator is useful for introducing larger-scale changes in the solution,
/// promoting exploration and diversity in the population.
///
/// # Arguments
/// * `genome` - The sequence of genes to be mutated. This could represent a solution in the
///   problem space.
/// * `rng` - A mutable reference to a random number generator
///
/// # Note
/// Scramble mutation is applicable to permutation-based problems.
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

/// Reverses the order of the genes within a randomly selected range, of at least two genes, of
/// the given genome.
///
/// This mutation operator can introduce both small and large changes in the solution, balancing
/// exploration and exploitation in the search process.
///
/// # Arguments
/// * `genome` - The sequence of genes to be mutated. This could represent a solution in the
///   problem space.
/// * `rng` - A mutable reference to a random number generator
///
/// # Note
/// Inversion mutation is applicable to permutation-based problems.
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
/// The mutation operator can introduce a small local rearrangement of genes which can be particularly
/// useful in permutation-based problems where the relative ordering of genes is important.
///
/// # Arguments
/// * `genome` - The sequence of genes to be mutated. This could represent a solution in the
///   problem space.
/// * `rng` - A mutable reference to a random number generator
///
/// # Note
/// Insertion mutation is applicable to permutation-based problems.
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

/// Selects a gene from a random position in the genome and copies it to another position, replacing
/// the gene that was previously there.
///
/// This mutation operator can introduce variation into the genome, and can be useful in problems where
/// duplications of certain genes may lead to a better solution.
///
/// # Arguments
/// * `genome` - The sequence of genes to be mutated. This could represent a solution in the
///   problem space.
/// * `rng` - A mutable reference to a random number generator
///
/// # Note
/// Duplicate mutation is not applicable to permutation-based problems where each gene should be unique.
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

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
    ///    in the problem space.
    /// * `rng` - A random number generator for use in stochastic processes.
    ///
    /// # Note
    /// The mutation operation directly modifies the `genome` in place.
    pub fn mutate<G: Copy, R: Rng>(&self, genome: &mut [G], rng: &mut R) {
        match self {
            Self::Swap => swap_mutation(genome, rng),
            Self::Scramble => scramble_mutation(genome, rng),
            Self::Inversion => inversion_mutation(genome, rng),
            Self::Insertion => insertion_mutation(genome, rng),
            Self::Duplicate => duplicate_mutation(genome, rng),
        }
    }
}

/// Performs a swap mutation on the given genome.
///
/// This mutation operator is effective at making small changes in the solution structure,
/// allowing for local search and fine-tuning of the solution.
///
/// # Arguments
/// * `genome` - The sequence of genes to be mutated. This could represent a solution in the
///    problem space.
/// * `rng` - A mutable reference to a random number generator implementing the Rng trait
///
/// # Note
/// Swap mutation is applicable to permutation-based problems.
///
/// # Example
/// ```
/// use rusty_genes::*;
/// use rand::prelude::*;
///
/// let mut rng = SmallRng::seed_from_u64(999);
/// let mut genome = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
///
/// swap_mutation(&mut genome, &mut rng);
///
/// assert_eq!(genome, vec![0, 1, 6, 3, 4, 5, 2, 7, 8, 9]);
/// ```
#[inline]
pub fn swap_mutation<G, R: Rng>(genome: &mut [G], rng: &mut R) {
    let n = genome.len();
    if n < 2 {
        return;
    }

    let (p1, p2) = random_pair(n, rng);
    genome.swap(p1, p2);
}

/// Shuffles the elements within a randomly selected subset of the given genetic material.
///
/// This mutation operator is useful for introducing larger-scale changes in the solution,
/// promoting exploration and diversity in the population.
///
/// # Arguments
/// * `genome` - The sequence of genes to be mutated. This could represent a solution in the
///    problem space.
/// * `rng` - A mutable reference to a random number generator implementing the Rng trait
///
/// # Note
/// Scramble mutation is applicable to permutation-based problems.
///
/// # Example
/// ```
/// use rusty_genes::*;
/// use rand::prelude::*;
///
/// let mut rng = SmallRng::seed_from_u64(999);
/// let mut genome = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
///
/// scramble_mutation(&mut genome, &mut rng);
///
/// assert_eq!(genome, vec![0, 5, 6, 1, 2, 3, 4, 7, 8, 9]);
/// ```
#[inline]
pub fn scramble_mutation<G, R: Rng>(genome: &mut [G], rng: &mut R) {
    let n = genome.len();
    if n < 2 {
        return;
    }

    let range = bounded_random_range(n, 2, n, rng);
    genome[range].shuffle(rng);
}

/// Reverses the order of the elements within a randomly selected subset of the given genetic
/// material.
///
/// This mutation operator can introduce both small and large changes in the solution, balancing
/// exploration and exploitation in the search process.
///
/// # Arguments
/// * `genome` - The sequence of genes to be mutated. This could represent a solution in the
///    problem space.
/// * `rng` - A mutable reference to a random number generator implementing the Rng trait
///
/// # Note
/// Inversion mutation is applicable to permutation-based problems.
///
/// # Example
/// ```
/// use rusty_genes::*;
/// use rand::prelude::*;
///
/// let mut rng = SmallRng::seed_from_u64(999);
/// let mut genome = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
///
/// inversion_mutation(&mut genome, &mut rng);
///
/// assert_eq!(genome, vec![0, 6, 5, 4, 3, 2, 1, 7, 8, 9]);
/// ```
#[inline]
pub fn inversion_mutation<G, R: Rng>(genome: &mut [G], rng: &mut R) {
    let n = genome.len();
    if n < 2 {
        return;
    }

    let range = bounded_random_range(n, 2, n, rng);
    genome[range].reverse();
}

/// Removes a gene from a random position in the genome and insert it in another position. This
/// results in the shifting of the rest of the genes between two positions.
///
/// The mutation operator can introduce a small local rearrangement of genes which can be particularly
/// useful in permutation-based problems where the relative ordering of genes is important.
///
/// # Arguments
/// * `genome` - The sequence of genes to be mutated. This could represent a solution in the
///    problem space.
/// * `rng` - A mutable reference to a random number generator implementing the Rng trait
///
/// # Note
/// Insertion mutation is applicable to permutation-based problems.
///
/// # Example
/// ```
/// use rusty_genes::*;
/// use rand::prelude::*;
///
/// let mut rng = SmallRng::seed_from_u64(999);
/// let mut genome = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
///
/// insertion_mutation(&mut genome, &mut rng);
///
/// assert_eq!(genome, vec![0, 1, 6, 2, 3, 4, 5, 7, 8, 9]);
/// ```
pub fn insertion_mutation<G, R: Rng>(genome: &mut [G], rng: &mut R) {
    let n = genome.len();
    if n < 2 {
        return;
    }

    let (mut p1, mut p2) = random_pair(n, rng);
    if p1 > p2 {
        std::mem::swap(&mut p1, &mut p2);
    }
    genome[p1..=p2].rotate_right(1);
}

/// Selects a gene from a random position in the genome and copies it to another position, replacing
/// the gene that was previously there.
///
/// This mutation operator can introduce variation into the genome, and can be useful in problems where
/// duplications of certain genes may lead to a better solution.
///
/// # Arguments
/// * `genome` - The sequence of genes to be mutated. This could represent a solution in the
///    problem space.
/// * `rng` - A mutable reference to a random number generator implementing the Rng trait
///
/// # Note
/// Duplicate mutation is not applicable to permutation-based problems where each gene should be unique.
///
/// # Example
/// ```
/// use rusty_genes::*;
/// use rand::prelude::*;
///
/// let mut rng = SmallRng::seed_from_u64(999);
/// let mut genome = vec!['G', 'T', 'A', 'G', 'C'];
///
/// duplicate_mutation(&mut genome, &mut rng);
///
/// assert_eq!(genome, vec!['G', 'T', 'A', 'A', 'C']);
/// ```
pub fn duplicate_mutation<G: Copy, R: Rng>(genome: &mut [G], rng: &mut R) {
    let n = genome.len();
    if n < 2 {
        return;
    }

    let (p1, p2) = random_pair(n, rng);
    genome[p2] = genome[p1];
}

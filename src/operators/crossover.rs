use super::*;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    ops::Range,
};

/// Defines various crossover strategies for combining genetic information of two individuals.
///
/// These strategies are mainly applicable to permutation-based problems where each gene in a
/// genome represents a unique element of the solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossoverStrategy {
    /// Generalized Partition Crossover 2 (GPX2) is a permutation-based method that identifies common
    /// sequences of genes appearing in the same order (partitions) in the parent genomes. It then
    /// builds a graph where the nodes represent these common partitions, and the edges represent
    /// possible transitions between these partitions. This graph is used to generate offspring
    /// genomes, preserving the common partitions from the parents and exploring new combinations of
    /// these partitions.
    GeneralizedPartition2,

    /// Sequential Constructive Crossover (SCX) is a permutation-based method that selects the best
    /// next gene from the parent genomes by prioritizing genes that appear earlier in either parent
    /// genome, while ensuring that each gene appears exactly once in the offspring.
    SequentialConstructive,

    /// Edge Recombination Crossover (ERX) is a permutation-based method that creates a neighbor
    /// list for each gene based on its neighbors in the parent genomes, and then constructs the
    /// offspring by iteratively selecting the next gene with the shortest neighbor list.
    EdgeRecombination,

    /// Partially Mapped Crossover (PMX) is a permutation-based method that creates a mapping
    /// between the parent genomes to generate a new offspring while preserving relative orderings.
    PartiallyMapped,

    /// Order Crossover (OX) is a permutation-based method that preserves the relative order of
    /// genes from one parent while filling in the gaps with genes from the other parent.
    Order,

    /// Cycle Crossover (CX) is a permutation-based method that identifies cycles between the parent
    /// genomes and combines them to create the offspring, ensuring that each gene appears exactly
    /// once in the offspring.
    ///
    /// It is worth to mention that unlike the other methods, no randomness is involved in CX.
    Cycle,

    /// Single-Point Crossover (SPX) is a method that selects a single point on the genome. Every
    /// gene before this point is copied from the first parent, and genes from this point forward
    /// is copied from the second parent.
    SinglePoint,
}

impl CrossoverStrategy {
    /// Performs a crossover operation between two parent genomes using the selected crossover
    /// strategy to produce a new offspring.
    ///
    /// # Arguments
    /// * `parent1` - A reference to the first parent's genome.
    /// * `parent2` - A reference to the second parent's genome.
    /// * `rng` - A mutable reference to a random number generator implementing the Rng trait
    ///
    /// # Returns
    /// A new `Vec<G>` representing the offspring's genome. The length of this genome will be
    /// the same as the parents' genomes.
    ///
    /// # Panics
    /// This method will panic if the parent genomes have different lengths.
    pub fn crossover<G: Copy + PartialOrd + Eq + Hash, R: Rng>(
        &self,
        parent1: &[G],
        parent2: &[G],
        rng: &mut R,
    ) -> Vec<G> {
        match self {
            Self::GeneralizedPartition2 => {
                generalized_partition_2_crossover(parent1, parent2, |_, partitions| {
                    rng.gen_range(0..partitions.len())
                })
            }
            Self::SequentialConstructive => {
                let pivot = parent1.choose(rng).copied().unwrap();
                sequential_constructive_crossover(parent1, parent2, pivot)
            }
            Self::EdgeRecombination => {
                let pivot = parent1.choose(rng).copied().unwrap();
                edge_recombination_crossover(parent1, parent2, pivot)
            }
            Self::PartiallyMapped => {
                let range = random_range(parent1.len(), rng);
                partially_mapped_crossover(parent1, parent2, range)
            }
            Self::Order => {
                let range = random_range(parent1.len(), rng);
                order_crossover(parent1, parent2, range)
            }
            Self::Cycle => cycle_crossover(parent1, parent2),
            Self::SinglePoint => {
                let point = rng.gen_range(0..parent1.len());
                single_point_crossover(parent1, parent2, point)
            }
        }
    }
}

/// Performs a Generalized Partition Crossover 2 (GPX2) operation between two parent genomes.
///
/// Generalized Partition Crossover 2 (GPX2) identifies common partitions (sequences of genes that
/// are common to both genomes) between the two parent genomes and then assembles a new offspring
/// genome from these partitions. The order in which the partitions are selected and added to the
/// offspring genome is determined by the provided partition selection function.
///
/// The GPX2 crossover algorithm is designed to preserve common structures (partitions) between the
/// parents in the offspring, which can be beneficial in many optimization problems.
///
/// # Arguments
/// * `parent1` - A reference to the first parent's genome.
/// * `parent2` - A reference to the second parent's genome.
/// * `partition_selector` - A function to select which partition to add to the offspring genome
///   next. The function is given an optional reference to the last gene added to the offspring
///   genome and a slice of the remaining partitions. It should return the index of the partition
///   to add next.
///
/// # Returns
/// A `Vec<G>` representing the offspring's genome. The length of this genome will be the
/// same as the parents' genomes.
///
/// # Note
/// GPX2 is applicable to permutation-based problems.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let parent1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let parent2 = vec![8, 9, 0, 7, 4, 5, 6, 3, 2, 1];
///
/// let offspring1 = generalized_partition_2_crossover(&parent1, &parent2, |_last, _partitions| 0);
/// assert_eq!(offspring1, vec![0, 9, 8, 7, 4, 5, 6, 1, 2, 3]);
///
/// let offspring2 = generalized_partition_2_crossover(&parent1, &parent2, |last, partitions| {
///     last.and_then(|last_selected_gene| {
///         let preferred_gene = (last_selected_gene + 3) % 10;
///         partitions
///             .iter()
///             .enumerate()
///             .find(|(i, partition)| partition.contains(&preferred_gene))
///             .map(|(i, _)| i)
///     }).unwrap_or(0)
/// });
/// assert_eq!(offspring2, vec![0, 9, 8, 1, 2, 3, 4, 5, 6, 7]);
/// ```
pub fn generalized_partition_2_crossover<G, F>(
    parent1: &[G],
    parent2: &[G],
    mut partition_selector: F,
) -> Vec<G>
where
    G: Copy + PartialOrd + Eq + Hash,
    F: FnMut(Option<&G>, &[Vec<G>]) -> usize,
{
    let mut offspring = Vec::with_capacity(parent1.len());

    let mut genes_in_selected_partition = HashSet::with_capacity(offspring.len());
    let mut partitions = common_partitions(parent1, parent2);
    while !partitions.is_empty() {
        // Select a partition
        let partition_index = partition_selector(offspring.last(), &partitions);
        let mut partition = partitions.swap_remove(partition_index);

        // Make a copy of the partition as a set for fast lookup
        genes_in_selected_partition.clear();
        genes_in_selected_partition.extend(partition.iter().copied());

        // Add the selected partition to the offspring
        offspring.append(&mut partition);

        // Remove the selected partition from all other partitions
        for other_partition in &mut partitions {
            other_partition.retain(|gene| !genes_in_selected_partition.contains(gene));
        }

        // Remove empty partitions
        partitions.retain(|partition| !partition.is_empty());
    }
    offspring
}

/// Performs a Sequential Constructive Crossover (SCX) operation between two parent genomes.
///
/// Sequential Constructive Crossover (SCX) selects the best next/ gene from the parent genomes
/// by prioritizing genes that appear earlier in either parent genome, while ensuring that each
/// gene appears exactly once in the offspring.
///
/// # Arguments
/// * `parent1` - A reference to the first parent's genome.
/// * `parent2` - A reference to the second parent's genome.
/// * `pivot` - The gene used to start constructing the offspring genome.
///
/// # Returns
/// A new `Vec<G>` representing the offspring's genome. The length of this genome will be
/// the same as the parents' genomes.
///
/// # Note
/// SCX is applicable to permutation-based problems.
///
/// # Panics
/// This method will panic if the parent genomes have different lengths.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let parent1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let parent2 = vec![8, 9, 0, 7, 4, 5, 6, 3, 2, 1];
///
/// let offspring = sequential_constructive_crossover(&parent1, &parent2, 2);
/// assert_eq!(offspring, vec![2, 1, 8, 9, 0, 7, 4, 5, 6, 3]);
/// ```
pub fn sequential_constructive_crossover<G>(parent1: &[G], parent2: &[G], pivot: G) -> Vec<G>
where
    G: Copy + Eq + Hash,
{
    assert!(parent1.len() == parent2.len());

    let n = parent1.len();
    if n < 2 {
        return parent1.to_vec();
    }

    let mut visited = HashSet::with_capacity(n);

    let mut current = pivot;
    let mut offspring = vec![pivot; n];
    for next in offspring.iter_mut().skip(1) {
        visited.insert(current);

        // Find the position of the current gene in each parent genome
        let current_pos1 = find_position(&current, parent1).unwrap();
        let current_pos2 = find_position(&current, parent2).unwrap();

        // Select the next gene in each parent genome
        let next1 = parent1[if current_pos1 + 1 != n {
            current_pos1 + 1
        } else {
            0
        }];
        let next2 = parent2[if current_pos2 + 1 != n {
            current_pos2 + 1
        } else {
            0
        }];

        // Select the next gene in the offspring based on the visited genes and order
        // of appearance in parent genomes
        let is_next1_visited = visited.contains(&next1);
        let is_next2_visited = visited.contains(&next2);
        *next = if is_next1_visited && is_next2_visited {
            // Both genes are visited; select an unvisited one
            parent1
                .iter()
                .find(|gene| !visited.contains(gene))
                .copied()
                .unwrap()
        } else if is_next1_visited {
            // Gene in parent1 is visited; select parent2's gene
            next2
        } else if is_next2_visited {
            // Gene in parent2 is visited; select parent1's gene
            next1
        } else {
            // Both genes are unvisited; select the one that appeared first
            let next1_pos2 = find_position(&next1, parent2).unwrap();
            let next2_pos1 = find_position(&next2, parent1).unwrap();
            if next1_pos2 < next2_pos1 {
                next1
            } else {
                next2
            }
        };

        // Set the current gene to the selected next gene
        current = *next;
    }
    offspring
}

/// Performs an Edge Recombination Crossover (ERX) operation between two parent genomes.
///
/// Edge Recombination Crossover (ERX) creates a neighbor list for each gene based on its
/// neighbors in the parent genomes, and then constructs the offspring by iteratively
/// selecting the next gene with the shortest neighbor list.
///
/// # Arguments
/// * `parent1` - A reference to the first parent's genome.
/// * `parent2` - A reference to the second parent's genome.
/// * `pivot` - The gene used to start constructing the offspring genome.
///
/// # Returns
/// A new `Vec<G>` representing the offspring's genome. The length of this genome will be
/// the same as the parents' genomes.
///
/// # Note
/// ERX is applicable to permutation-based problems.
///
/// # Panics
/// This method will panic if the parent genomes have different lengths.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let parent1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let parent2 = vec![8, 9, 0, 7, 4, 5, 6, 3, 2, 1];
///
/// let mut offspring = edge_recombination_crossover(&parent1, &parent2, 2);
///
/// // Because the order of items in an HashMap's iterator are not predictable,
/// // we test that the offspring has the same set of genes of its parents,
/// // without duplicates.
/// assert_eq!(offspring.len(), parent1.len());
/// offspring.sort_unstable();
/// offspring.dedup();
/// assert_eq!(offspring.len(), parent1.len());
/// assert!(offspring.iter().all(|gene| parent1.contains(gene)));
/// ```
pub fn edge_recombination_crossover<G>(parent1: &[G], parent2: &[G], pivot: G) -> Vec<G>
where
    G: Copy + Eq + Hash,
{
    assert!(parent1.len() == parent2.len());

    let n = parent1.len();
    if n < 2 {
        return parent1.to_vec();
    }

    // Initialize the neighbor list for each gene
    let mut neighbor_lists = HashMap::with_capacity(n);
    [parent1, parent2]
        .iter()
        .flat_map(|genome| {
            genome.iter().enumerate().map(|(i, &gene)| {
                let prev = if i != 0 { i - 1 } else { n - 1 };
                let next = if i + 1 != n { i + 1 } else { 0 };
                (gene, [genome[prev], genome[next]])
            })
        })
        .for_each(|(gene, neighbors)| {
            neighbor_lists
                .entry(gene)
                .or_insert_with(HashSet::new)
                .extend(neighbors);
        });

    let mut visited = HashSet::with_capacity(n);

    let mut current = pivot;
    let mut offspring = vec![pivot; n];
    for next in offspring.iter_mut().skip(1) {
        visited.insert(current);

        // Remove the current gene from the neighbor list of all genes
        for neighbors in neighbor_lists.values_mut() {
            neighbors.remove(&current);
        }

        // Select the next gene with the shortest neighbor list
        *next = neighbor_lists[&current]
            .iter()
            .filter(|gene| !visited.contains(gene))
            .min_by_key(|&gene| neighbor_lists[gene].len())
            .copied()
            .unwrap_or_else(|| {
                // No gene is found in the current neighbor list
                // Select an unvisited gene
                parent1
                    .iter()
                    .find(|gene| !visited.contains(gene))
                    .copied()
                    .unwrap()
            });

        current = *next;
    }
    offspring
}

/// Performs a Partially Mapped Crossover (PMX) operation between two parent genomes.
///
/// Partially Mapped Crossover (PMX) creates a mapping between the parent genomes to
/// generate a new offspring while preserving relative orderings.
///
/// # Arguments
/// * `parent1` - A reference to the first parent's genome.
/// * `parent2` - A reference to the second parent's genome.
/// * `range` - The range of genes to be mapped from `parent1` to the offspring.
///
/// # Returns
/// A new `Vec<G>` representing the offspring's genome. The length of this genome will be
/// the same as the parents' genomes.
///
/// # Note
/// PMX is applicable to permutation-based problems.
///
/// # Panics
/// This method will panic if the parent genomes have different lengths.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let parent1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let parent2 = vec![8, 9, 0, 7, 4, 5, 6, 3, 2, 1];
///
/// let offspring = partially_mapped_crossover(&parent1, &parent2, 1..4);
/// assert_eq!(offspring, vec![8, 1, 2, 3, 4, 5, 6, 7, 0, 9]);
/// ```
pub fn partially_mapped_crossover<G>(parent1: &[G], parent2: &[G], range: Range<usize>) -> Vec<G>
where
    G: Copy + PartialEq,
{
    assert!(parent1.len() == parent2.len());

    let n = parent1.len();
    if n < 2 {
        return parent1.to_vec();
    }

    let p1 = range.start;
    let p2 = range.end;
    let mut offspring = vec![parent1[0]; n];

    // Copy the genes from the range in parent1 to the offspring
    offspring[p1..p2].copy_from_slice(&parent1[p1..p2]);

    // Map the remaining genes to the offspring while preserving relative orderings
    let start = &parent1[p1..p2];
    let end = &parent2[p1..p2];
    for (i, gene) in offspring.iter_mut().enumerate() {
        if !range.contains(&i) {
            let mut mapped_gene = parent2[i];
            while let Some(pos) = find_position(&mapped_gene, start) {
                mapped_gene = end[pos];
            }
            *gene = mapped_gene
        }
    }

    offspring
}

/// Performs an Order Crossover (OX) operation between two parent genomes.
///
/// Order Crossover (OX) preserves the relative order of genes from one parent while
/// filling in the gaps with genes from the other parent.
///
/// # Arguments
/// * `parent1` - A reference to the first parent's genome.
/// * `parent2` - A reference to the second parent's genome.
/// * `range` - The range of genes to be mapped from `parent1` to the offspring.
///
/// # Returns
/// A new `Vec<G>` representing the offspring's genome. The length of this genome will be
/// the same as the parents' genomes.
///
/// # Note
/// OX is applicable to permutation-based problems.
///
/// # Panics
/// This method will panic if the parent genomes have different lengths.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let parent1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let parent2 = vec![8, 9, 0, 7, 4, 5, 6, 3, 2, 1];
///
/// let offspring = order_crossover(&parent1, &parent2, 1..4);
/// assert_eq!(offspring, vec![7, 1, 2, 3, 4, 5, 6, 8, 9, 0]);
/// ```
pub fn order_crossover<G>(parent1: &[G], parent2: &[G], range: Range<usize>) -> Vec<G>
where
    G: Copy + PartialEq,
{
    assert!(parent1.len() == parent2.len());

    let n = parent1.len();
    if n < 2 {
        return parent1.to_vec();
    }

    let p1 = range.start;
    let p2 = range.end;
    let mut genome = vec![parent1[0]; n];

    // Copy the genes from the range in parent1 to the offspring
    genome[p1..p2].copy_from_slice(&parent1[p1..p2]);

    // Fill the remaining gaps in the offspring with genes from parent2 while preserving
    // the relative order of genes from parent1
    let mut index1 = if p2 != n { p2 } else { 0 };
    let mut index2 = index1;
    while index1 != p1 {
        if !genome[p1..p2].contains(&parent2[index2]) {
            genome[index1] = parent2[index2];
            index1 += 1;
            if index1 == n {
                index1 = 0;
            }
        }
        index2 += 1;
        if index2 == n {
            index2 = 0;
        }
    }

    genome
}

/// Performs a Cycle Crossover (CX) operation between two parent genomes.
///
/// Cycle Crossover (CX) identifies cycles between the parent genomes and combines them
/// to create the offspring, ensuring that each gene appears exactly once in the offspring.
///
/// # Arguments
/// * `parent1` - A reference to the first parent's genome.
/// * `parent2` - A reference to the second parent's genome.
///
/// # Returns
/// A new `Vec<G>` representing the offspring's genome. The length of this genome will be
/// the same as the parents' genomes.
///
/// # Note
/// CX is applicable to permutation-based problems.
///
/// # Panics
/// This method will panic if the parent genomes have different lengths.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let parent1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let parent2 = vec![8, 9, 0, 7, 4, 5, 6, 3, 2, 1];
///
/// let offspring = cycle_crossover(&parent1, &parent2);
/// assert_eq!(offspring, vec![0, 9, 2, 3, 4, 5, 6, 7, 8, 1]);
/// ```
pub fn cycle_crossover<G>(parent1: &[G], parent2: &[G]) -> Vec<G>
where
    G: Copy + PartialEq,
{
    assert!(parent1.len() == parent2.len());

    let n = parent1.len();
    if n < 2 {
        return parent1.to_vec();
    }

    let mut offspring = vec![parent1[0]; n];
    let mut visited = vec![false; n];

    let mut cycle_start = 0;
    let mut first_parent = true;
    // Find the cycles between the parent genomes and combine them to create the offspring
    while cycle_start < n {
        let mut index = cycle_start;

        // Find the next cycle
        while !visited[index] {
            visited[index] = true;

            offspring[index] = if first_parent {
                parent1[index]
            } else {
                parent2[index]
            };

            // Find the index of the current element in the other parent
            index = if first_parent {
                find_position(&parent1[index], parent2).unwrap()
            } else {
                find_position(&parent2[index], parent1).unwrap()
            };
        }

        // Switch the parent for the next cycle
        first_parent = !first_parent;

        // Find the next unvisited element
        if let Some(next_start) = visited.iter().position(|&x| !x) {
            cycle_start = next_start;
        } else {
            break;
        }
    }

    offspring
}

/// Performs a Single-Point Crossover (SPX) operation between two parent genomes.
///
/// Single-Point Crossover (SPX) swaps all genes beyond a selected point on the first
/// parent genome with all the genes from the same point in the second parent genome.
///
/// # Arguments
/// * `parent1` - A reference to the first parent's genome.
/// * `parent2` - A reference to the second parent's genome.
/// * `point` - The crossover point where the genes should be swapped.
///
/// # Returns
/// A new `Vec<G>` representing the offspring's genome. The length of this genome will be
/// the same as the parents' genomes.
///
/// # Note
/// SPX is not applicable to permutation-based problems where each gene should be unique.
///
/// # Panics
/// This method will panic if the parent genomes have different lengths.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let parent1 = vec!['a', 'b', 'c', 'd', 'e'];
/// let parent2 = vec!['v', 'w', 'x', 'y', 'z'];
///
/// let offspring = single_point_crossover(&parent1, &parent2, 3);
/// assert_eq!(offspring, vec!['a', 'b', 'c', 'y', 'z']);
/// ```
pub fn single_point_crossover<G>(parent1: &[G], parent2: &[G], point: usize) -> Vec<G>
where
    G: Copy,
{
    assert!(parent1.len() == parent2.len());

    let mut offspring = parent1.to_vec();
    if point < parent2.len() {
        offspring[point..].copy_from_slice(&parent2[point..]);
    }
    offspring
}

#[inline]
fn find_position<G: PartialEq>(target_gene: &G, genome: &[G]) -> Option<usize> {
    genome.iter().position(|gene| gene == target_gene)
}

use super::*;
use std::{hash::Hash, ops::Range};

/// Defines various crossover strategies for combining genetic information of two individuals.
///
/// These strategies are mainly applicable to permutation-based problems where each gene in a
/// genome represents a unique element of the solution. For genomes in which genes repeat, see
/// [`crossover_with_repetition`](CrossoverStrategy::crossover_with_repetition).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CrossoverStrategy {
    /// Keeps chains of neighbors shared by both parents and joins them in a random order.
    /// See [`common_partition_crossover`] to choose the partition order yourself.
    CommonPartition,

    /// Starts at a random gene and follows successors from either parent.
    /// See [`successor_crossover`] for how it chooses between the successors.
    Successor,

    /// Starts at a random gene and prefers neighbors with the fewest remaining links, breaking
    /// ties and dead ends randomly. See [`edge_recombination_crossover`] (ERX).
    EdgeRecombination,

    /// Partially Mapped Crossover (PMX) is a permutation-based method that creates a mapping
    /// between the parent genomes to generate a new offspring while preserving relative orderings.
    PartiallyMapped,

    /// Order Crossover (OX) is a permutation-based method that preserves the relative order of
    /// genes from one parent while filling in the gaps with genes from the other parent.
    Order,

    /// Takes the cycle through a random position from the first parent and all other positions
    /// from the second. See [`cycle_crossover`] (CX).
    Cycle,

    /// Joins the first parent's prefix to the second's suffix at a random cut between genes.
    /// A single-gene genome is copied from the first parent. See [`single_point_crossover`] (SPX).
    SinglePoint,
}

impl CrossoverStrategy {
    /// Performs a crossover operation between two parent genomes using the selected crossover
    /// strategy to produce a new offspring.
    ///
    /// # Arguments
    /// * `parent1` - A reference to the first parent's genome.
    /// * `parent2` - A reference to the second parent's genome.
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// # Returns
    /// A new `Vec<G>` representing the offspring's genome. The length of this genome will be
    /// the same as the parents' genomes.
    ///
    /// # Panics
    /// Panics if the parent genomes have different lengths. Strategies other than
    /// single-point crossover may also panic if the parents are not permutations of the same
    /// genes.
    pub fn crossover<G: Copy + PartialOrd + Eq + Hash>(
        &self,
        parent1: &[G],
        parent2: &[G],
        rng: &mut Rng,
    ) -> Vec<G> {
        assert!(parent1.len() == parent2.len());
        if parent1.is_empty() {
            return Vec::new();
        }

        match self {
            Self::CommonPartition => {
                common_partition_crossover(parent1, parent2, |_, partitions| {
                    rng.random_range(0..partitions.len())
                })
            }
            Self::Successor => {
                let pivot = *rng.choose(parent1).unwrap();
                successor_crossover(parent1, parent2, pivot)
            }
            Self::EdgeRecombination => {
                let pivot = *rng.choose(parent1).unwrap();
                edge_recombination_crossover(parent1, parent2, pivot, rng)
            }
            Self::PartiallyMapped => {
                let range = random_range(parent1.len(), rng);
                partially_mapped_crossover(parent1, parent2, range)
            }
            Self::Order => {
                let range = random_range(parent1.len(), rng);
                order_crossover(parent1, parent2, range)
            }
            Self::Cycle => {
                let start = rng.random_range(0..parent1.len());
                cycle_crossover(parent1, parent2, start)
            }
            Self::SinglePoint => {
                // The point is between two genes, so the offspring takes at least one gene of
                // each parent; a single gene has no such point and is copied from parent1.
                let point = rng.random_range(1..parent1.len().max(2));
                single_point_crossover(parent1, parent2, point)
            }
        }
    }

    /// Combines equal-length parents by distinguishing repeated genes by their occurrence order.
    ///
    /// The copies of a gene are told apart by their order: the k-th copy of a gene in one parent
    /// stands for its k-th copy in the other, and the strategy combines the parents as if the
    /// copies were distinct genes. This suits genomes in which that order has a meaning, such as
    /// a schedule in which the k-th copy of a job is its k-th operation. The strategies that
    /// follow neighboring genes, such as edge recombination and common partition crossover, see
    /// neighbors between copies rather than between genes: parents in which the same two genes
    /// are neighbors through different copies do not share that pair of neighbors.
    ///
    /// Without repeated genes, the offspring is the one [`crossover`](Self::crossover) produces
    /// from the same state of `rng`.
    ///
    /// The offspring has the parents' length. For permutation strategies, the parents must
    /// contain the same number of copies of every gene. Single-point crossover does not
    /// preserve those counts.
    ///
    /// # Panics
    /// Panics if the parent genomes have different lengths. Strategies other than
    /// single-point crossover may also panic if the parents do not have the same genes, each
    /// repeated the same number of times.
    ///
    /// # Example
    /// ```
    /// use rusty_genes::*;
    ///
    /// // Three jobs of two operations each, in the order that two schedules process them.
    /// let parent1 = vec![0, 0, 1, 1, 2, 2];
    /// let parent2 = vec![2, 1, 0, 2, 1, 0];
    ///
    /// let mut offspring = CrossoverStrategy::Order.crossover_with_repetition(
    ///     &parent1,
    ///     &parent2,
    ///     &mut Rng::from_seed(1),
    /// );
    ///
    /// // The offspring schedules every operation of every job.
    /// offspring.sort_unstable();
    /// assert_eq!(offspring, parent1);
    /// ```
    pub fn crossover_with_repetition<G: Copy + PartialOrd + Eq + Hash>(
        &self,
        parent1: &[G],
        parent2: &[G],
        rng: &mut Rng,
    ) -> Vec<G> {
        let offspring = self.crossover(&numbered(parent1), &numbered(parent2), rng);
        offspring.into_iter().map(|(gene, _)| gene).collect()
    }
}

/// Joins the parents' [`common_partitions`] without breaking the chains inside them.
///
/// Parents must be permutations of the same distinct genes. `partition_selector` receives
/// the last gene appended, or [`None`] initially, and the remaining partitions. It must return
/// an index into that slice. The offspring has the parents' length; the new links between
/// partitions need not appear in either parent.
///
/// # Panics
/// Panics if the parent genomes have different lengths, or if `parent1` has a
/// gene more than once, or if `partition_selector` returns an out-of-bounds index.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let parent1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let parent2 = vec![8, 9, 0, 7, 4, 5, 6, 3, 2, 1];
///
/// let offspring1 = common_partition_crossover(&parent1, &parent2, |_last, _partitions| 0);
/// assert_eq!(offspring1, vec![0, 9, 8, 7, 4, 5, 6, 1, 2, 3]);
///
/// let offspring2 = common_partition_crossover(&parent1, &parent2, |last, partitions| {
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
pub fn common_partition_crossover<G, F>(
    parent1: &[G],
    parent2: &[G],
    mut partition_selector: F,
) -> Vec<G>
where
    G: Copy + PartialOrd + Eq + Hash,
    F: FnMut(Option<&G>, &[Vec<G>]) -> usize,
{
    assert!(parent1.len() == parent2.len());

    let mut offspring = Vec::with_capacity(parent1.len());

    // The partitions are disjoint and non-empty, so each selected partition can be appended
    // to the offspring as a whole.
    let mut partitions = common_partitions(parent1, parent2);
    while !partitions.is_empty() {
        let partition_index = partition_selector(offspring.last(), &partitions);
        let mut partition = partitions.swap_remove(partition_index);
        offspring.append(&mut partition);
    }
    // The partitions hold each gene of parent1 once, so a repeated gene would be lost.
    assert!(
        offspring.len() == parent1.len(),
        "the parents should be permutations of the same genes"
    );
    offspring
}

/// Builds a permutation from `pivot` by following the successors of each gene in its parents.
///
/// Successor Crossover starts the offspring with `pivot` and adds one gene at a time, treating each
/// parent genome as a cycle. It considers the successors of the last added gene in the two
/// parents:
/// * If only one of them is not yet in the offspring, it adds that one.
/// * If neither is in the offspring yet, it adds the successor in `parent1` when that gene is at
///   an earlier position in `parent2` than the successor in `parent2` is in `parent1`, and the
///   successor in `parent2` otherwise.
/// * If both are in the offspring already, it adds the first gene of `parent1` that is not.
///
/// Parents must be permutations of the same distinct genes. Genomes with fewer than two
/// genes are copied from `parent1` without using `pivot`.
///
/// # Panics
/// Panics if the parent genomes have different lengths, or if `pivot` is not a
/// gene of genomes that have at least two genes.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let parent1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let parent2 = vec![8, 9, 0, 7, 4, 5, 6, 3, 2, 1];
///
/// let offspring = successor_crossover(&parent1, &parent2, 2);
/// assert_eq!(offspring, vec![2, 1, 8, 9, 0, 7, 4, 5, 6, 3]);
/// ```
pub fn successor_crossover<G>(parent1: &[G], parent2: &[G], pivot: G) -> Vec<G>
where
    G: Copy + Eq + Hash,
{
    assert!(parent1.len() == parent2.len());

    let n = parent1.len();
    if n < 2 {
        return parent1.to_vec();
    }

    let positions1 = positions(parent1);
    let positions2 = positions(parent2);
    let mut visited = GeneSet::with_capacity_and_hasher(n, Default::default());
    // Every gene of parent1 before this position is visited.
    let mut first_unvisited = 0;

    let mut current = pivot;
    let mut offspring = vec![pivot; n];
    for next in offspring.iter_mut().skip(1) {
        visited.insert(current);

        // Find the position of the current gene in each parent genome
        let current_pos1 = positions1[&current];
        let current_pos2 = positions2[&current];

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
            // Both genes are visited; select the first unvisited one of parent1
            while visited.contains(&parent1[first_unvisited]) {
                first_unvisited += 1;
            }
            parent1[first_unvisited]
        } else if is_next1_visited {
            // Gene in parent1 is visited; select parent2's gene
            next2
        } else if is_next2_visited {
            // Gene in parent2 is visited; select parent1's gene
            next1
        } else {
            // Both genes are unvisited; select the one that appeared first
            let next1_pos2 = positions2[&next1];
            let next2_pos1 = positions1[&next2];
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
/// Edge Recombination Crossover (ERX) creates a neighbor list for each gene from its neighbors in
/// both parent genomes, treating each genome as a cycle. Starting from `pivot`, it removes the
/// current gene from every list and continues with the neighbor of the current gene that has the
/// fewest neighbors left, choosing at random among neighbors with equally few. When the current
/// gene has no neighbor left, it continues with a random gene not yet in the offspring.
///
/// Parents must be permutations of the same distinct genes. Genomes with fewer than two
/// genes are copied from `parent1` without using `pivot`.
///
/// # Panics
/// Panics if the parent genomes have different lengths, or if `pivot` is not a
/// gene of genomes that have at least two genes.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let parent1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let parent2 = vec![8, 9, 0, 7, 4, 5, 6, 3, 2, 1];
///
/// let mut offspring = edge_recombination_crossover(&parent1, &parent2, 2, &mut Rng::from_seed(1));
///
/// // The offspring has every gene of its parents exactly once.
/// assert_eq!(offspring.len(), parent1.len());
/// offspring.sort_unstable();
/// offspring.dedup();
/// assert_eq!(offspring.len(), parent1.len());
/// assert!(offspring.iter().all(|gene| parent1.contains(gene)));
/// ```
pub fn edge_recombination_crossover<G>(
    parent1: &[G],
    parent2: &[G],
    pivot: G,
    rng: &mut Rng,
) -> Vec<G>
where
    G: Copy + Eq + Hash,
{
    assert!(parent1.len() == parent2.len());

    let n = parent1.len();
    if n < 2 {
        return parent1.to_vec();
    }

    // Two cyclic parents contribute at most four distinct neighbors per gene.
    let mut neighbor_lists: GeneMap<G, Vec<G>> =
        GeneMap::with_capacity_and_hasher(n, Default::default());
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
            let list = neighbor_lists.entry(gene).or_default();
            for neighbor in neighbors {
                if !list.contains(&neighbor) {
                    list.push(neighbor);
                }
            }
        });

    // The genes not yet in the offspring, and the position of each in that list, so a random
    // one can be drawn and removed in constant time.
    let mut unvisited = parent1.to_vec();
    let mut unvisited_positions = positions(parent1);

    let mut current = pivot;
    let mut offspring = vec![pivot; n];
    for next in offspring.iter_mut().skip(1) {
        let position = unvisited_positions
            .remove(&current)
            .expect("the pivot should be a gene of the parents");
        unvisited.swap_remove(position);
        if let Some(&moved) = unvisited.get(position) {
            unvisited_positions.insert(moved, position);
        }

        // Remove the current gene from the neighbor list of all genes. Neighborhood is
        // symmetric, so only the neighbors of the current gene have it in their lists.
        let current_neighbors = neighbor_lists
            .get_mut(&current)
            .map(std::mem::take)
            .unwrap_or_default();
        for neighbor in &current_neighbors {
            if let Some(neighbors) = neighbor_lists.get_mut(neighbor) {
                neighbors.retain(|gene| *gene != current);
            }
        }

        let neighbors = current_neighbors
            .iter()
            .filter(|&gene| unvisited_positions.contains_key(gene));
        let fewest = neighbors
            .clone()
            .map(|gene| neighbor_lists[gene].len())
            .min();
        *next = match fewest {
            Some(fewest) => {
                let mut candidates = neighbors.filter(|&gene| neighbor_lists[gene].len() == fewest);
                let count = candidates.clone().count();
                let skip = if count > 1 {
                    rng.random_range(0..count)
                } else {
                    0
                };
                *candidates.nth(skip).unwrap()
            }
            None => unvisited[rng.random_range(0..unvisited.len())],
        };

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
/// Panics if the parent genomes have different lengths, if `range` is not within
/// the bounds of genomes that have at least two genes, or if the genes of `range` cannot be
/// mapped because the parents are not permutations of the same genes.
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
    G: Copy + Eq + Hash,
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
    let start = GenePositions::new(&parent1[p1..p2]);
    let end = &parent2[p1..p2];
    for (i, gene) in offspring.iter_mut().enumerate() {
        if !range.contains(&i) {
            let mut mapped_gene = parent2[i];
            // Between permutations of the same genes, the mapping leaves the range within as
            // many steps as the range has genes; otherwise it can return to a gene forever.
            let mut steps = 0;
            while let Some(pos) = start.get(&mapped_gene) {
                steps += 1;
                assert!(
                    steps <= end.len(),
                    "the parents should be permutations of the same genes"
                );
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
/// * `range` - The range of genes to be copied from `parent1` to the offspring. If the range is
///   empty, the offspring is a copy of `parent2`.
///
/// # Returns
/// A new `Vec<G>` representing the offspring's genome. The length of this genome will be
/// the same as the parents' genomes.
///
/// # Note
/// OX is applicable to permutation-based problems.
///
/// # Panics
/// Panics if the parent genomes have different lengths, if `range` is not
/// within the bounds of the genomes, or if every gene of `parent2` is one of the genes that
/// `range` copies from `parent1`, because the parents are not permutations of the same genes.
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
    G: Copy + Eq + Hash,
{
    assert!(parent1.len() == parent2.len());
    assert!(range.start <= range.end && range.end <= parent1.len());

    let n = parent1.len();
    if range.is_empty() {
        return parent2.to_vec();
    }
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
    let copied_genes = GenePositions::new(&parent1[p1..p2]);
    let mut index1 = if p2 != n { p2 } else { 0 };
    let mut index2 = index1;
    // Between permutations of the same genes, fewer than `n` genes in a row of parent2 are
    // copied ones; otherwise a whole pass over parent2 can find no gene to fill a gap with, and
    // then no later pass can.
    let mut skipped = 0;
    while index1 != p1 {
        if !copied_genes.contains(&parent2[index2]) {
            genome[index1] = parent2[index2];
            index1 += 1;
            if index1 == n {
                index1 = 0;
            }
            skipped = 0;
        } else {
            skipped += 1;
            assert!(
                skipped < n,
                "the parents should be permutations of the same genes"
            );
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
/// Cycle Crossover (CX) keeps every gene in the position it has in one of the parents. Starting
/// at position `start`, it takes the gene of `parent1`, finds the position of that gene in
/// `parent2`, and repeats from there until it returns to `start`. The offspring takes the genes
/// of `parent1` at the positions of this cycle, and the genes of `parent2` everywhere else.
///
/// Parents must be permutations of the same distinct genes. `start` chooses the cycle; if
/// both parents have the same gene there, that cycle has one position and the result equals
/// `parent2`. Empty parents return an empty genome without using `start`.
///
/// # Panics
/// Panics if the parent genomes have different lengths, if they are not empty and
/// `start` is not one of their positions, or if a gene of the cycle in `parent1` is missing from
/// `parent2` or the cycle does not return to `start` because the parents are not permutations
/// of the same genes.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let parent1 = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// let parent2 = vec![8, 9, 0, 7, 4, 5, 6, 3, 2, 1];
///
/// // The cycle through position 1 is the positions 1 and 9.
/// let offspring = cycle_crossover(&parent1, &parent2, 1);
/// assert_eq!(offspring, vec![8, 1, 0, 7, 4, 5, 6, 3, 2, 9]);
/// ```
pub fn cycle_crossover<G>(parent1: &[G], parent2: &[G], start: usize) -> Vec<G>
where
    G: Copy + Eq + Hash,
{
    assert!(parent1.len() == parent2.len());

    let n = parent1.len();
    if n == 0 {
        return Vec::new();
    }
    assert!(start < n, "the start should be a position of the genomes");

    let positions2 = GenePositions::new(parent2);
    let mut offspring = parent2.to_vec();

    // Between permutations of the same genes, the cycle returns to its start within as many
    // steps as the genomes have genes; otherwise it can circle elsewhere forever.
    let mut index = start;
    for _ in 0..n {
        offspring[index] = parent1[index];
        index = positions2
            .get(&parent1[index])
            .expect("the parents should be permutations of the same genes");
        if index == start {
            return offspring;
        }
    }
    panic!("the parents should be permutations of the same genes");
}

/// Performs a Single-Point Crossover (SPX) operation between two parent genomes.
///
/// A point of zero copies `parent2`; a point at or beyond the end copies `parent1`. The
/// offspring has the parents' length, but may repeat or omit genes, so this does not preserve
/// a permutation.
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

/// Pairs each gene of `genome` with the number of its copies before it.
fn numbered<G: Copy + Eq + Hash>(genome: &[G]) -> Vec<(G, usize)> {
    let mut copies = GeneMap::with_capacity_and_hasher(genome.len(), Default::default());
    genome
        .iter()
        .map(|&gene| {
            let copy = copies.entry(gene).or_insert(0);
            *copy += 1;
            (gene, *copy - 1)
        })
        .collect()
}

/// Maps each gene of `genome` to the position of its first occurrence.
fn positions<G: Copy + Eq + Hash>(genome: &[G]) -> GeneMap<G, usize> {
    let mut positions = GeneMap::with_capacity_and_hasher(genome.len(), Default::default());
    for (position, &gene) in genome.iter().enumerate() {
        positions.entry(gene).or_insert(position);
    }
    positions
}

/// Finds the positions of genes in a genome.
///
/// A short genome is scanned linearly, which is faster than hashing its genes; a longer one is
/// indexed by a hash map, which keeps each lookup constant-time.
enum GenePositions<'a, G> {
    Scan(&'a [G]),
    Map(GeneMap<G, usize>),
}

impl<'a, G: Copy + Eq + Hash> GenePositions<'a, G> {
    /// The longest genome that is scanned linearly. Measured on 64-bit integer genes, a linear
    /// scan was faster than hashing up to about 25 to 50 genes.
    const SCAN_LIMIT: usize = 32;

    fn new(genome: &'a [G]) -> Self {
        if genome.len() <= Self::SCAN_LIMIT {
            Self::Scan(genome)
        } else {
            Self::Map(positions(genome))
        }
    }

    /// Returns the position of the first occurrence of `gene`, if any.
    #[inline]
    fn get(&self, gene: &G) -> Option<usize> {
        match self {
            Self::Scan(genome) => genome.iter().position(|g| g == gene),
            Self::Map(positions) => positions.get(gene).copied(),
        }
    }

    /// Returns whether the genome contains `gene`.
    #[inline]
    fn contains(&self, gene: &G) -> bool {
        match self {
            Self::Scan(genome) => genome.contains(gene),
            Self::Map(positions) => positions.contains_key(gene),
        }
    }
}

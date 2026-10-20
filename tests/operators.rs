#[path = "operators/crossover.rs"]
mod crossover;
#[path = "operators/selection.rs"]
mod selection;
mod support;

use rusty_genes::*;

#[test]
fn bounded_ranges_allow_the_maximum_length() {
    let mut rng = Rng::from_seed(0);
    assert_eq!(
        bounded_random_range(usize::MAX, usize::MAX, usize::MAX, &mut rng),
        0..usize::MAX
    );
    for _ in 0..100 {
        let range = bounded_random_range(usize::MAX, 1, usize::MAX, &mut rng);
        assert!(!range.is_empty());
        assert!(range.start <= range.end);
    }
}

#[test]
fn bounded_ranges_allow_zero_length_at_the_maximum_bound() {
    let mut rng = Rng::from_seed(0);
    for _ in 0..100 {
        let range = bounded_random_range(usize::MAX, 0, 0, &mut rng);
        assert_eq!(range.start, range.end);
    }
    assert_eq!(bounded_random_range(0, 0, 0, &mut rng), 0..0);
}

#[test]
fn common_neighbors_handles_small_genomes_symmetrically() {
    assert!(common_neighbors::<u32>(&[], &[]).is_empty());
    assert!(common_neighbors(&[0], &[0]).is_empty());
    assert!(!common_neighbors(&[0, 1], &[0, 1]).is_empty());
    let forward = common_neighbors(&[0, 1], &[0, 1, 2]);
    let backward = common_neighbors(&[0, 1, 2], &[0, 1]);
    assert_eq!(forward, backward);
    assert_eq!(common_partitions(&[0, 1], &[0, 1]), vec![vec![0, 1]]);
}

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

use rusty_genes::*;

const PARENT1: [u32; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
const PARENT2: [u32; 10] = [8, 9, 0, 7, 4, 5, 6, 3, 2, 1];

const STRATEGIES: [CrossoverStrategy; 7] = [
    CrossoverStrategy::CommonPartition,
    CrossoverStrategy::Successor,
    CrossoverStrategy::EdgeRecombination,
    CrossoverStrategy::PartiallyMapped,
    CrossoverStrategy::Order,
    CrossoverStrategy::Cycle,
    CrossoverStrategy::SinglePoint,
];

#[test]
fn order_crossover_with_empty_range_copies_second_parent() {
    for start in 0..=PARENT1.len() {
        assert_eq!(
            order_crossover(&PARENT1, &PARENT2, start..start),
            PARENT2.to_vec(),
            "range {start}..{start}"
        );
    }
}

#[test]
fn permutation_crossovers_produce_permutations_of_the_parents() {
    let strategies = [
        CrossoverStrategy::CommonPartition,
        CrossoverStrategy::Successor,
        CrossoverStrategy::EdgeRecombination,
        CrossoverStrategy::PartiallyMapped,
        CrossoverStrategy::Order,
        CrossoverStrategy::Cycle,
    ];
    let mut rng = Rng::from_seed(0);

    for strategy in strategies {
        for _ in 0..100 {
            let mut offspring = strategy.crossover(&PARENT1, &PARENT2, &mut rng);
            offspring.sort_unstable();
            assert_eq!(offspring, PARENT1.to_vec(), "{strategy:?}");
        }
    }
}

#[test]
fn crossovers_with_repetition_keep_the_number_of_copies_of_each_gene() {
    // Three jobs of three operations each, in the order the parents schedule them.
    let parent1 = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    let parent2 = [2, 1, 0, 1, 2, 0, 0, 2, 1];
    let mut rng = Rng::from_seed(0);

    // Single-point crossover keeps positions rather than genes, so it is not among them.
    for strategy in STRATEGIES {
        if strategy == CrossoverStrategy::SinglePoint {
            continue;
        }
        for _ in 0..100 {
            let mut offspring = strategy.crossover_with_repetition(&parent1, &parent2, &mut rng);
            offspring.sort_unstable();
            assert_eq!(offspring, parent1.to_vec(), "{strategy:?}");
        }
    }
}

#[test]
fn crossovers_with_repetition_of_distinct_genes_are_the_crossovers() {
    for strategy in STRATEGIES {
        let (mut rng, mut repeated_rng) = (Rng::from_seed(0), Rng::from_seed(0));
        for _ in 0..100 {
            assert_eq!(
                strategy.crossover_with_repetition(&PARENT1, &PARENT2, &mut repeated_rng),
                strategy.crossover(&PARENT1, &PARENT2, &mut rng),
                "{strategy:?}"
            );
        }
    }
}

#[test]
fn crossover_of_empty_genomes_is_empty() {
    let empty: [u32; 0] = [];
    for strategy in [
        CrossoverStrategy::CommonPartition,
        CrossoverStrategy::Successor,
        CrossoverStrategy::EdgeRecombination,
        CrossoverStrategy::PartiallyMapped,
        CrossoverStrategy::Order,
        CrossoverStrategy::Cycle,
        CrossoverStrategy::SinglePoint,
    ] {
        assert!(
            strategy
                .crossover(&empty, &empty, &mut Rng::from_seed(1))
                .is_empty()
        );
    }
}

#[test]
fn edge_recombination_breaks_ties_at_random() {
    // In a tour shared by both parents, the two neighbors of the first gene have one link left
    // each, so either of them can follow it.
    let tour = [0, 1, 2, 3, 4, 5];
    let mut rng = Rng::from_seed(0);
    let mut followers = [0; 6];
    for _ in 0..1000 {
        let offspring = CrossoverStrategy::EdgeRecombination.crossover(&tour, &tour, &mut rng);
        let start = tour.iter().position(|&gene| gene == offspring[0]).unwrap();
        let next = tour.iter().position(|&gene| gene == offspring[1]).unwrap();
        followers[(next + 6 - start) % 6] += 1;
    }
    // Only the next and the previous gene of the tour can follow, about equally often.
    assert_eq!(followers[0] + followers[2] + followers[3] + followers[4], 0);
    assert!(
        (400..=600).contains(&followers[1]),
        "next gene followed {} of 1000 times",
        followers[1]
    );
}

#[test]
fn cycle_crossover_takes_one_cycle_from_the_first_parent() {
    // The cycles of the parents are the positions {0, 2, 8}, {1, 9}, {3, 7}, {4}, {5}, and {6}.
    // The parents agree on the single-position cycles, so taking one of them copies parent2.
    let cycles: [&[usize]; 4] = [&[0, 2, 8], &[1, 9], &[3, 7], &[]];
    let mut rng = Rng::from_seed(0);
    let mut taken = [0_usize; 4];
    for _ in 0..1000 {
        let offspring = CrossoverStrategy::Cycle.crossover(&PARENT1, &PARENT2, &mut rng);
        let from_parent1 = (0..PARENT1.len())
            .filter(|&i| offspring[i] != PARENT2[i])
            .collect::<Vec<_>>();
        assert!(from_parent1.iter().all(|&i| offspring[i] == PARENT1[i]));
        let cycle = cycles
            .iter()
            .position(|&cycle| cycle == from_parent1)
            .unwrap_or_else(|| panic!("{offspring:?} mixes cycles"));
        taken[cycle] += 1;
    }
    // The cycle is the one through a random position, so its chance is its share of positions.
    for (expected, count) in [300, 200, 200, 300].into_iter().zip(taken) {
        assert!(count.abs_diff(expected) <= 50, "cycles taken {taken:?}");
    }
}

#[test]
fn single_point_crossover_takes_genes_from_both_parents() {
    let parent1 = ['a', 'b', 'c'];
    let parent2 = ['x', 'y', 'z'];
    let mut rng = Rng::from_seed(0);
    let mut points = [0_usize; 4];
    for _ in 0..1000 {
        let offspring = CrossoverStrategy::SinglePoint.crossover(&parent1, &parent2, &mut rng);
        let point = offspring
            .iter()
            .take_while(|gene| parent1.contains(gene))
            .count();
        assert_eq!(offspring[..point], parent1[..point]);
        assert_eq!(offspring[point..], parent2[point..]);
        points[point] += 1;
    }
    assert_eq!(points[0] + points[3], 0, "points {points:?}");
    assert!(points[1].abs_diff(500) <= 50, "points {points:?}");
}

#[test]
fn single_point_crossover_of_one_gene_copies_the_first_parent() {
    let offspring = CrossoverStrategy::SinglePoint.crossover(&[1], &[2], &mut Rng::from_seed(0));
    assert_eq!(offspring, vec![1]);
}

#[test]
#[should_panic(expected = "permutations of the same genes")]
fn cycle_crossover_rejects_parents_that_are_not_permutations_of_each_other() {
    // From position 0, the cycle reaches position 2, whose gene leads back to position 2.
    cycle_crossover(&[1, 2, 2], &[0, 1, 2], 0);
}

#[test]
#[should_panic(expected = "permutations of the same genes")]
fn partially_mapped_crossover_rejects_parents_that_are_not_permutations_of_each_other() {
    // The mapping of the copied range leads `true` back to itself, so it never ends.
    partially_mapped_crossover(&[true, false], &[true, true], 0..1);
}

#[test]
#[should_panic(expected = "permutations of the same genes")]
fn order_crossover_rejects_parents_that_are_not_permutations_of_each_other() {
    // Every gene of the second parent is in the copied range, so no gene can fill the others.
    order_crossover(&[0, 1, 2, 3], &[0, 1, 1, 0], 0..2);
}

#[test]
#[should_panic(expected = "permutations of the same genes")]
fn common_partition_crossover_rejects_a_first_parent_that_repeats_a_gene() {
    // The repeated gene is in one partition, so the partitions are shorter than the genomes.
    common_partition_crossover(&[0, 0, 1, 2], &[0, 1, 2, 0], |_, _| 0);
}

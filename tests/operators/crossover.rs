use rusty_genes::*;

const PARENT1: [u32; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
const PARENT2: [u32; 10] = [8, 9, 0, 7, 4, 5, 6, 3, 2, 1];

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
        CrossoverStrategy::GeneralizedPartition2,
        CrossoverStrategy::SequentialConstructive,
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
fn crossover_of_empty_genomes_is_empty() {
    let empty: [u32; 0] = [];
    for strategy in [
        CrossoverStrategy::GeneralizedPartition2,
        CrossoverStrategy::SequentialConstructive,
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

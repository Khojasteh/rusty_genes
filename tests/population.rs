mod support;
use rusty_genes::*;
use support::*;

#[test]
fn deduplication_keeps_the_best_evaluation_of_the_same_genome() {
    let old = WeightedBits::new(vec![1.0], Objective::Maximize).individual(&[true]);
    let new = WeightedBits::new(vec![2.0], Objective::Maximize).individual(&[true]);
    for (objective, expected) in [(Objective::Maximize, 2.0), (Objective::Minimize, 1.0)] {
        let mut population =
            Population::with_objective(vec![old.clone(), new.clone()], objective).unwrap();
        assert_eq!(population.len(), 1);
        assert_eq!(population.fittest().score(), expected);
        population
            .replace(&mut vec![new.clone(), old.clone()])
            .unwrap();
        assert_eq!(population.fittest().score(), expected);
    }
}

#[test]
fn normalization_preserves_the_order_of_distinct_equal_score_genomes() {
    let model = WeightedBits::new(vec![1.0; 3], Objective::Maximize);
    let first = model.individual(&[true, false, false]);
    let second = model.individual(&[false, true, false]);
    let third = model.individual(&[false, false, true]);
    let mut population =
        Population::new(vec![first.clone(), second.clone(), first, third.clone()]).unwrap();
    assert_eq!(population.len(), 3);
    assert_eq!(population[0], second);
    assert_eq!(population[2], third);
    let mut invalid = vec![BitGenome::from(vec![])];
    assert!(matches!(
        population.replace(&mut invalid),
        Err(EvolutionError::EmptyPopulation)
    ));
    assert_eq!(population.len(), 3);
    assert!(invalid.is_empty());
}

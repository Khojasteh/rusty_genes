use rusty_genes::one_max::*;
use rusty_genes::*;

#[test]
fn one_max_identity_does_not_include_its_cached_evaluation() {
    let unevaluated = OneMax::from(vec![true, false]);
    assert!(!unevaluated.is_valid());
    let mut evaluated = unevaluated.clone();
    OneMaxEvolutionModel.evaluate(&mut evaluated);
    assert!(evaluated.is_valid());
    assert_eq!(unevaluated, evaluated);
}

#[test]
fn mutation_can_reach_the_one_max_goal_from_an_all_zero_genome() {
    let mut individual = OneMax::from(vec![false]);
    OneMaxEvolutionModel.evaluate(&mut individual);
    assert!(individual.is_valid());
    OneMaxEvolutionModel.mutate(&mut individual, &mut Rng::from_seed(0));
    assert!(!individual.is_valid());
    OneMaxEvolutionModel.evaluate(&mut individual);
    assert_eq!(individual.genome(), &[true]);
    assert_eq!(individual.score(), 1.0);
}

#[test]
fn one_max_scores_are_counts_of_set_bits() {
    for (genome, expected) in [
        (vec![false], 0.0),
        (vec![true], 1.0),
        (vec![true, false, true, false], 2.0),
        (vec![true, true, false, false, false, false], 2.0),
        (vec![true; 8], 8.0),
    ] {
        let mut individual = OneMax::from(genome);
        OneMaxEvolutionModel.evaluate(&mut individual);
        assert_eq!(individual.score(), expected);
        assert!(individual.is_valid());
    }
}

#[test]
fn empty_one_max_genomes_remain_invalid() {
    let mut individual = OneMax::from(vec![]);
    OneMaxEvolutionModel.evaluate(&mut individual);
    assert!(!individual.is_valid());
    assert!(matches!(
        Population::new(vec![individual]),
        Err(EvolutionError::EmptyPopulation)
    ));
}

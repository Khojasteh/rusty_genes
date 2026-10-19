use rusty_genes::one_max::*;
use rusty_genes::*;
use std::hash::{BuildHasher, RandomState};

#[test]
fn one_max_identity_does_not_include_its_cached_evaluation() {
    let unevaluated = OneMax::from(vec![true, false]);
    assert!(!unevaluated.is_valid());
    let mut evaluated = unevaluated.clone();
    OneMaxEvolutionModel.evaluate(&mut evaluated);
    assert!(evaluated.is_valid());
    assert_eq!(unevaluated, evaluated);
    let hasher = RandomState::new();
    assert_eq!(hasher.hash_one(&unevaluated), hasher.hash_one(&evaluated));
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

#[test]
fn one_max_stops_at_the_native_count_target() {
    let mut evolution = GeneticAlgorithm::new(OneMaxEvolutionModel, Default::default())
        .unwrap()
        .into_evolution()
        .with_seed(42);
    let outcome = evolution
        .evolve_population(vec![OneMax::from(vec![true; 8])], |state| {
            Limits::new().generations(5).check(state)
        })
        .unwrap();
    assert_eq!(outcome.reason(), &StopReason::GoalReached);
    assert_eq!(outcome.state().generation(), 0);
    assert_eq!(outcome.state().population().fittest().score(), 8.0);
}

#[test]
fn one_max_recomputes_its_goal_for_each_genome_length() {
    let mut evolution = GeneticAlgorithm::new(OneMaxEvolutionModel, Default::default())
        .unwrap()
        .into_evolution()
        .with_seed(42);
    let limits = Limits::new().generations(2);
    let first = evolution
        .evolve_population(vec![OneMax::from(vec![true])], |s| limits.check(s))
        .unwrap();
    assert_eq!(first.state().generation(), 0);
    let second = evolution
        .evolve_population(vec![OneMax::from(vec![true, false, true, false])], |s| {
            limits.check(s)
        })
        .unwrap();
    assert!(second.state().generation() > 0);
    assert_eq!(
        OneMaxEvolutionModel.goal_score(second.state().population()),
        Some(4.0)
    );
}

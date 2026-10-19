use crate::support::*;
use rusty_genes::*;

#[test]
fn a_flat_score_landscape_keeps_exploration_active() {
    let mut configuration = AdaptiveGeneticAlgorithmParameters::default();
    configuration.population_size = 2;
    configuration.elitism_rate = 0.0;
    let mut evolution = AdaptiveGeneticAlgorithm::new(
        WeightedBits::new(vec![0.0, 0.0], Objective::Maximize),
        configuration,
    )
    .unwrap()
    .into_evolution()
    .with_seed(0);
    let mut changed = false;
    evolution
        .evolve_population(vec![BitGenome::from(vec![false, false])], |s| {
            changed |= s
                .population()
                .iter()
                .any(|individual| individual.genes.iter().any(|&bit| bit));
            Limits::new().generations(5).check(s)
        })
        .unwrap();
    assert!(changed);
    assert_eq!(evolution.algorithm().reproduction().crossover_rate(), 0.3);
    assert!((evolution.algorithm().reproduction().mutation_rate() - 1.0).abs() <= f64::EPSILON);
}

#[test]
fn adaptive_spread_is_scale_independent_and_resets_for_a_new_run() {
    for scale in [1.0, 1e154, f64::MAX] {
        let model = WeightedBits::new(
            vec![-scale, scale, -scale / 2.0, scale / 2.0],
            Objective::Maximize,
        );
        let initial = vec![
            model.individual(&[true, false, false, false]),
            model.individual(&[false, true, false, false]),
        ];
        let reduced = vec![
            model.individual(&[false, false, true, false]),
            model.individual(&[false, false, false, true]),
        ];
        let mut algorithm = AdaptiveGeneticAlgorithm::new(model, Default::default()).unwrap();
        let replace = |state: &mut RunState<BitGenome>| {
            state
                .advance(|population| population.replace(&mut reduced.clone()))
                .unwrap();
        };
        let mut state = RunState::new(Population::new(initial).unwrap());
        algorithm.start_run();
        algorithm.prepare_generation(&state);
        replace(&mut state);
        algorithm.prepare_generation(&state);
        assert!((algorithm.reproduction().crossover_rate() - 0.65).abs() <= 4.0 * f64::EPSILON);
        assert!((algorithm.reproduction().mutation_rate() - 0.55).abs() <= 4.0 * f64::EPSILON);
        let mut new_run = RunState::new(Population::new(reduced.clone()).unwrap());
        algorithm.start_run();
        algorithm.prepare_generation(&new_run);
        replace(&mut new_run);
        algorithm.prepare_generation(&new_run);
        assert_eq!(algorithm.reproduction().crossover_rate(), 1.0);
        assert!((algorithm.reproduction().mutation_rate() - 0.1).abs() <= f64::EPSILON);
    }
}

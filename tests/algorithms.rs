#[path = "algorithms/aga.rs"]
mod aga;
#[path = "algorithms/dgga.rs"]
mod dgga;
#[path = "algorithms/ga.rs"]
mod ga;
mod support;

use rusty_genes::*;
use support::*;

fn parameters(size: usize, elitism: f64) -> GeneticAlgorithmParameters {
    let mut parameters = GeneticAlgorithmParameters::default();
    parameters.population_size = size;
    parameters.elitism_rate = elitism;
    parameters.mutation_rate = 1.0;
    parameters
}

#[test]
fn algorithms_and_runs_are_send() {
    fn assert_send<T: Send>() {}
    assert_send::<Evolution<GeneticAlgorithm<WeightedBits>>>();
    assert_send::<Evolution<AdaptiveGeneticAlgorithm<WeightedBits>>>();
    assert_send::<Evolution<DiversityGuidedGeneticAlgorithm<WeightedBits>>>();
}

#[test]
fn a_single_candidate_can_evolve_in_every_algorithm() {
    let model = || WeightedBits::new(vec![1.0], Objective::Maximize);
    let mut ga = GeneticAlgorithm::new(model(), parameters(1, 0.0))
        .unwrap()
        .into_evolution();
    let mut adaptive = AdaptiveGeneticAlgorithmParameters::default();
    adaptive.population_size = 1;
    adaptive.elitism_rate = 0.0;
    adaptive.minimum_mutation_rate = 1.0;
    let mut aga = AdaptiveGeneticAlgorithm::new(model(), adaptive)
        .unwrap()
        .into_evolution();
    // A single individual has no diversity, so the diversity-guided algorithm explores by
    // mutation.
    let mut guided = DiversityGuidedGeneticAlgorithmParameters::default();
    guided.population_size = 1;
    guided.elitism_rate = 0.0;
    let mut dgga = DiversityGuidedGeneticAlgorithm::new(model(), guided)
        .unwrap()
        .into_evolution();
    let limits = Limits::new().generations(3);
    let ga_result = ga
        .evolve_population(vec![BitGenome::from(vec![false])], |s| limits.check(s))
        .unwrap();
    let aga_result = aga
        .evolve_population(vec![BitGenome::from(vec![false])], |s| limits.check(s))
        .unwrap();
    let dgga_result = dgga
        .evolve_population(vec![BitGenome::from(vec![false])], |s| limits.check(s))
        .unwrap();
    for outcome in [ga_result, aga_result, dgga_result] {
        let state = outcome.state();
        assert_eq!(state.generation(), 3);
        assert_eq!(state.population().len(), 1);
        assert_eq!(state.fittest_so_far().genes, vec![true]);
        assert_eq!(state.fittest_so_far().score(), 1.0);
    }
}

#[test]
fn invalid_parameters_and_invalid_initial_genomes_are_rejected() {
    fn invalid_ga(expected: &str, invalidate: impl FnOnce(&mut GeneticAlgorithmParameters)) {
        let mut configuration = GeneticAlgorithmParameters::default();
        invalidate(&mut configuration);
        assert!(matches!(
            GeneticAlgorithm::new(
                WeightedBits::new(vec![1.0], Objective::Maximize),
                configuration
            ),
            Err(EvolutionError::InvalidParameter { name, .. }) if name == expected
        ));
    }
    fn invalid_aga(
        expected: &str,
        invalidate: impl FnOnce(&mut AdaptiveGeneticAlgorithmParameters),
    ) {
        let mut configuration = AdaptiveGeneticAlgorithmParameters::default();
        invalidate(&mut configuration);
        assert!(matches!(
            AdaptiveGeneticAlgorithm::new(
                WeightedBits::new(vec![1.0], Objective::Maximize),
                configuration
            ),
            Err(EvolutionError::InvalidParameter { name, .. }) if name == expected
        ));
    }
    fn invalid_dgga(
        expected: &str,
        invalidate: impl FnOnce(&mut DiversityGuidedGeneticAlgorithmParameters),
    ) {
        let mut configuration = DiversityGuidedGeneticAlgorithmParameters::default();
        invalidate(&mut configuration);
        assert!(matches!(
            DiversityGuidedGeneticAlgorithm::new(
                WeightedBits::new(vec![1.0], Objective::Maximize),
                configuration
            ),
            Err(EvolutionError::InvalidParameter { name, .. }) if name == expected
        ));
    }
    invalid_ga("population_size", |p| p.population_size = 0);
    invalid_ga("elitism_rate", |p| p.elitism_rate = 1.0);
    invalid_ga("crossover_rate", |p| p.crossover_rate = 1.5);
    invalid_ga("mutation_rate", |p| p.mutation_rate = f64::NAN);
    invalid_ga("tournament size", |p| {
        p.selection_strategy = SelectionStrategy::Tournament(0)
    });
    invalid_aga("population_size", |p| p.population_size = 0);
    invalid_aga("elitism_rate", |p| p.elitism_rate = -0.1);
    invalid_aga("crossover_rate", |p| p.crossover_rate = 1.5);
    invalid_aga("subaverage_crossover_rate", |p| {
        p.subaverage_crossover_rate = -0.5
    });
    invalid_aga("mutation_rate", |p| p.mutation_rate = f64::NAN);
    invalid_aga("subaverage_mutation_rate", |p| {
        p.subaverage_mutation_rate = 2.0
    });
    invalid_aga("minimum_mutation_rate", |p| {
        p.minimum_mutation_rate = -0.005
    });
    invalid_aga("tournament size", |p| {
        p.selection_strategy = SelectionStrategy::Tournament(0)
    });
    invalid_dgga("population_size", |p| p.population_size = 0);
    invalid_dgga("elitism_rate", |p| p.elitism_rate = 1.0);
    invalid_dgga("low_diversity", |p| p.low_diversity = -0.1);
    invalid_dgga("high_diversity", |p| p.high_diversity = 1.5);
    invalid_dgga("low_diversity", |p| {
        p.low_diversity = 0.5;
        p.high_diversity = 0.25;
    });
    invalid_dgga("tournament size", |p| {
        p.selection_strategy = SelectionStrategy::Tournament(0)
    });
    let mut evolution = GeneticAlgorithm::new(
        WeightedBits::new(vec![1.0], Objective::Maximize),
        Default::default(),
    )
    .unwrap()
    .into_evolution();
    assert!(matches!(
        evolution.evolve_population(vec![BitGenome::from(vec![])], |_| {
            std::ops::ControlFlow::Continue(())
        }),
        Err(EvolutionError::EmptyPopulation)
    ));
}

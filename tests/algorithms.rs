mod support;
use rusty_genes::*;
use std::{
    ops::ControlFlow,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};
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
}

#[test]
fn a_single_candidate_can_evolve_in_both_algorithms() {
    let model = || WeightedBits::new(vec![1.0], Objective::Maximize);
    let mut ga = GeneticAlgorithm::new(model(), parameters(1, 0.0))
        .unwrap()
        .into_evolution();
    let mut adaptive = AdaptiveGeneticAlgorithmParameters::default();
    adaptive.population_size = 1;
    adaptive.elitism_rate = 0.0;
    adaptive.mutation_rate_range = 1.0..=1.0;
    let mut aga = AdaptiveGeneticAlgorithm::new(model(), adaptive)
        .unwrap()
        .into_evolution();
    let limits = Limits::new().generations(3);
    let ga_result = ga
        .evolve_population(vec![BitGenome::from(vec![false])], |s| limits.check(s))
        .unwrap();
    let aga_result = aga
        .evolve_population(vec![BitGenome::from(vec![false])], |s| limits.check(s))
        .unwrap();
    for outcome in [ga_result, aga_result] {
        let state = outcome.state();
        assert_eq!(state.generation(), 3);
        assert_eq!(state.population().len(), 1);
        assert_eq!(state.fittest_so_far().genes, vec![true]);
        assert_eq!(state.fittest_so_far().score(), 1.0);
    }
}

#[test]
fn breeding_fills_the_slot_after_elites() {
    for (objective, elites) in [
        (Objective::Maximize, vec![1.0, 2.0, 3.0]),
        (Objective::Minimize, vec![2.0, 1.0, 0.0]),
    ] {
        let model = WeightedBits::new(vec![1.0, 2.0], objective);
        let mut algorithm = GeneticAlgorithm::new(model, parameters(4, 0.75)).unwrap();
        let mut individuals = vec![
            vec![false, false],
            vec![true, false],
            vec![false, true],
            vec![true, true],
        ]
        .into_iter()
        .map(BitGenome::from)
        .collect::<Vec<_>>();
        algorithm.evaluate_initial(&mut individuals);
        let state = RunState::new(Population::with_objective(individuals, objective).unwrap());
        algorithm.prepare_generation(&state);
        let mut offspring = Vec::new();
        algorithm.select_elites(state.population(), &mut offspring);
        assert_eq!(
            offspring.iter().map(Individual::score).collect::<Vec<_>>(),
            elites
        );
        algorithm.generate_offspring(&state, &mut offspring, &mut Rng::from_seed(0));
        assert_eq!(offspring.len(), 4);
    }
}

#[test]
fn every_run_entry_preserves_the_best_genome_across_worse_generations() {
    for (objective, initial) in [(Objective::Maximize, true), (Objective::Minimize, false)] {
        let mut algorithm =
            GeneticAlgorithm::new(WeightedBits::new(vec![1.0], objective), parameters(1, 0.0))
                .unwrap();
        algorithm.reproduction_mut().set_survival(Generational);
        let mut evolution = algorithm.into_evolution().with_seed(0);
        let limits = Limits::new().generations(3);
        // Flipping the single bit with generational survival alternately loses and regains
        // the best solution. After three generations the current solution is worse.
        let mut last_genome = Vec::new();
        let best = evolution
            .run(&vec![initial], |s| {
                last_genome = s.population().fittest().genes.clone();
                limits.check(s)
            })
            .unwrap()
            .into_fittest_so_far();
        assert_eq!(best.genes, vec![initial]);
        assert_eq!(last_genome, vec![!initial]);
        let best = evolution
            .evolve_individual(BitGenome::from(vec![initial]), |s| limits.check(s))
            .unwrap()
            .into_fittest_so_far();
        assert_eq!(best.genes, vec![initial]);
        let outcome = evolution
            .evolve_population(vec![BitGenome::from(vec![initial])], |s| limits.check(s))
            .unwrap();
        let state = outcome.state();
        assert_eq!(state.population().fittest().genes, vec![!initial]);
        assert_eq!(state.fittest_so_far().genes, vec![initial]);
        assert_eq!(state.stagnation(), 3);
    }
}

#[test]
fn a_run_stops_with_the_reason_its_callback_gives() {
    for reason in [StopReason::Interrupted, StopReason::Custom("enough".into())] {
        let outcome = GeneticAlgorithm::new(
            WeightedBits::new(vec![6.0], Objective::Maximize),
            parameters(1, 0.0),
        )
        .unwrap()
        .into_evolution()
        .with_seed(1)
        .evolve_population(vec![BitGenome::from(vec![false])], |s| {
            if s.generation() < 2 {
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(reason.clone())
            }
        })
        .unwrap();
        assert_eq!(outcome.reason(), &reason);
        assert_eq!(outcome.state().generation(), 2);
    }
}

/// A model that mutates more bits in each generation of a run, and records how many bits it
/// mutates at a time.
struct Strengthening {
    bits: WeightedBits,
    strength: usize,
    strengths: Mutex<Vec<usize>>,
}

impl Strengthening {
    fn new() -> Self {
        Self {
            bits: WeightedBits::new(vec![1.0; 8], Objective::Maximize),
            strength: 1,
            strengths: Mutex::new(Vec::new()),
        }
    }
}

impl EvolutionModel for Strengthening {
    type Individual = BitGenome;
    fn start_run(&mut self) {
        self.strength = 1;
    }
    fn prepare(&mut self, _: &RunState<BitGenome>) {
        self.strength += 1;
    }
    fn crossover(&self, first: &BitGenome, second: &BitGenome, rng: &mut Rng) -> BitGenome {
        self.bits.crossover(first, second, rng)
    }
    fn mutate(&self, individual: &mut BitGenome, rng: &mut Rng) {
        self.strengths.lock().unwrap().push(self.strength);
        for _ in 0..self.strength {
            self.bits.mutate(individual, rng);
        }
    }
    fn evaluate(&self, individual: &mut BitGenome) {
        self.bits.evaluate(individual);
    }
}

/// Asserts that two seeded runs of `algorithm` from the same individual mutate its starting
/// copies at the strength of a new run, and end with the same population.
fn assert_runs_start_afresh<A: EvolutionAlgorithm<Model = Strengthening>>(algorithm: A) {
    let mut evolution = algorithm.into_evolution().with_seed(7);
    let copies = evolution.algorithm().population_size() - 1;
    let mut populations = Vec::new();
    for _ in 0..2 {
        let outcome = evolution
            .evolve_individual(BitGenome::from(vec![false; 8]), |s| {
                Limits::new().generations(3).check(s)
            })
            .unwrap();
        let strengths =
            std::mem::take(&mut *evolution.algorithm().model().strengths.lock().unwrap());
        // The starting copies are mutated before the first generation is prepared.
        assert_eq!(strengths[..copies], vec![1; copies]);
        populations.push(outcome.into_state().into_population().to_vec());
    }
    assert_eq!(populations[0], populations[1]);
}

#[test]
fn every_run_resets_the_algorithm_and_model_before_creating_its_starting_individuals() {
    assert_runs_start_afresh(
        GeneticAlgorithm::new(Strengthening::new(), Default::default()).unwrap(),
    );
    assert_runs_start_afresh(
        AdaptiveGeneticAlgorithm::new(Strengthening::new(), Default::default()).unwrap(),
    );
}

#[test]
fn a_custom_selector_is_retained_between_runs() {
    struct CountingSelector(Arc<AtomicUsize>);
    impl SelectionMechanism<BitGenome> for CountingSelector {
        fn prepare(&mut self, _: &Population<BitGenome>) {}
        fn select<'a>(&self, p: &'a Population<BitGenome>, _: &mut Rng) -> (usize, &'a BitGenome) {
            self.0.fetch_add(1, Ordering::Relaxed);
            (p.len() - 1, p.fittest())
        }
    }
    let selections = Arc::new(AtomicUsize::new(0));
    let mut configuration = parameters(4, 0.0);
    configuration.mutation_rate = 0.0;
    let mut algorithm = GeneticAlgorithm::new(
        WeightedBits::new(vec![1.0], Objective::Maximize),
        configuration,
    )
    .unwrap();
    algorithm
        .reproduction_mut()
        .set_selector(CountingSelector(selections.clone()));
    let mut evolution = algorithm.into_evolution();
    for run in 1..=2 {
        evolution
            .run(&vec![false], |s| Limits::new().generations(3).check(s))
            .unwrap();
        assert_eq!(selections.load(Ordering::Relaxed), run * 12);
    }
}

#[test]
fn seeded_runs_are_independent_of_thread_count() {
    let run = |threads, seed| {
        let mut configuration = AdaptiveGeneticAlgorithmParameters::default();
        configuration.population_size = 20;
        let model = WeightedBits::new((1..=16).map(|v| v as f64).collect(), Objective::Maximize);
        let mut evolution = AdaptiveGeneticAlgorithm::new(model, configuration)
            .unwrap()
            .into_evolution()
            .with_seed(seed);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                evolution
                    .evolve_population(
                        (0..16)
                            .map(|i| BitGenome::from((0..16).map(|j| i == j).collect::<Vec<_>>()))
                            .collect(),
                        |s| Limits::new().generations(4).check(s),
                    )
                    .unwrap()
                    .into_state()
                    .into_population()
                    .to_vec()
            })
    };
    assert_eq!(run(1, 5), run(4, 5));
    assert_ne!(run(4, 5), run(4, 6));
}

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
    invalid_ga("population_size", |p| p.population_size = 0);
    invalid_ga("elitism_rate", |p| p.elitism_rate = 1.0);
    invalid_ga("crossover_rate", |p| p.crossover_rate = 1.5);
    invalid_ga("mutation_rate", |p| p.mutation_rate = f64::NAN);
    invalid_ga("tournament size", |p| {
        p.selection_strategy = SelectionStrategy::Tournament(0)
    });
    invalid_aga("crossover_rate_range", |p| {
        p.crossover_rate_range = 0.9..=0.3
    });
    invalid_aga("mutation_rate_range", |p| p.mutation_rate_range = 0.1..=1.1);
    invalid_aga("parameter_adjustment_interval", |p| {
        p.parameter_adjustment_interval = 0
    });
    let mut evolution = GeneticAlgorithm::new(
        WeightedBits::new(vec![1.0], Objective::Maximize),
        Default::default(),
    )
    .unwrap()
    .into_evolution();
    assert!(matches!(
        evolution.evolve_population(vec![BitGenome::from(vec![])], |_| {
            ControlFlow::Continue(())
        }),
        Err(EvolutionError::EmptyPopulation)
    ));
}

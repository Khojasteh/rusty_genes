mod support;
use rusty_genes::*;
use std::{ops::ControlFlow, sync::Mutex};
use support::*;

fn evolution(objective: Objective, goal: Option<f64>) -> Evolution<GeneticAlgorithm<WeightedBits>> {
    let mut model = WeightedBits::new(vec![6.0], objective);
    model.goal = goal;
    let mut configuration = GeneticAlgorithmParameters::default();
    configuration.population_size = 1;
    configuration.elitism_rate = 0.0;
    configuration.mutation_rate = 1.0;
    GeneticAlgorithm::new(model, configuration)
        .unwrap()
        .into_evolution()
        .with_seed(1)
}

fn parameters(size: usize, elitism: f64) -> GeneticAlgorithmParameters {
    let mut parameters = GeneticAlgorithmParameters::default();
    parameters.population_size = size;
    parameters.elitism_rate = elitism;
    parameters.mutation_rate = 1.0;
    parameters
}

#[test]
fn goals_stop_at_equality_or_beyond_in_both_directions() {
    for (objective, initial, goal, expected) in [
        (Objective::Maximize, false, 6.0, 6.0),
        (Objective::Maximize, false, 5.0, 6.0),
        (Objective::Minimize, true, 0.0, 0.0),
        (Objective::Minimize, true, 1.0, 0.0),
    ] {
        let mut evolution = evolution(objective, Some(goal));
        let mut observed = Vec::new();
        let initial = evolution.algorithm().model().individual(&[initial]);
        let outcome = evolution
            .evolve_population(vec![initial], |s| {
                observed.push(s.population().fittest().score());
                Limits::new().generations(20).check(s)
            })
            .unwrap();
        assert_eq!(outcome.reason(), &StopReason::GoalReached);
        assert_eq!(outcome.state().fittest_so_far().score(), expected);
        assert_eq!(outcome.state().generation(), 1);
        // The callback does not observe the generation that reaches the goal.
        assert_eq!(observed.len(), 1);
        assert_ne!(observed.last(), Some(&expected));
    }
}

#[test]
fn an_initial_goal_is_reported_without_breeding() {
    for (objective, initial, goal) in [
        (Objective::Maximize, true, 6.0),
        (Objective::Maximize, true, 5.0),
        (Objective::Minimize, false, 0.0),
        (Objective::Minimize, false, 1.0),
    ] {
        let mut evolution = evolution(objective, Some(goal));
        let mut observations = 0;
        let outcome = evolution
            .evolve_population(vec![BitGenome::from(vec![initial])], |s| {
                observations += 1;
                Limits::new().generations(3).check(s)
            })
            .unwrap();
        assert_eq!(outcome.reason(), &StopReason::GoalReached);
        assert_eq!(outcome.state().generation(), 0);
        assert_eq!(outcome.state().fittest_so_far().genes, vec![initial]);
        assert_eq!(observations, 0);
    }
}

#[test]
fn callbacks_can_stop_runs_without_a_reachable_goal() {
    for goal in [None, Some(100.0)] {
        let outcome = evolution(Objective::Maximize, goal)
            .evolve_population(vec![BitGenome::from(vec![false])], |s| {
                Limits::new().generations(3).check(s)
            })
            .unwrap();
        assert_eq!(outcome.reason(), &StopReason::GenerationLimit);
        assert_eq!(outcome.state().generation(), 3);
        assert_eq!(outcome.state().fittest_so_far().score(), 6.0);
    }
}

#[test]
fn a_run_stops_with_the_reason_its_callback_gives() {
    for reason in [StopReason::Interrupted, StopReason::Custom("enough".into())] {
        let outcome = evolution(Objective::Maximize, None)
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

#[test]
fn invalid_goal_scores_are_rejected_before_progress() {
    for goal in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let result = evolution(Objective::Maximize, Some(goal))
            .evolve_population(vec![BitGenome::from(vec![false])], |_| {
                panic!("invalid goal reached progress")
            });
        assert!(matches!(
            result,
            Err(EvolutionError::InvalidParameter {
                name: "goal_score",
                ..
            })
        ));
    }
}

#[test]
fn each_run_takes_its_goal_from_its_own_initial_population() {
    /// A model that records the best initial score of every population it takes a goal from.
    struct RecordedGoals(WeightedBits, Mutex<Vec<f64>>);

    impl EvolutionModel for RecordedGoals {
        type Individual = BitGenome;
        fn goal_score(&self, initial_population: &Population<BitGenome>) -> Option<f64> {
            let score = initial_population.fittest().score();
            self.1.lock().unwrap().push(score);
            Some(score + 1.0)
        }
        fn crossover(&self, first: &BitGenome, second: &BitGenome, rng: &mut Rng) -> BitGenome {
            self.0.crossover(first, second, rng)
        }
        fn mutate(&self, individual: &mut BitGenome, rng: &mut Rng) {
            self.0.mutate(individual, rng);
        }
        fn evaluate(&self, individual: &mut BitGenome) {
            self.0.evaluate(individual);
        }
    }

    let model = RecordedGoals(
        WeightedBits::new(vec![1.0, 2.0], Objective::Maximize),
        Mutex::new(Vec::new()),
    );
    let mut evolution = GeneticAlgorithm::new(model, parameters(1, 0.0))
        .unwrap()
        .into_evolution()
        .with_seed(0);
    for genes in [vec![true, false], vec![false, false]] {
        evolution
            .evolve_population(vec![BitGenome::from(genes)], |s| {
                Limits::new().generations(3).check(s)
            })
            .unwrap();
    }
    assert_eq!(
        *evolution.algorithm().model().1.lock().unwrap(),
        vec![1.0, 0.0]
    );
}

#[test]
fn all_entry_points_and_both_algorithms_stop_at_the_goal() {
    fn assert_goal_reached(outcome: Outcome<BitGenome>, genes: Vec<bool>) {
        assert_eq!(outcome.reason(), &StopReason::GoalReached);
        assert_eq!(outcome.state().generation(), 1);
        assert_eq!(outcome.into_fittest_so_far().genes, genes);
    }

    let mut ga = evolution(Objective::Maximize, Some(6.0));
    let mut generations = Vec::new();
    let outcome = ga.run(&vec![false], |s| {
        generations.push(s.generation());
        Limits::new().generations(20).check(s)
    });
    assert_goal_reached(outcome.unwrap(), vec![true]);
    assert_eq!(generations, vec![0]);
    generations.clear();
    let outcome = ga.evolve_individual(BitGenome::from(vec![false]), |s| {
        generations.push(s.generation());
        Limits::new().generations(20).check(s)
    });
    assert_goal_reached(outcome.unwrap(), vec![true]);
    assert_eq!(generations, vec![0]);
    generations.clear();
    let mut model = WeightedBits::new(vec![6.0], Objective::Minimize);
    model.goal = Some(0.0);
    let mut configuration = AdaptiveGeneticAlgorithmParameters::default();
    configuration.population_size = 1;
    configuration.elitism_rate = 0.0;
    configuration.mutation_rate_range = 1.0..=1.0;
    let mut aga = AdaptiveGeneticAlgorithm::new(model, configuration)
        .unwrap()
        .into_evolution();
    let outcome = aga.run(&vec![true], |s| {
        generations.push(s.generation());
        Limits::new().generations(20).check(s)
    });
    assert_goal_reached(outcome.unwrap(), vec![false]);
    assert_eq!(generations, vec![0]);
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

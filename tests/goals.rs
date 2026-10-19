mod support;
use rusty_genes::*;
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

mod support;
use rusty_genes::*;
use support::*;

fn candidate(model: &WeightedBits, index: usize) -> BitGenome {
    let genes = (0..model.weights.len())
        .map(|i| i == index)
        .collect::<Vec<_>>();
    model.individual(&genes)
}

fn population(scores: &[f64], objective: Objective) -> Population<BitGenome> {
    let model = WeightedBits::new(scores.to_vec(), objective);
    Population::with_objective(
        (0..scores.len()).map(|i| candidate(&model, i)).collect(),
        objective,
    )
    .unwrap()
}

#[test]
fn minimization_preserves_native_scores() {
    let model = WeightedBits::new(vec![30.0, 10.0, 20.0, 15.0, -5.0], Objective::Minimize);
    let mut population = Population::with_objective(
        (0..3).map(|i| candidate(&model, i)).collect(),
        Objective::Minimize,
    )
    .unwrap();
    assert_eq!(
        population.iter().map(Individual::score).collect::<Vec<_>>(),
        vec![30.0, 20.0, 10.0]
    );
    assert_eq!(
        (
            population.stats().min(),
            population.stats().max(),
            population.stats().avg()
        ),
        (10.0, 30.0, 20.0)
    );
    assert_eq!(
        (population.stats().arg_min(), population.stats().arg_max()),
        (2, 0)
    );
    population
        .replace(&mut vec![candidate(&model, 3), candidate(&model, 4)])
        .unwrap();
    assert_eq!(population.fittest().score(), -5.0);
    assert!(matches!(
        population.replace(&mut vec![BitGenome::from(vec![])]),
        Err(EvolutionError::EmptyPopulation)
    ));
    assert_eq!(population.objective(), Objective::Minimize);
    assert_eq!(population.take_fittest(), candidate(&model, 4));
}

#[test]
fn objective_comparison_preserves_precision_and_treats_signed_zero_as_equal() {
    let lower: f64 = 10_000_000_000_000_000.0;
    let higher = f64::from_bits(lower.to_bits() + 1);
    assert!(Objective::Minimize.is_fitter(lower, higher));
    assert!(Objective::Maximize.is_fitter(higher, lower));
    for objective in [Objective::Minimize, Objective::Maximize] {
        assert!(objective.cmp_fitness(-0.0, 0.0).is_eq());
        assert!(objective.is_goal_reached(-0.0, 0.0));
        assert!(objective.is_goal_reached(0.0, -0.0));
    }
    let population = population(&[higher, lower], Objective::Minimize);
    assert_eq!(population.fittest().score().to_bits(), lower.to_bits());
    assert_eq!(Objective::default(), Objective::Maximize);
    let model = WeightedBits::new(vec![10.0, 20.0], Objective::Maximize);
    assert_eq!(
        Population::new(vec![candidate(&model, 0), candidate(&model, 1)])
            .unwrap()
            .take_fittest()
            .score(),
        20.0
    );
}

#[test]
fn objective_comparison_orders_nan_by_total_order() {
    for nan in [f64::NAN, -f64::NAN] {
        for score in [f64::NEG_INFINITY, 0.0, f64::INFINITY, nan] {
            assert_eq!(
                Objective::Maximize.cmp_fitness(nan, score),
                nan.total_cmp(&score)
            );
            assert_eq!(
                Objective::Minimize.cmp_fitness(nan, score),
                score.total_cmp(&nan)
            );
        }
    }
}

#[test]
fn every_selector_uses_minimization_order() {
    let population = population(&[0.0, 10.0, 20.0], Objective::Minimize);
    let boltzmann_sum = (-2.0_f64).exp() + (-1.0_f64).exp() + 1.0;
    let strategies = [
        (
            SelectionStrategy::Tournament(2),
            [1.0 / 9.0, 3.0 / 9.0, 5.0 / 9.0],
        ),
        (SelectionStrategy::Rank, [1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]),
        (SelectionStrategy::Linear(2.0), [0.0, 1.0 / 3.0, 2.0 / 3.0]),
        (SelectionStrategy::Elitist(0.5), [0.0, 0.5, 0.5]),
        (
            SelectionStrategy::RouletteWheel,
            [0.0, 1.0 / 3.0, 2.0 / 3.0],
        ),
        (
            SelectionStrategy::Boltzmann(10.0, 1.0),
            [
                (-2.0_f64).exp() / boltzmann_sum,
                (-1.0_f64).exp() / boltzmann_sum,
                1.0 / boltzmann_sum,
            ],
        ),
    ];
    for (strategy, expected) in strategies {
        let mut selector = strategy.create_selector::<BitGenome>().unwrap();
        selector.prepare(&population);
        let mut rng = Rng::from_seed(0);
        let mut counts = [0; 3];
        for _ in 0..100_000 {
            counts[selector.select(&population, &mut rng).0] += 1;
        }
        for (index, count) in counts.into_iter().enumerate() {
            assert!(
                (count as f64 / 100_000.0 - expected[index]).abs() < 0.01,
                "{strategy:?}: {counts:?}"
            );
        }
    }
}

#[test]
fn minimizing_roulette_handles_negative_and_extreme_native_scores() {
    for (scores, expected) in [
        ([-30.0, -20.0, -10.0], [1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]),
        ([-1e308, 0.0, 1e308], [0.0, 1.0 / 3.0, 2.0 / 3.0]),
    ] {
        let population = population(&scores, Objective::Minimize);
        let mut selector = RouletteWheelSelection::new();
        selector.prepare(&population);
        let mut rng = Rng::from_seed(0);
        let mut counts = [0; 3];
        for _ in 0..100_000 {
            counts[selector.select(&population, &mut rng).0] += 1;
        }
        for (count, expected) in counts.into_iter().zip(expected) {
            assert!((count as f64 / 100_000.0 - expected).abs() < 0.01);
        }
        assert_eq!(population.fittest().score(), scores[0]);
    }
}

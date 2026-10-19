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
fn objective_comparison_preserves_precision_and_treats_signed_zero_as_equal() {
    let lower: f64 = 10_000_000_000_000_000.0;
    let higher = f64::from_bits(lower.to_bits() + 1);
    assert!(Objective::Minimize.cmp_fitness(lower, higher).is_gt());
    assert!(Objective::Maximize.cmp_fitness(lower, higher).is_lt());
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
fn fitness_follows_the_objective_and_equal_scores_are_equally_fit() {
    for (objective, fitter, less_fit) in [
        (Objective::Maximize, 2.0, 1.0),
        (Objective::Minimize, 1.0, 2.0),
    ] {
        assert!(objective.is_fitter(fitter, less_fit));
        assert!(!objective.is_fitter(less_fit, fitter));
        assert!(objective.is_at_least_as_fit(fitter, less_fit));
        assert!(!objective.is_at_least_as_fit(less_fit, fitter));
        for (score, other) in [(fitter, fitter), (-0.0, 0.0)] {
            assert!(!objective.is_fitter(score, other));
            assert!(objective.is_at_least_as_fit(score, other));
        }
    }
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

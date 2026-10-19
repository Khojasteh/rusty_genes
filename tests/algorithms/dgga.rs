use crate::support::*;
use rusty_genes::*;

/// Returns a population of the given genomes of four bits.
fn population(model: &WeightedBits, genomes: &[[bool; 4]]) -> Vec<BitGenome> {
    genomes
        .iter()
        .map(|genes| model.individual(genes))
        .collect()
}

#[test]
fn modes_switch_only_when_the_diversity_crosses_a_threshold() {
    let model = WeightedBits::new(vec![1.0, 2.0, 4.0, 8.0], Objective::Maximize);
    // Two genomes that differ in two bits are each 0.707 from their average point, which is
    // 0.354 of the diagonal of length 2.0; in one bit, 0.5 from it, which is 0.25 of the
    // diagonal; and a single genome has no diversity.
    let high = population(
        &model,
        &[[true, false, false, false], [false, true, false, false]],
    );
    let middle = population(
        &model,
        &[[true, true, false, false], [true, false, false, false]],
    );
    let low = population(&model, &[[true, false, false, false]]);
    assert!(
        (model.diversity(&Population::new(high.clone()).unwrap()) - 0.5f64.sqrt() / 2.0).abs()
            < 1e-12
    );
    assert!((model.diversity(&Population::new(middle.clone()).unwrap()) - 0.25).abs() < 1e-12);

    let mut parameters = DiversityGuidedGeneticAlgorithmParameters::default();
    parameters.low_diversity = 0.1;
    parameters.high_diversity = 0.3;
    let mut algorithm = DiversityGuidedGeneticAlgorithm::new(model, parameters).unwrap();
    let mut current = RunState::new(Population::new(high.clone()).unwrap());
    let mut exploring = Vec::new();
    for next in [middle.clone(), low, middle.clone(), high, middle.clone()] {
        algorithm.prepare_generation(&current);
        exploring.push(algorithm.is_exploring());
        current
            .advance(|population| population.replace(&mut next.clone()))
            .unwrap();
    }
    algorithm.prepare_generation(&current);
    exploring.push(algorithm.is_exploring());
    // High, middle: exploiting. Low, middle: exploring. High, middle: exploiting.
    assert_eq!(exploring, [false, false, true, true, false, false]);

    // A new run starts by exploiting, so a diversity between the thresholds keeps it exploiting
    // even after the last run ended exploring, while the run before it keeps exploring.
    let single = population(algorithm.model(), &[[true; 4]]);
    algorithm.prepare_generation(&RunState::new(Population::new(single).unwrap()));
    assert!(algorithm.is_exploring());
    let middle = RunState::new(Population::new(middle).unwrap());
    algorithm.prepare_generation(&middle);
    assert!(algorithm.is_exploring());
    algorithm.start_run();
    algorithm.prepare_generation(&middle);
    assert!(!algorithm.is_exploring());
}

#[test]
fn exploration_mutates_every_candidate_and_exploitation_crosses_every_one_over() {
    let model = WeightedBits::new(vec![1.0; 4], Objective::Maximize);
    let mut algorithm = DiversityGuidedGeneticAlgorithm::new(model, Default::default()).unwrap();
    let single =
        RunState::new(Population::new(vec![algorithm.model().individual(&[true; 4])]).unwrap());
    algorithm.prepare_generation(&single);
    assert!(algorithm.is_exploring());
    assert_eq!(
        (algorithm.rates().crossover(), algorithm.rates().mutation()),
        (0.0, 1.0)
    );

    let spread = RunState::new(
        Population::new(population(
            algorithm.model(),
            &[[true, true, false, false], [false, false, true, true]],
        ))
        .unwrap(),
    );
    algorithm.prepare_generation(&spread);
    assert!(!algorithm.is_exploring());
    assert_eq!(
        (algorithm.rates().crossover(), algorithm.rates().mutation()),
        (1.0, 0.0)
    );
}

#[test]
fn distance_to_average_point_is_relative_to_the_diagonal() {
    let none: [[f64; 2]; 0] = [];
    assert_eq!(distance_to_average_point(none.iter().copied(), 1.0), 0.0);
    let same = [[0.3, 0.7]; 3];
    assert!(distance_to_average_point(same.iter().copied(), 1.0).abs() < 1e-12);
    // Each corner of a square of side 2.0 is the half diagonal from the center.
    let corners = [[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]];
    let diversity = distance_to_average_point(corners.iter().copied(), 8f64.sqrt());
    assert!((diversity - 0.5).abs() < 1e-12);
}

#[test]
fn diversity_preserves_distances_at_extreme_scales() {
    // The endpoints of a line are half its length from their average, at every scale.
    for scale in [
        1e-200,
        1e200,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
    ] {
        let points = [[0.0], [scale]];
        let diversity = distance_to_average_point(points.iter().copied(), scale);
        assert_eq!(diversity, 0.5, "scale {scale}");
    }
}

#[test]
fn diversity_preserves_distances_with_a_large_coordinate_offset() {
    let points = [[1e308], [1.1e308]];
    let diagonal = points[1][0] - points[0][0];
    assert_eq!(
        distance_to_average_point(points.iter().copied(), diagonal),
        0.5
    );
    let same = [[1e308]; 3];
    assert_eq!(
        distance_to_average_point(same.iter().copied(), diagonal),
        0.0
    );
}

#[test]
fn diversity_preserves_a_mean_below_the_smallest_coordinate_unit() {
    // The average is one third of the diagonal; distances are 1/3, 1/3, and 2/3.
    let unit = f64::from_bits(1);
    let points = [[0.0], [0.0], [unit]];
    let diversity = distance_to_average_point(points.iter().copied(), unit);
    assert!((diversity - 4.0 / 9.0).abs() <= 4.0 * f64::EPSILON);
}

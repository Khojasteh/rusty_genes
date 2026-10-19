use rusty_genes::*;
use std::hash::{Hash, Hasher};

/// An individual identified by its position, so that individuals with equal scores are distinct.
#[derive(Clone, PartialEq)]
struct Scored {
    id: usize,
    score: f64,
}

impl Hash for Scored {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Individual for Scored {
    fn score(&self) -> f64 {
        self.score
    }
}

/// Returns a population of individuals with the given scores, which it orders from worst to best
/// for `objective`.
fn population(scores: &[f64], objective: Objective) -> Population<Scored> {
    let individuals = scores
        .iter()
        .enumerate()
        .map(|(id, &score)| Scored { id, score })
        .collect();
    Population::with_objective(individuals, objective).unwrap()
}

#[test]
fn statistics_preserve_representable_results_at_extreme_scales() {
    for scale in [1.0, 1e154, 1e-200, f64::MIN_POSITIVE, f64::from_bits(1)] {
        let samples = [-scale, 0.0, scale];
        let batch = samples.into_iter().collect::<Stats>();
        let mut streaming = Stats::default();
        for sample in samples {
            streaming.record(sample);
        }
        for stats in [batch, streaming] {
            assert_eq!(stats.avg(), Some(0.0));
            // The sample variance of [-s, 0, s] is s squared, so its STD is s.
            assert!((stats.std().unwrap() / scale - 1.0).abs() <= 4.0 * f64::EPSILON);
            if scale >= 1.0 {
                assert!((stats.var().unwrap() / (scale * scale) - 1.0).abs() <= 4.0 * f64::EPSILON);
            }
            let snapshot = stats.snapshot().unwrap();
            assert_eq!(snapshot.avg(), stats.avg().unwrap());
            assert_eq!(snapshot.var(), stats.var().unwrap());
            assert_eq!(snapshot.std(), stats.std().unwrap());
        }
    }
    let stats = [-1e308, 1e308].into_iter().collect::<Stats>();
    assert_eq!(stats.avg(), Some(0.0));
    assert!(stats.std().unwrap().is_finite());
}

#[test]
fn statistics_keep_empty_singleton_and_changing_scale_semantics() {
    let empty = Stats::default();
    assert_eq!(empty.avg(), None);
    assert_eq!(empty.var(), None);
    assert_eq!(empty.std(), None);
    assert!(empty.snapshot().is_none());

    let mut stats = Stats::default();
    stats.record(0.0);
    assert_eq!(stats.var(), Some(0.0));
    assert_eq!(stats.std(), Some(0.0));
    stats.record(1e-200);
    stats.record(-1e-200);
    assert_eq!(stats.avg(), Some(0.0));
    assert!((stats.std().unwrap() / 1e-200 - 1.0).abs() <= 4.0 * f64::EPSILON);
    stats.record(1e154);
    stats.record(-1e154);
    assert_eq!(stats.count(), 5);
    assert_eq!(stats.avg(), Some(0.0));
    assert_eq!(stats.arg_min(), Some(4));
    assert_eq!(stats.arg_max(), Some(3));
    assert!((stats.var().unwrap() / 5e307 - 1.0).abs() <= 4.0 * f64::EPSILON);
}

#[test]
fn population_statistics_find_the_extrema_that_recording_finds() {
    let sequences: [&[f64]; 4] = [
        &[3.0, 1.0, 3.0, 2.0],
        &[1.0, 2.0, 1.0, 3.0, 3.0],
        &[0.0, -0.0, 0.0, -1.0],
        &[5.0, 5.0],
    ];
    for scores in sequences {
        for objective in [Objective::Maximize, Objective::Minimize] {
            let population = population(scores, objective);
            let recorded = population.iter().map(Individual::score).collect::<Stats>();
            let stats = population.stats();
            assert_eq!(stats.count(), recorded.count());
            assert_eq!(stats.min().to_bits(), recorded.min().unwrap().to_bits());
            assert_eq!(stats.max().to_bits(), recorded.max().unwrap().to_bits());
            assert_eq!(stats.arg_min(), recorded.arg_min().unwrap());
            assert_eq!(stats.arg_max(), recorded.arg_max().unwrap());
        }
    }
}

#[test]
fn population_statistics_preserve_representable_results_at_extreme_scales() {
    for scale in [1.0, 1e154, 1e-200, f64::MIN_POSITIVE, f64::from_bits(1)] {
        for objective in [Objective::Maximize, Objective::Minimize] {
            let population = population(&[-scale, 0.0, scale], objective);
            let stats = population.stats();
            assert_eq!(stats.avg(), 0.0);
            // The sample variance of [-s, 0, s] is s squared, so its STD is s.
            assert!((stats.std() / scale - 1.0).abs() <= 4.0 * f64::EPSILON);
            if scale >= 1.0 {
                assert!((stats.var() / (scale * scale) - 1.0).abs() <= 4.0 * f64::EPSILON);
            }
        }
    }
    let stats = *population(&[-1e308, 1e308], Objective::Maximize).stats();
    assert_eq!(stats.avg(), 0.0);
    assert!(stats.std().is_finite());

    let samples = [-1e154, -1e-200, 0.0, 1e-200, 1e154];
    let stats = *population(&samples, Objective::Maximize).stats();
    assert_eq!(stats.avg(), 0.0);
    assert!((stats.var() / 5e307 - 1.0).abs() <= 4.0 * f64::EPSILON);

    // A spread far below the magnitude of the scores keeps its precision.
    let samples = [1e9 + 1.0, 1e9 + 2.0, 1e9 + 3.0];
    let stats = *population(&samples, Objective::Maximize).stats();
    assert_eq!(stats.avg(), 1e9 + 2.0);
    assert!((stats.var() - 1.0).abs() <= 4.0 * f64::EPSILON);
}

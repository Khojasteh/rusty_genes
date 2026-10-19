use crate::support::*;
use rusty_genes::*;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Copy, PartialEq)]
struct MockIndividual {
    score: f64,
}

impl Hash for MockIndividual {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_f64(self.score, state);
    }
}

impl Individual for MockIndividual {
    fn score(&self) -> f64 {
        self.score
    }
}

const POPULATION_SIZE: usize = 10;
const ITERATIONS: usize = 100000;
const CONFIDENCE: f64 = 0.01;

fn create_population(score: impl Fn(usize) -> f64) -> Population<MockIndividual> {
    (0..POPULATION_SIZE)
        .map(|i| MockIndividual { score: score(i) })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn create_test_population() -> Population<MockIndividual> {
    create_population(|i| i as f64 / POPULATION_SIZE as f64)
}

fn observed_probabilities<S>(strategy: &S, population: &Population<MockIndividual>) -> Vec<f64>
where
    S: SelectionMechanism<MockIndividual>,
{
    observed_probabilities_at(strategy, population, 0)
}

/// Returns the observed probabilities of selecting each individual of `population` as
/// generation number `generation` of a run.
fn observed_probabilities_at<S>(
    strategy: &S,
    population: &Population<MockIndividual>,
    generation: usize,
) -> Vec<f64>
where
    S: SelectionMechanism<MockIndividual>,
{
    let mut rng = Rng::from_seed(0);
    let mut counts = vec![0; population.len()];

    let sampler = strategy.sampler(population, generation);
    for _ in 0..ITERATIONS {
        let (index, _) = sampler.select(&mut rng);
        counts[index] += 1;
    }
    counts
        .into_iter()
        .map(|count| count as f64 / ITERATIONS as f64)
        .collect()
}

fn assert_probabilities(observed: &[f64], expected: impl Fn(usize) -> f64) {
    for (index, &observed) in observed.iter().enumerate() {
        let expected = expected(index);
        assert!(
            (observed - expected).abs() < CONFIDENCE,
            "index {index}: observed {observed}, expected {expected}"
        );
    }
}

#[test]
fn tournament_selection_probabilities() {
    let tournament = TournamentSelection::new(2).unwrap();
    let population = create_test_population();
    let observed = observed_probabilities(&tournament, &population);

    let n = population.len() as f64;
    let size = tournament.size() as i32;
    assert_probabilities(&observed, |index| {
        (1.0 - (index as f64 / n).powi(size)) - (1.0 - ((index as f64 + 1.0) / n).powi(size))
    });
}

#[test]
fn roulette_wheel_selection_probabilities() {
    let roulette_wheel = RouletteWheelSelection::new();
    let population = create_test_population();
    let observed = observed_probabilities(&roulette_wheel, &population);

    let total_score = population.iter().map(|i| i.score).sum::<f64>();
    assert_probabilities(&observed, |index| population[index].score / total_score);
}

#[test]
fn roulette_wheel_selection_with_negative_score() {
    let roulette_wheel = RouletteWheelSelection::new();
    let population = create_population(|i| i as f64 - 20.0);
    let observed = observed_probabilities(&roulette_wheel, &population);

    // The scores are shifted by the lowest one, -20.
    let total_shifted = (0..POPULATION_SIZE).sum::<usize>() as f64;
    assert_probabilities(&observed, |index| index as f64 / total_shifted);
}

#[test]
fn roulette_wheel_selection_with_zero_score() {
    let roulette_wheel = RouletteWheelSelection::new();
    let population: Population<MockIndividual> =
        vec![MockIndividual { score: 0.0 }].try_into().unwrap();
    let observed = observed_probabilities(&roulette_wheel, &population);

    assert_probabilities(&observed, |_| 1.0);
}

#[test]
fn roulette_preserves_probabilities_at_large_score_scales() {
    for (scores, expected) in [
        (vec![-1e308, 0.0, 1e308], vec![0.0, 1.0 / 3.0, 2.0 / 3.0]),
        (vec![8e307, 1.6e308], vec![1.0 / 3.0, 2.0 / 3.0]),
        (vec![0.0, 1e-308, 2e-308], vec![0.0, 1.0 / 3.0, 2.0 / 3.0]),
    ] {
        let population = Population::new(
            scores
                .into_iter()
                .map(|score| MockIndividual { score })
                .collect(),
        )
        .unwrap();
        let observed = observed_probabilities(&RouletteWheelSelection::new(), &population);
        assert_probabilities(&observed, |index| expected[index]);
    }
}

#[test]
fn rank_selection_probabilities() {
    let rank = RankSelection::new();
    let population = create_test_population();
    let observed = observed_probabilities(&rank, &population);

    let total_ranks = (population.len() * (population.len() + 1) / 2) as f64;
    assert_probabilities(&observed, |index| (index + 1) as f64 / total_ranks);
}

#[test]
fn elitist_selection_probabilities() {
    let elitist = ElitistSelection::new(0.7).unwrap();
    let population = create_test_population();
    let observed = observed_probabilities(&elitist, &population);

    let selection_bound = (population.len() as f64 * elitist.ratio()).floor() as usize;
    assert_probabilities(&observed, |index| {
        if index >= selection_bound {
            1.0 / (population.len() - selection_bound) as f64
        } else {
            0.0
        }
    });
}

#[test]
fn boltzmann_selection_probabilities() {
    // We keep the temperature constant for this test (cooling_rate = 1.0)
    let boltzmann = BoltzmannSelection::new(5.0, 1.0).unwrap();
    let population = create_test_population();
    let observed = observed_probabilities(&boltzmann, &population);

    let scaled = |index: usize| (population[index].score / 5.0).exp();
    let total_scaled = (0..population.len()).map(scaled).sum::<f64>();
    assert_probabilities(&observed, |index| scaled(index) / total_scaled);
}

#[test]
fn boltzmann_selection_favors_fittest_after_cooling() {
    let boltzmann = BoltzmannSelection::new(5.0, 0.9).unwrap();
    let population = create_test_population();
    let observed = observed_probabilities_at(&boltzmann, &population, 100);

    // After 100 coolings, the temperature is about 1.3e-4, and the fittest individual is
    // the only one with a non-negligible weight.
    assert_probabilities(&observed, |index| {
        if index == population.len() - 1 {
            1.0
        } else {
            0.0
        }
    });
}

#[test]
fn boltzmann_selection_cools_once_per_generation_and_restarts_with_a_run() {
    let mut boltzmann = BoltzmannSelection::new(5.0, 0.5).unwrap();
    assert_eq!(boltzmann.temperature_at(0), 5.0);
    assert_eq!(boltzmann.temperature_at(1), 2.5);
    assert_eq!(boltzmann.temperature_at(1), 2.5);
    assert_eq!(boltzmann.temperature_at(3), 0.625);
    // A new run starts again at generation zero.
    assert_eq!(boltzmann.temperature_at(0), 5.0);
    assert_eq!(boltzmann.temperature_at(2), 1.25);
    // A new cooling rate applies from the initial temperature.
    boltzmann.set_cooling_rate(0.25).unwrap();
    assert_eq!(boltzmann.temperature_at(2), 0.3125);
}

#[test]
fn linear_selection_probabilities() {
    let n = POPULATION_SIZE as f64;
    for pressure in [1.0, 1.25, 1.5, 2.0] {
        let linear = LinearSelection::new(pressure).unwrap();
        let population = create_test_population();
        let observed = observed_probabilities(&linear, &population);

        assert_probabilities(&observed, |index| {
            ((2.0 - pressure) + 2.0 * (pressure - 1.0) * index as f64 / (n - 1.0)) / n
        });
    }
}

#[test]
fn selection_parameters_are_validated() {
    assert!(TournamentSelection::new(0).is_err());
    assert!(BoltzmannSelection::new(0.0, 0.9).is_err());
    assert!(BoltzmannSelection::new(1.0, f64::INFINITY).is_err());
    assert!(LinearSelection::new(0.9).is_err());
    assert!(LinearSelection::new(2.1).is_err());
    assert!(ElitistSelection::new(1.0).is_err());
    assert!(SelectionStrategy::Linear(f64::NAN).validate().is_err());
    assert!(SelectionStrategy::Rank.validate().is_ok());
}

#[test]
fn roulette_preserves_adjacent_score_differences() {
    let center = 1.5_f64;
    let scores = [
        -center,
        -f64::from_bits(center.to_bits() - 1),
        -f64::from_bits(center.to_bits() - 2),
    ];
    let population = Population::new(
        scores
            .into_iter()
            .map(|score| MockIndividual { score })
            .collect(),
    )
    .unwrap();
    let observed = observed_probabilities(&RouletteWheelSelection::new(), &population);
    assert_probabilities(&observed, |index| index as f64 / 3.0);
}

#[test]
fn boltzmann_scales_overflowing_differences_before_exponentiation() {
    let population = Population::new(vec![
        MockIndividual { score: -1e308 },
        MockIndividual { score: 1e308 },
    ])
    .unwrap();
    let observed =
        observed_probabilities(&BoltzmannSelection::new(1e308, 1.0).unwrap(), &population);
    let lower = (-2.0_f64).exp();
    assert_probabilities(&observed, |index| {
        if index == 0 {
            lower / (1.0 + lower)
        } else {
            1.0 / (1.0 + lower)
        }
    });
}

#[test]
fn boltzmann_preserves_probabilities_in_subnormal_score_units() {
    let unit = f64::from_bits(1);
    let population = Population::new(
        [0.0, unit, 2.0 * unit]
            .into_iter()
            .map(|score| MockIndividual { score })
            .collect(),
    )
    .unwrap();
    let boltzmann = BoltzmannSelection::new(unit, 0.5).unwrap();
    for generation in 0..2 {
        let observed = observed_probabilities_at(&boltzmann, &population, generation);
        let total = (-2.0_f64).exp() + (-1.0_f64).exp() + 1.0;
        assert_probabilities(&observed, |index| ((index as f64) - 2.0).exp() / total);
        assert_eq!(boltzmann.temperature_at(generation), unit);
    }
}

fn candidate(model: &WeightedBits, index: usize) -> BitGenome {
    let genes = (0..model.weights.len())
        .map(|i| i == index)
        .collect::<Vec<_>>();
    model.individual(&genes)
}

fn weighted_population(scores: &[f64], objective: Objective) -> Population<BitGenome> {
    let model = WeightedBits::new(scores.to_vec(), objective);
    Population::with_objective(
        (0..scores.len()).map(|i| candidate(&model, i)).collect(),
        objective,
    )
    .unwrap()
}

#[test]
fn every_selector_uses_minimization_order() {
    let population = weighted_population(&[0.0, 10.0, 20.0], Objective::Minimize);
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
        let selector = strategy.create_selector::<BitGenome>().unwrap();
        let sampler = selector.sampler(&population, 0);
        let mut rng = Rng::from_seed(0);
        let mut counts = [0; 3];
        for _ in 0..100_000 {
            counts[sampler.select(&mut rng).0] += 1;
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
fn every_selector_gives_equal_scores_equal_chances() {
    let population = weighted_population(&[0.0, 10.0, 10.0, 20.0], Objective::Maximize);
    let boltzmann_sum = (-2.0_f64).exp() + 2.0 * (-1.0_f64).exp() + 1.0;
    let strategies = [
        // The two individuals of score 10 win a tournament of two whenever the fittest
        // individual is not drawn but one of them is, which has a chance of 9/16 - 1/16.
        (
            SelectionStrategy::Tournament(2),
            [1.0 / 16.0, 4.0 / 16.0, 4.0 / 16.0, 7.0 / 16.0],
        ),
        // The tied individuals share the ranks 2 and 3.
        (SelectionStrategy::Rank, [0.1, 0.25, 0.25, 0.4]),
        (SelectionStrategy::Linear(2.0), [0.0, 0.25, 0.25, 0.5]),
        // Half of the population is excluded, which splits the tied individuals.
        (SelectionStrategy::Elitist(0.5), [0.0, 0.25, 0.25, 0.5]),
        (SelectionStrategy::RouletteWheel, [0.0, 0.25, 0.25, 0.5]),
        (
            SelectionStrategy::Boltzmann(10.0, 1.0),
            [
                (-2.0_f64).exp() / boltzmann_sum,
                (-1.0_f64).exp() / boltzmann_sum,
                (-1.0_f64).exp() / boltzmann_sum,
                1.0 / boltzmann_sum,
            ],
        ),
    ];
    for (strategy, expected) in strategies {
        let selector = strategy.create_selector::<BitGenome>().unwrap();
        let sampler = selector.sampler(&population, 0);
        let mut rng = Rng::from_seed(0);
        let mut counts = [0; 4];
        for _ in 0..100_000 {
            counts[sampler.select(&mut rng).0] += 1;
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
        let population = weighted_population(&scores, Objective::Minimize);
        let selector = RouletteWheelSelection::new();
        let sampler = selector.sampler(&population, 0);
        let mut rng = Rng::from_seed(0);
        let mut counts = [0; 3];
        for _ in 0..100_000 {
            counts[sampler.select(&mut rng).0] += 1;
        }
        for (count, expected) in counts.into_iter().zip(expected) {
            assert!((count as f64 / 100_000.0 - expected).abs() < 0.01);
        }
        assert_eq!(population.fittest().score(), scores[0]);
    }
}

#[test]
fn samplers_select_another_individual_by_index() {
    let mut rng = Rng::from_seed(0);
    let single = create_population(|_| 1.0);
    assert_eq!(single.len(), 1);
    let rank = RankSelection::new();
    assert!(rank.sampler(&single, 0).select_other(&mut rng, 0).is_none());

    let population = create_test_population();
    let tournament = TournamentSelection::new(4).unwrap();
    let sampler = tournament.sampler(&population, 0);
    let fittest = population.len() - 1;
    for _ in 0..1000 {
        let (index, individual) = sampler.select_other(&mut rng, fittest).unwrap();
        assert_ne!(index, fittest);
        assert_eq!(individual, &population[index]);
    }
}

/// Returns how often `sampler` selects each individual other than the one at `excluded`.
fn observed_other_probabilities(sampler: &dyn Sampler<'_, BitGenome>, excluded: usize) -> Vec<f64> {
    let mut rng = Rng::from_seed(0);
    let mut counts = vec![0; sampler.population().len()];
    for _ in 0..ITERATIONS {
        counts[sampler.select_other(&mut rng, excluded).unwrap().0] += 1;
    }
    counts
        .into_iter()
        .map(|count| count as f64 / ITERATIONS as f64)
        .collect()
}

#[test]
fn every_selector_selects_another_individual_as_if_selecting_until_it_does() {
    // The probabilities of selecting each individual, as in the test of equal chances.
    let population = weighted_population(&[0.0, 10.0, 10.0, 20.0], Objective::Maximize);
    let boltzmann_sum = (-2.0_f64).exp() + 2.0 * (-1.0_f64).exp() + 1.0;
    let strategies = [
        (
            SelectionStrategy::Tournament(2),
            [1.0 / 16.0, 4.0 / 16.0, 4.0 / 16.0, 7.0 / 16.0],
        ),
        (SelectionStrategy::Rank, [0.1, 0.25, 0.25, 0.4]),
        (SelectionStrategy::Linear(2.0), [0.0, 0.25, 0.25, 0.5]),
        (SelectionStrategy::Elitist(0.5), [0.0, 0.25, 0.25, 0.5]),
        (SelectionStrategy::RouletteWheel, [0.0, 0.25, 0.25, 0.5]),
        (
            SelectionStrategy::Boltzmann(10.0, 1.0),
            [
                (-2.0_f64).exp() / boltzmann_sum,
                (-1.0_f64).exp() / boltzmann_sum,
                (-1.0_f64).exp() / boltzmann_sum,
                1.0 / boltzmann_sum,
            ],
        ),
    ];
    for (strategy, selected) in strategies {
        let selector = strategy.create_selector::<BitGenome>().unwrap();
        let sampler = selector.sampler(&population, 0);
        // Excluding the fittest individual, or one of two with equal scores, leaves the others
        // their chances in proportion.
        for excluded in [3, 1] {
            let observed = observed_other_probabilities(&*sampler, excluded);
            let others = 1.0 - selected[excluded];
            for (index, &observed) in observed.iter().enumerate() {
                let expected = if index == excluded {
                    0.0
                } else {
                    selected[index] / others
                };
                assert!(
                    (observed - expected).abs() < CONFIDENCE,
                    "{strategy:?} without {excluded}: {observed:?}"
                );
            }
        }
    }
}

#[test]
fn others_are_equally_likely_when_a_selector_selects_only_the_excluded_individual() {
    // Elitist selection of the fittest quarter, and roulette-wheel selection of the only
    // individual with a positive score.
    for (scores, strategy) in [
        ([0.0, 10.0, 20.0, 30.0], SelectionStrategy::Elitist(0.75)),
        ([0.0, 0.0, 0.0, 30.0], SelectionStrategy::RouletteWheel),
    ] {
        let population = weighted_population(&scores, Objective::Maximize);
        let selector = strategy.create_selector::<BitGenome>().unwrap();
        let sampler = selector.sampler(&population, 0);
        let observed = observed_other_probabilities(&*sampler, 3);
        assert_probabilities(&observed, |index| if index == 3 { 0.0 } else { 1.0 / 3.0 });
    }
}

#[test]
fn large_tournaments_preserve_conditional_selection_probabilities() {
    for (objective, sign) in [(Objective::Maximize, 1.0), (Objective::Minimize, -1.0)] {
        for (scores, expected) in [
            (vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 0.0]),
            (vec![0.0, 1.0, 1.0, 2.0], vec![0.0, 0.5, 0.5, 0.0]),
            (vec![0.0, 2.0, 2.0], vec![0.0, 1.0, 0.0]),
        ] {
            let scores = scores
                .into_iter()
                .map(|score| sign * score)
                .collect::<Vec<_>>();
            let population = weighted_population(&scores, objective);
            let selector = TournamentSelection::new(4096).unwrap();
            let sampler = selector.sampler(&population, 0);
            let excluded = population.len() - 1;
            let mut rng = Rng::from_seed(0);
            let mut counts = vec![0; population.len()];
            for _ in 0..1000 {
                counts[sampler.select_other(&mut rng, excluded).unwrap().0] += 1;
            }
            for (count, expected) in counts.into_iter().zip(expected) {
                let observed = count as f64 / 1000.0;
                // The runners-up beat the worst with probability indistinguishable from 1;
                // tied runners-up split it, with sampling STD at most 0.016.
                assert!(
                    (observed - expected).abs() < 0.08,
                    "{objective:?}: {scores:?}: {observed}"
                );
                if expected == 0.0 {
                    assert_eq!(count, 0);
                }
            }
        }
    }
}

#[test]
fn custom_samplers_select_another_individual_by_default() {
    /// Selects the fittest individual three times in four.
    struct MostlyFittest<'a>(&'a Population<MockIndividual>);
    impl<'a> Sampler<'a, MockIndividual> for MostlyFittest<'a> {
        fn population(&self) -> &'a Population<MockIndividual> {
            self.0
        }
        fn select(&self, rng: &mut Rng) -> (usize, &'a MockIndividual) {
            let index = if rng.random_bool(0.75) {
                self.0.len() - 1
            } else {
                0
            };
            (index, &self.0[index])
        }
    }

    let population = create_test_population();
    let fittest = population.len() - 1;
    let sampler = MostlyFittest(&population);
    let mut rng = Rng::from_seed(0);
    let mut counts = vec![0; population.len()];
    for _ in 0..ITERATIONS {
        counts[sampler.select_other(&mut rng, fittest).unwrap().0] += 1;
    }
    // Three selections of the fittest individual, which happen with a chance of 27 in 64, are
    // followed by a uniform selection among the others.
    let uniform = 27.0 / 64.0 / fittest as f64;
    let observed = counts
        .into_iter()
        .map(|count| count as f64 / ITERATIONS as f64)
        .collect::<Vec<_>>();
    assert_probabilities(&observed, |index| {
        if index == 0 {
            1.0 - 27.0 / 64.0 + uniform
        } else if index == fittest {
            0.0
        } else {
            uniform
        }
    });
    assert!(
        MostlyFittest(&create_population(|_| 1.0))
            .select_other(&mut rng, 0)
            .is_none()
    );
}

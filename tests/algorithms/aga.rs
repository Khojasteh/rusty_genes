use rusty_genes::*;
use std::{
    hash::{Hash, Hasher},
    sync::Mutex,
};

/// An individual that records how it was bred: its first parent's score, whether it was
/// mutated, and the scores of its parents if it was crossed over.
#[derive(Debug, Clone)]
struct Tagged {
    id: usize,
    score: f64,
    mutated: bool,
    parents: Option<(f64, f64)>,
}

impl Tagged {
    fn new(id: usize, score: f64) -> Self {
        Self {
            id,
            score,
            mutated: false,
            parents: None,
        }
    }
}

impl PartialEq for Tagged {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Hash for Tagged {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Individual for Tagged {
    fn score(&self) -> f64 {
        self.score
    }
}

/// A model that keeps every candidate it improves, which it is given once for each candidate.
struct Recorder {
    objective: Objective,
    bred: Mutex<Vec<Tagged>>,
}

impl EvolutionModel for Recorder {
    type Individual = Tagged;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn crossover(&self, parent1: &Tagged, parent2: &Tagged, _: &mut Rng) -> Tagged {
        let (low, high) = if parent1.score <= parent2.score {
            (parent1.score, parent2.score)
        } else {
            (parent2.score, parent1.score)
        };
        Tagged {
            parents: Some((low, high)),
            ..Tagged::new(usize::MAX, parent1.score)
        }
    }

    fn mutate(&self, individual: &mut Tagged, _: &mut Rng) {
        individual.mutated = true;
    }

    fn improve(&self, individual: &mut Tagged, _: &mut Rng) {
        self.bred.lock().unwrap().push(individual.clone());
    }

    fn evaluate(&self, _: &mut Tagged) {}
}

/// Breeds `count` candidates from a population with the given scores, whose parents are
/// selected uniformly, and returns them.
fn breed(
    objective: Objective,
    scores: &[f64],
    parameters: AdaptiveGeneticAlgorithmParameters,
    count: usize,
) -> Vec<Tagged> {
    let individuals = scores
        .iter()
        .enumerate()
        .map(|(id, &score)| Tagged::new(id, score))
        .collect();
    let population = Population::with_objective(individuals, objective).unwrap();
    let model = Recorder {
        objective,
        bred: Mutex::new(Vec::new()),
    };
    let algorithm = AdaptiveGeneticAlgorithm::new(model, parameters).unwrap();
    let mut next = Generation::new(count);
    algorithm.generate_offspring(
        &RunState::new(population),
        &mut next,
        &mut Rng::from_seed(1),
    );
    let bred = algorithm.model().bred.lock().unwrap().clone();
    assert_eq!(bred.len(), count);
    bred
}

fn rates(
    crossover: f64,
    subaverage_crossover: f64,
    mutation: f64,
    subaverage_mutation: f64,
    minimum_mutation: f64,
) -> AdaptiveGeneticAlgorithmParameters {
    let mut parameters = AdaptiveGeneticAlgorithmParameters::default();
    parameters.crossover_rate = crossover;
    parameters.subaverage_crossover_rate = subaverage_crossover;
    parameters.mutation_rate = mutation;
    parameters.subaverage_mutation_rate = subaverage_mutation;
    parameters.minimum_mutation_rate = minimum_mutation;
    parameters.selection_strategy = SelectionStrategy::Elitist(0.0);
    parameters
}

#[test]
fn parents_that_include_a_fittest_individual_do_not_cross_over() {
    // The average score is 2.0, so a pair whose fitter parent scores 2.0 crosses over at the
    // full rate, and one that includes the fittest individual never does.
    for (objective, crossed) in [
        (Objective::Maximize, (1.0, 2.0)),
        (Objective::Minimize, (2.0, 3.0)),
    ] {
        let bred = breed(
            objective,
            &[1.0, 2.0, 3.0],
            rates(1.0, 1.0, 0.0, 0.0, 0.0),
            3000,
        );
        let pairs = bred.iter().filter_map(|c| c.parents).collect::<Vec<_>>();
        assert!(!pairs.is_empty(), "{objective:?}");
        assert!(pairs.iter().all(|&pair| pair == crossed), "{objective:?}");
        // Two of the three pairs include the fittest individual.
        assert!(pairs.len() < bred.len() / 2, "{objective:?}");
    }
}

#[test]
fn parents_less_fit_than_the_average_cross_over_at_their_own_rate() {
    // The average score is 4.0, and only the two least fit individuals are below it.
    for (objective, crossed) in [
        (Objective::Maximize, (1.0, 2.0)),
        (Objective::Minimize, (6.0, 7.0)),
    ] {
        let bred = breed(
            objective,
            &[1.0, 2.0, 6.0, 7.0],
            rates(0.0, 1.0, 0.0, 0.0, 0.0),
            3000,
        );
        let pairs = bred.iter().filter_map(|c| c.parents).collect::<Vec<_>>();
        assert!(!pairs.is_empty(), "{objective:?}");
        assert!(pairs.iter().all(|&pair| pair == crossed), "{objective:?}");
    }
}

#[test]
fn mutation_rates_fall_in_proportion_to_the_distance_to_the_fittest_score() {
    // The average score is 3.5. A copy of an individual less fit than the average is mutated at
    // the subaverage rate of 0.25. The individual scoring 5.0 when maximizing, or 2.0 when
    // minimizing, is 1.0 from the fittest score where the average is 2.5 from it, so its copies
    // are mutated at 0.4 of the full rate of 1.0, and copies of the fittest are not mutated.
    let scores = [1.0, 2.0, 5.0, 6.0];
    for (objective, expected) in [
        (Objective::Maximize, [0.25, 0.25, 0.4, 0.0]),
        (Objective::Minimize, [0.0, 0.4, 0.25, 0.25]),
    ] {
        let bred = breed(objective, &scores, rates(0.0, 0.0, 1.0, 0.25, 0.0), 40000);
        for (&score, &rate) in scores.iter().zip(&expected) {
            let copies = bred.iter().filter(|c| c.score == score).collect::<Vec<_>>();
            let mutated = copies.iter().filter(|c| c.mutated).count();
            let observed = mutated as f64 / copies.len() as f64;
            // Each score has about 10,000 copies, whose observed rate has a standard deviation
            // of at most 0.005.
            assert!(
                (observed - rate).abs() < 0.02,
                "{objective:?}: score {score} mutated at {observed}, not {rate}"
            );
            if rate == 0.0 {
                assert_eq!(mutated, 0, "{objective:?}: score {score}");
            }
        }
    }
}

#[test]
fn every_candidate_is_mutated_at_least_at_the_minimum_rate() {
    let bred = breed(
        Objective::Maximize,
        &[1.0, 2.0, 3.0],
        rates(0.0, 0.0, 0.0, 0.0, 1.0),
        1000,
    );
    assert!(bred.iter().all(|c| c.mutated));
}

#[test]
fn subnormal_score_differences_preserve_adaptive_mutation_rates() {
    let unit = f64::from_bits(1);
    for (objective, sign) in [(Objective::Maximize, 1.0), (Objective::Minimize, -1.0)] {
        let scores = [0.0, 2.0 * unit, 3.0 * unit, 4.0 * unit].map(|score| sign * score);
        // The stored mean rounds to 2 units. The parent at 3 units is halfway from it to
        // the fittest score, so it mutates at half the full rate.
        let bred = breed(objective, &scores, rates(0.0, 0.0, 1.0, 1.0, 0.0), 40000);
        for (&score, expected) in scores.iter().zip([1.0, 1.0, 0.5, 0.0]) {
            let copies = bred
                .iter()
                .filter(|candidate| candidate.score == score)
                .collect::<Vec<_>>();
            let mutated = copies.iter().filter(|candidate| candidate.mutated).count();
            let observed = mutated as f64 / copies.len() as f64;
            // About 10,000 copies per score give a standard deviation of at most 0.005.
            assert!(
                (observed - expected).abs() < 0.02,
                "{objective:?}: {score} mutated at {observed}"
            );
        }
    }
}

#[test]
fn a_population_of_equal_scores_is_not_disrupted() {
    // Every individual has the fittest score, so none is crossed over or mutated.
    let bred = breed(
        Objective::Maximize,
        &[2.0, 2.0, 2.0],
        rates(1.0, 1.0, 1.0, 1.0, 0.0),
        1000,
    );
    assert!(bred.iter().all(|c| c.parents.is_none() && !c.mutated));
}

#[test]
fn overflowing_score_distances_keep_adaptive_rates_proportional() {
    for (objective, sign) in [(Objective::Maximize, 1.0), (Objective::Minimize, -1.0)] {
        let scores = [-1.7e308, -1.7e308, -1.7e308, 0.0, 1.7e308].map(|score| sign * score);
        // The average is -0.4 times the best score. The zero-score parent's relative
        // distance to the best is 1 / 1.4 = 5/7, although the denominator overflows.
        let bred = breed(objective, &scores, rates(0.0, 0.0, 1.0, 1.0, 0.0), 40000);
        let copies = bred
            .iter()
            .filter(|candidate| candidate.score == 0.0)
            .collect::<Vec<_>>();
        let mutated = copies.iter().filter(|candidate| candidate.mutated).count();
        let observed = mutated as f64 / copies.len() as f64;
        // About 8,000 copies give a sampling STD below 0.006.
        assert!(
            (observed - 5.0 / 7.0).abs() < 0.03,
            "{objective:?}: {observed}"
        );
    }
}

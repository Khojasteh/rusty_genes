mod support;
use rusty_genes::*;
use std::{
    hash::{Hash, Hasher},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use support::*;

fn candidate(model: &WeightedBits, index: usize) -> BitGenome {
    let genes = (0..model.weights.len())
        .map(|i| i == index)
        .collect::<Vec<_>>();
    model.individual(&genes)
}

#[test]
fn deduplication_keeps_the_best_evaluation_of_the_same_genome() {
    let old = WeightedBits::new(vec![1.0], Objective::Maximize).individual(&[true]);
    let new = WeightedBits::new(vec![2.0], Objective::Maximize).individual(&[true]);
    for (objective, expected) in [(Objective::Maximize, 2.0), (Objective::Minimize, 1.0)] {
        let mut population =
            Population::with_objective(vec![old.clone(), new.clone()], objective).unwrap();
        assert_eq!(population.len(), 1);
        assert_eq!(population.fittest().score(), expected);
        population
            .replace(&mut vec![new.clone(), old.clone()])
            .unwrap();
        assert_eq!(population.fittest().score(), expected);
    }
}

#[test]
fn normalization_preserves_the_order_of_distinct_equal_score_genomes() {
    let model = WeightedBits::new(vec![1.0; 3], Objective::Maximize);
    let first = model.individual(&[true, false, false]);
    let second = model.individual(&[false, true, false]);
    let third = model.individual(&[false, false, true]);
    let mut population =
        Population::new(vec![first.clone(), second.clone(), first, third.clone()]).unwrap();
    assert_eq!(population.len(), 3);
    assert_eq!(population[0], second);
    assert_eq!(population[2], third);
    let mut invalid = vec![BitGenome::from(vec![])];
    assert!(matches!(
        population.replace(&mut invalid),
        Err(EvolutionError::EmptyPopulation)
    ));
    assert_eq!(population.len(), 3);
    assert!(invalid.is_empty());
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

#[derive(Debug, Clone, PartialEq)]
struct Mock {
    id: u32,
    score: f64,
}

impl Hash for Mock {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Individual for Mock {
    fn score(&self) -> f64 {
        self.score
    }
}

fn mock(id: u32, score: f64) -> Mock {
    Mock { id, score }
}

/// An individual whose equality ignores its score and counts the comparisons. It hashes its
/// key, or nothing if it has no key, so that its hash can collide with other identities.
#[derive(Debug, Clone)]
struct Counted {
    id: u64,
    score: f64,
    key: Option<u64>,
    comparisons: Arc<AtomicUsize>,
}

impl PartialEq for Counted {
    fn eq(&self, other: &Self) -> bool {
        self.comparisons.fetch_add(1, Ordering::Relaxed);
        self.id == other.id
    }
}

impl Hash for Counted {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if let Some(key) = self.key {
            key.hash(state);
        }
    }
}

impl Individual for Counted {
    fn score(&self) -> f64 {
        self.score
    }
}

fn counted_individuals(
    individuals: &[(u64, f64)],
    key: fn(u64) -> Option<u64>,
    comparisons: &Arc<AtomicUsize>,
) -> Vec<Counted> {
    individuals
        .iter()
        .map(|&(id, score)| Counted {
            id,
            score,
            key: key(id),
            comparisons: comparisons.clone(),
        })
        .collect()
}

#[test]
fn deduplication_compares_only_individuals_with_equal_hashes() {
    let comparisons = Arc::new(AtomicUsize::new(0));
    let individuals = (0..100).map(|id| (id, id as f64)).collect::<Vec<_>>();
    let population =
        Population::new(counted_individuals(&individuals, Some, &comparisons)).unwrap();

    assert_eq!(population.len(), 100);
    assert_eq!(comparisons.load(Ordering::Relaxed), 0);
}

#[test]
fn deduplication_keeps_the_best_evaluation_with_colliding_or_constant_hashes() {
    let individuals = [(1, 1.0), (2, 2.0), (1, 3.0), (3, 2.0), (2, 0.0)];
    let keys: [fn(u64) -> Option<u64>; 3] =
        [|id| Some(id % 2), |_| None, |id| (id != 3).then_some(id)];
    for key in keys {
        let comparisons = Arc::new(AtomicUsize::new(0));
        let population =
            Population::new(counted_individuals(&individuals, key, &comparisons)).unwrap();

        let kept = population
            .iter()
            .map(|individual| (individual.id, individual.score))
            .collect::<Vec<_>>();
        assert_eq!(kept, vec![(2, 2.0), (3, 2.0), (1, 3.0)]);
    }
}

#[test]
fn new_removes_non_adjacent_duplicates_with_equal_score() {
    let individuals = (0..300).map(|i| mock(i % 3, 1.0)).collect();
    let population = Population::new(individuals).unwrap();

    let mut ids = population.iter().map(|m| m.id).collect::<Vec<_>>();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 1, 2]);
}

#[test]
fn new_keeps_distinct_individuals_with_equal_score() {
    let individuals = vec![mock(1, 2.0), mock(2, 2.0), mock(3, 1.0)];
    let population = Population::new(individuals).unwrap();

    assert_eq!(population.len(), 3);
    assert_eq!(population[0].id, 3);
}

#[test]
fn new_discards_invalid_individuals_and_sorts_by_score() {
    let individuals = vec![
        mock(1, 3.0),
        mock(2, f64::NAN),
        mock(3, -1.0),
        mock(4, f64::INFINITY),
    ];
    let population = Population::new(individuals).unwrap();

    let ids = population.iter().map(|m| m.id).collect::<Vec<_>>();
    assert_eq!(ids, vec![3, 1]);
    assert_eq!(population.fittest().id, 1);
}

#[test]
fn new_fails_without_valid_individual() {
    assert_eq!(
        Population::<Mock>::new(vec![]).err(),
        Some(EvolutionError::EmptyPopulation)
    );
    assert_eq!(
        Population::new(vec![mock(1, f64::NAN)]).err(),
        Some(EvolutionError::EmptyPopulation)
    );
}

#[test]
fn replace_recomputes_the_statistics_of_the_new_generation() {
    let mut population = Population::new(vec![mock(1, 1.0), mock(2, 3.0)]).unwrap();
    assert_eq!(population.stats().avg(), 2.0);

    population
        .replace(&mut vec![mock(3, 5.0), mock(4, 9.0), mock(5, 7.0)])
        .unwrap();
    let stats = population.stats();
    assert_eq!(stats.count(), 3);
    assert_eq!((stats.min(), stats.max(), stats.avg()), (5.0, 9.0, 7.0));
    assert_eq!((stats.arg_min(), stats.arg_max()), (0, 2));
}

#[test]
fn replace_without_valid_offspring_leaves_population_unchanged() {
    let mut population = Population::new(vec![mock(1, 1.0)]).unwrap();
    let mut offspring = vec![mock(2, f64::NAN)];

    let result = population.replace(&mut offspring);

    assert_eq!(result, Err(EvolutionError::EmptyPopulation));
    assert_eq!(population.fittest().id, 1);
    assert!(offspring.is_empty());
}

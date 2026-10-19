use crate::{parameters, support::*};
use rusty_genes::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

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

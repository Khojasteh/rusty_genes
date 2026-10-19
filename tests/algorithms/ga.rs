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
        // A third bit of no weight leaves the breeding solutions that the population lacks.
        let model = WeightedBits::new(vec![1.0, 2.0, 0.0], objective);
        let mut algorithm = GeneticAlgorithm::new(model, parameters(4, 0.75)).unwrap();
        let mut individuals = vec![
            vec![false, false, false],
            vec![true, false, false],
            vec![false, true, false],
            vec![true, true, false],
        ]
        .into_iter()
        .map(BitGenome::from)
        .collect::<Vec<_>>();
        algorithm.evaluate_initial(&mut individuals);
        let state = RunState::new(Population::with_objective(individuals, objective).unwrap());
        algorithm.prepare_generation(&state);
        let mut offspring = Generation::new(4);
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
        fn sampler<'a>(
            &'a self,
            p: &'a Population<BitGenome>,
            _: usize,
        ) -> Box<dyn Sampler<'a, BitGenome> + 'a> {
            Box::new(CountingSampler(&self.0, p))
        }
    }
    struct CountingSampler<'a>(&'a AtomicUsize, &'a Population<BitGenome>);
    impl<'a> Sampler<'a, BitGenome> for CountingSampler<'a> {
        fn population(&self) -> &'a Population<BitGenome> {
            self.1
        }
        fn select(&self, _: &mut Rng) -> (usize, &'a BitGenome) {
            self.0.fetch_add(1, Ordering::Relaxed);
            (self.1.len() - 1, self.1.fittest())
        }
    }
    let selections = Arc::new(AtomicUsize::new(0));
    // Each candidate is bred from one parent, so a generation selects once per candidate.
    let mut configuration = parameters(4, 0.0);
    configuration.crossover_rate = 0.0;
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

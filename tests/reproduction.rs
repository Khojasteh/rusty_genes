mod support;
use rusty_genes::*;
use support::*;

struct FixedParents(usize);
impl SelectionMechanism<BitGenome> for FixedParents {
    fn prepare(&mut self, _: &Population<BitGenome>) {}
    fn select<'a>(&self, p: &'a Population<BitGenome>, _: &mut Rng) -> (usize, &'a BitGenome) {
        (self.0, &p[self.0])
    }
    fn select_distinct<'a>(
        &self,
        p: &'a Population<BitGenome>,
        _: &mut Rng,
        _: usize,
        _: &BitGenome,
    ) -> Option<(usize, &'a BitGenome)> {
        let i = 1 - self.0;
        Some((i, &p[i]))
    }
}

#[test]
fn survival_compares_evaluated_genomes_using_the_population_objective() {
    for (objective, initial) in [(Objective::Maximize, true), (Objective::Minimize, false)] {
        let model = WeightedBits::new(vec![1.0], objective);
        let population =
            Population::with_objective(vec![model.individual(&[initial])], objective).unwrap();
        let mut reproduction = Reproduction::new(FixedParents(0), 0.0, 1.0).unwrap();
        reproduction.prepare(&population);
        let preserved = reproduction.breed(&model, &population, &mut Rng::from_seed(0));
        assert_eq!(preserved.genes, vec![initial]);
        reproduction.set_survival(Generational);
        let replaced = reproduction.breed(&model, &population, &mut Rng::from_seed(0));
        assert_eq!(replaced.genes, vec![!initial]);
        assert_ne!(preserved.score(), replaced.score());
    }
}

#[test]
fn invalid_reevaluation_keeps_the_better_parent_in_either_selection_order() {
    for objective in [Objective::Maximize, Objective::Minimize] {
        let model = WeightedBits::new(vec![1.0, 2.0], objective);
        let population = Population::with_objective(
            vec![
                model.individual(&[true, false]),
                model.individual(&[false, true]),
            ],
            objective,
        )
        .unwrap();
        // A changed constraint rejects every offspring with the old genome length.
        let changed_problem = WeightedBits::new(vec![1.0, 2.0, 3.0], objective);
        for first in [0, 1] {
            let mut reproduction = Reproduction::new(FixedParents(first), 1.0, 0.0).unwrap();
            reproduction.prepare(&population);
            assert_eq!(
                reproduction.breed(&changed_problem, &population, &mut Rng::from_seed(0)),
                *population.fittest()
            );
        }
    }
}

#[test]
fn invalid_rates_leave_reproduction_unchanged() {
    assert!(Reproduction::<BitGenome>::new(RankSelection::new(), 1.5, 0.1).is_err());
    assert!(Reproduction::<BitGenome>::new(RankSelection::new(), 0.5, f64::NAN).is_err());
    let mut reproduction = Reproduction::<BitGenome>::new(RankSelection::new(), 0.5, 0.1).unwrap();
    assert!(reproduction.set_rates(0.2, -0.1).is_err());
    assert_eq!(
        (reproduction.crossover_rate(), reproduction.mutation_rate()),
        (0.5, 0.1)
    );
}

#[test]
fn parallel_breeding_is_reproducible() {
    let model = WeightedBits::new(vec![1.0; 8], Objective::Maximize);
    let population = Population::new(
        (0..8)
            .map(|i| model.individual(&(0..8).map(|j| i == j).collect::<Vec<_>>()))
            .collect(),
    )
    .unwrap();
    let mut reproduction =
        Reproduction::from_strategy(SelectionStrategy::Tournament(2), 0.9, 0.5).unwrap();
    reproduction.prepare(&population);
    let breed = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let mut offspring = Vec::new();
                reproduction.breed_into(
                    &model,
                    &population,
                    &mut offspring,
                    50,
                    &mut Rng::from_seed(9),
                );
                offspring
            })
    };
    assert_eq!(breed(1), breed(4));
}

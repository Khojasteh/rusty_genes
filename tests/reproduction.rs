mod support;
use rusty_genes::*;
use support::*;

/// Selects the individual at a fixed index first, and the other one of two individuals second.
struct FixedParents(usize);
impl SelectionMechanism<BitGenome> for FixedParents {
    fn sampler<'a>(
        &'a self,
        p: &'a Population<BitGenome>,
        _: usize,
    ) -> Box<dyn Sampler<'a, BitGenome> + 'a> {
        Box::new(FixedSampler(self.0, p))
    }
}

struct FixedSampler<'a>(usize, &'a Population<BitGenome>);
impl<'a> Sampler<'a, BitGenome> for FixedSampler<'a> {
    fn population(&self) -> &'a Population<BitGenome> {
        self.1
    }
    fn select(&self, _: &mut Rng) -> (usize, &'a BitGenome) {
        (self.0, &self.1[self.0])
    }
    fn select_other(&self, _: &mut Rng, _: usize) -> Option<(usize, &'a BitGenome)> {
        let i = 1 - self.0;
        Some((i, &self.1[i]))
    }
}

/// Returns the state of a run whose initial population holds the individuals of `population`.
fn run_state(population: &Population<BitGenome>) -> RunState<BitGenome> {
    RunState::new(Population::with_objective(population.to_vec(), population.objective()).unwrap())
}

fn sorted_genomes(individuals: &[BitGenome]) -> Vec<Vec<bool>> {
    let mut genomes = individuals
        .iter()
        .map(|individual| individual.genes.clone())
        .collect::<Vec<_>>();
    genomes.sort();
    genomes
}

fn one_hot_population(model: &WeightedBits, count: usize) -> Population<BitGenome> {
    let length = model.weights.len();
    Population::new(
        (0..count)
            .map(|i| model.individual(&(0..length).map(|j| i == j).collect::<Vec<_>>()))
            .collect(),
    )
    .unwrap()
}

fn assert_distinct(offspring: &[BitGenome]) {
    for (i, individual) in offspring.iter().enumerate() {
        assert!(
            offspring[..i].iter().all(|other| other != individual),
            "{:?} is repeated",
            individual.genes
        );
    }
}

#[test]
fn breeding_returns_the_candidate_and_its_parents() {
    // A copy of the parent is mutated even when that makes it less fit.
    let model = WeightedBits::new(vec![1.0], Objective::Maximize);
    let population = Population::new(vec![model.individual(&[true])]).unwrap();
    let reproduction = Reproduction::new(FixedParents(0), 0.0, 1.0).unwrap();
    let parents = reproduction.selector().sampler(&population, 0);
    let candidate = reproduction.breed(&model, &*parents, &mut Rng::from_seed(0));
    assert_eq!(candidate.individual().genes, vec![false]);
    assert_eq!(candidate.individual().score(), 0.0);
    assert_eq!(
        (candidate.fittest_parent(), candidate.other_parent()),
        (0, None)
    );

    // The single-point crossover of two genes takes the first gene of the first parent and the
    // second gene of the second parent, which is the fitter one.
    let model = WeightedBits::new(vec![1.0, 2.0], Objective::Maximize);
    let population = Population::new(vec![
        model.individual(&[true, false]),
        model.individual(&[false, true]),
    ])
    .unwrap();
    let reproduction = Reproduction::new(FixedParents(0), 1.0, 0.0).unwrap();
    let parents = reproduction.selector().sampler(&population, 0);
    let candidate = reproduction.breed(&model, &*parents, &mut Rng::from_seed(0));
    assert_eq!(candidate.individual().genes, vec![true, true]);
    assert_eq!(candidate.individual().score(), 3.0);
    assert_eq!(
        (candidate.fittest_parent(), candidate.other_parent()),
        (1, Some(0))
    );
}

#[test]
fn candidates_have_two_parents_even_when_only_the_fittest_can_be_selected() {
    // Elitist selection that excludes nine of ten individuals, and roulette-wheel selection in
    // which every other individual has a score of zero, only select the fittest individual.
    let only_last = (0..10).map(|i| if i < 9 { 0.0 } else { 1.0 }).collect();
    for (weights, strategy) in [
        (
            (1..=10).map(f64::from).collect(),
            SelectionStrategy::Elitist(0.9),
        ),
        (only_last, SelectionStrategy::RouletteWheel),
    ] {
        let model = WeightedBits::new(weights, Objective::Maximize);
        let population = one_hot_population(&model, 10);
        let reproduction = Reproduction::from_strategy(strategy, 1.0, 0.0).unwrap();
        let parents = reproduction.selector().sampler(&population, 0);
        let mut rng = Rng::from_seed(0);
        for _ in 0..100 {
            let candidate = reproduction.breed(&model, &*parents, &mut rng);
            assert_eq!(candidate.fittest_parent(), 9, "{strategy:?}");
            assert!(
                candidate.other_parent().is_some_and(|other| other != 9),
                "{strategy:?}"
            );
        }
    }
}

/// Sets a survival policy of a reproduction.
type SetSurvival = fn(&mut Reproduction<BitGenome>);

/// The survival policies of the library, by name.
fn policies() -> [(&'static str, SetSurvival); 3] {
    [
        ("family competition", |r| r.set_survival(FamilyCompetition)),
        ("place competition", |r| r.set_survival(PlaceCompetition)),
        ("generational", |r| r.set_survival(Generational)),
    ]
}

#[test]
fn survival_compares_evaluated_genomes_using_the_population_objective() {
    for (objective, initial) in [(Objective::Maximize, true), (Objective::Minimize, false)] {
        let model = WeightedBits::new(vec![1.0], objective);
        let population =
            Population::with_objective(vec![model.individual(&[initial])], objective).unwrap();
        // The candidate flips the only bit of its parent, which makes it less fit.
        let mut reproduction = Reproduction::new(FixedParents(0), 0.0, 1.0).unwrap();
        for ((policy, set_survival), survivor) in
            policies().into_iter().zip([initial, initial, !initial])
        {
            set_survival(&mut reproduction);
            let mut next = Generation::new(1);
            reproduction.breed_into(
                &model,
                &run_state(&population),
                &mut next,
                &mut Rng::from_seed(0),
            );
            assert_eq!(sorted_genomes(&next), [vec![survivor]], "{policy}");
        }
    }
}

#[test]
fn invalid_candidates_leave_the_population_in_place() {
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
        // A changed constraint rejects every candidate with the old genome length.
        let changed_problem = WeightedBits::new(vec![1.0, 2.0, 3.0], objective);
        for first in [0, 1] {
            let mut reproduction = Reproduction::new(FixedParents(first), 1.0, 0.0).unwrap();
            for (policy, set_survival) in policies() {
                set_survival(&mut reproduction);
                let mut next = Generation::new(2);
                reproduction.breed_into(
                    &changed_problem,
                    &run_state(&population),
                    &mut next,
                    &mut Rng::from_seed(0),
                );
                assert_eq!(
                    sorted_genomes(&next),
                    sorted_genomes(&population),
                    "{policy}"
                );
            }
        }
    }
}

#[test]
fn family_competition_keeps_a_candidate_or_else_its_fittest_parent() {
    let model = WeightedBits::new(vec![1.0, 2.0], Objective::Maximize);
    let population = Population::new(vec![
        model.individual(&[true, false]),
        model.individual(&[false, true]),
    ])
    .unwrap();

    // With the less fit parent first, the crossover is fitter than both parents.
    let reproduction = Reproduction::new(FixedParents(0), 1.0, 0.0).unwrap();
    let mut next = Generation::new(1);
    reproduction.breed_into(
        &model,
        &run_state(&population),
        &mut next,
        &mut Rng::from_seed(0),
    );
    assert_eq!(sorted_genomes(&next), [vec![true, true]]);

    // With the fitter parent first, the crossover is less fit than that parent, which survives.
    let reproduction = Reproduction::new(FixedParents(1), 1.0, 0.0).unwrap();
    let mut next = Generation::new(1);
    reproduction.breed_into(
        &model,
        &run_state(&population),
        &mut next,
        &mut Rng::from_seed(0),
    );
    assert_eq!(sorted_genomes(&next), [vec![false, true]]);
}

#[test]
fn place_competition_replaces_only_the_fittest_parent_of_a_fitter_candidate() {
    let model = WeightedBits::new(vec![1.0, 2.0], Objective::Maximize);
    let population = Population::new(vec![
        model.individual(&[true, false]),
        model.individual(&[false, true]),
    ])
    .unwrap();

    // With the less fit parent first, the crossover is fitter than both parents.
    let reproduction = Reproduction::new(FixedParents(0), 1.0, 0.0)
        .unwrap()
        .with_survival(PlaceCompetition);
    let mut next = Generation::new(2);
    reproduction.breed_into(
        &model,
        &run_state(&population),
        &mut next,
        &mut Rng::from_seed(0),
    );
    assert_eq!(sorted_genomes(&next), [vec![true, false], vec![true, true]]);

    // With the fitter parent first, the crossover is less fit than that parent, so both parents
    // keep their places.
    let reproduction = Reproduction::new(FixedParents(1), 1.0, 0.0)
        .unwrap()
        .with_survival(PlaceCompetition);
    let mut next = Generation::new(2);
    reproduction.breed_into(
        &model,
        &run_state(&population),
        &mut next,
        &mut Rng::from_seed(0),
    );
    assert_eq!(sorted_genomes(&next), sorted_genomes(&population));
}

#[test]
fn place_competition_gives_places_that_holders_leave_to_the_fittest_other_candidates() {
    for (objective, sign) in [(Objective::Maximize, 1.0), (Objective::Minimize, -1.0)] {
        let model = WeightedBits::new(vec![sign, 2.0 * sign, 4.0 * sign], objective);
        let population = Population::with_objective(
            vec![
                model.individual(&[true, false, false]),
                model.individual(&[false, false, true]),
            ],
            objective,
        )
        .unwrap();
        // Every candidate is bred from both individuals, so its fittest parent is the fitter one.
        let candidate = |genes: [bool; 3]| Candidate::new(model.individual(&genes), 1, Some(0));
        // The first candidate takes the place of the fitter individual, the second takes it from
        // the first, and the third is less fit than the second, so the first and the third are
        // left without a place.
        let candidates = vec![
            candidate([true, false, true]),
            candidate([true, true, true]),
            candidate([false, true, false]),
        ];

        // The two holders leave one place, which the fitter of the other candidates takes.
        let mut next = Generation::new(3);
        PlaceCompetition.select_survivors(
            &population,
            candidates,
            &mut next,
            &mut Rng::from_seed(0),
        );
        assert_eq!(
            sorted_genomes(&next),
            [
                vec![true, false, false],
                vec![true, false, true],
                vec![true, true, true]
            ],
            "{objective:?}"
        );
    }
}

#[test]
fn places_that_survivors_leave_go_to_the_fittest_individuals() {
    // Without crossover and mutation, every candidate copies the least fit individual.
    let model = WeightedBits::new(vec![1.0, 2.0, 4.0], Objective::Maximize);
    let population = one_hot_population(&model, 3);
    let least_fit = vec![true, false, false];
    let fittest = vec![false, false, true];
    let second_fittest = vec![false, true, false];
    // The copy survives in family competition and generational survival, and the fittest
    // individual takes the place of the copies that repeat it. In place competition, the copies
    // take no place, and only the two fittest of the individuals keeping their places fit in.
    let survivors = [
        [fittest.clone(), least_fit.clone()],
        [fittest.clone(), second_fittest],
        [fittest, least_fit],
    ];
    let mut reproduction = Reproduction::new(FixedParents(0), 0.0, 0.0).unwrap();
    for ((policy, set_survival), survivors) in policies().into_iter().zip(survivors) {
        set_survival(&mut reproduction);
        let mut next = Generation::new(2);
        reproduction.breed_into(
            &model,
            &run_state(&population),
            &mut next,
            &mut Rng::from_seed(0),
        );
        assert_eq!(sorted_genomes(&next), survivors, "{policy}");
    }
}

#[test]
fn repeated_candidates_do_not_shrink_the_generation() {
    // Without crossover and mutation, every candidate repeats an individual of the population.
    let model = WeightedBits::new(vec![1.0; 10], Objective::Maximize);
    let population = one_hot_population(&model, 4);
    let mut reproduction =
        Reproduction::from_strategy(SelectionStrategy::RouletteWheel, 0.0, 0.0).unwrap();
    for (policy, set_survival) in policies() {
        set_survival(&mut reproduction);
        let mut offspring = Generation::new(4);
        offspring.insert(population.fittest().clone());
        reproduction.breed_into(
            &model,
            &run_state(&population),
            &mut offspring,
            &mut Rng::from_seed(3),
        );
        assert_eq!(
            sorted_genomes(&offspring),
            sorted_genomes(&population),
            "{policy}"
        );
    }
}

#[test]
fn a_generation_is_smaller_only_when_the_problem_has_fewer_solutions() {
    // Two bits have only four genomes, fewer than the generation needs.
    let model = WeightedBits::new(vec![1.0; 2], Objective::Maximize);
    let population = one_hot_population(&model, 2);
    let reproduction =
        Reproduction::from_strategy(SelectionStrategy::RouletteWheel, 0.5, 0.5).unwrap();

    let mut offspring = Generation::new(10);
    reproduction.breed_into(
        &model,
        &run_state(&population),
        &mut offspring,
        &mut Rng::from_seed(3),
    );

    assert!((2..=4).contains(&offspring.len()), "{}", offspring.len());
    assert_distinct(&offspring);
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
    let population = one_hot_population(&model, 8);
    let reproduction =
        Reproduction::from_strategy(SelectionStrategy::Tournament(2), 0.9, 0.5).unwrap();
    let breed = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let mut offspring = Generation::new(50);
                reproduction.breed_into(
                    &model,
                    &run_state(&population),
                    &mut offspring,
                    &mut Rng::from_seed(9),
                );
                Vec::from(offspring)
            })
    };
    assert_eq!(breed(1), breed(4));
}

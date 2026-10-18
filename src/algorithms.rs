use super::*;
use rayon::prelude::*;

mod ga;
pub use ga::*;

mod aga;
pub use aga::*;

/// The type of individual evolved by an evolutionary algorithm `A`.
pub type IndividualOf<A> = <<A as EvolutionAlgorithm>::Model as EvolutionModel>::Individual;

/// Represents the core mechanism of an evolutionary algorithm.
///
/// # Note
/// This trait defines the steps of a generation: preparing the algorithm, carrying elites over,
/// breeding offspring, and replacing the population. [`Evolution`] drives these steps from one
/// generation to the next, so an algorithm is usually run by converting it into an [`Evolution`]
/// with [`into_evolution`](EvolutionAlgorithm::into_evolution).
///
/// A [`Reproduction`] provides the breeding step of the algorithms of this library, and can be
/// reused by other algorithms.
pub trait EvolutionAlgorithm {
    /// The evolutionary model the algorithm operates on.
    type Model: EvolutionModel;

    /// Returns a reference to the evolutionary model used by the algorithm.
    fn model(&self) -> &Self::Model;

    /// Returns the number of individuals in a full generation.
    fn population_size(&self) -> usize;

    /// Converts the algorithm into an [`Evolution`] that runs it.
    fn into_evolution(self) -> Evolution<Self>
    where
        Self: Sized,
    {
        Evolution::new(self)
    }

    /// Starts a new run, before its starting individuals are created.
    ///
    /// An algorithm that adapts its state during a run resets it here, so that every run
    /// starts as if the algorithm were new. [`Evolution`] calls this method before it reads
    /// the [`population_size`](EvolutionAlgorithm::population_size) or uses the model to create
    /// the starting individuals.
    ///
    /// The algorithm is also responsible for calling [`EvolutionModel::start_run`] on its model.
    fn start_run(&mut self);

    /// Prepares the initial population for the evolutionary process.
    ///
    /// # Arguments
    /// * `individuals` - The individuals of the initial population.
    ///
    /// # Note
    /// The default implementation evaluates the fitness of each individual in parallel and
    /// creates a [`Population`] from them, which discards the invalid individuals.
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if there is no valid individual.
    fn initialize_population(
        &mut self,
        mut individuals: Vec<IndividualOf<Self>>,
    ) -> Result<Population<IndividualOf<Self>>, EvolutionError> {
        let model = self.model();
        individuals
            .par_iter_mut()
            .for_each(|individual| model.evaluate(individual));
        Population::new(individuals)
    }

    /// Prepares the algorithm for breeding the next generation from the population of `state`.
    ///
    /// # Note
    /// This method is called once per generation, before any other step of the generation.
    ///
    /// The algorithm is also responsible for calling [`EvolutionModel::prepare`] on its model.
    fn prepare_generation(&mut self, state: &RunState<IndividualOf<Self>>);

    /// Carries the elite individuals of `population` over to the next generation by adding them
    /// to `elites`.
    ///
    /// # Note
    /// The default implementation adds the fittest individual.
    fn select_elites(
        &self,
        population: &Population<IndividualOf<Self>>,
        elites: &mut Vec<IndividualOf<Self>>,
    ) {
        elites.push(population.fittest().clone());
    }

    /// Breeds new individuals from the population of `state` and adds them to `offspring` until
    /// it holds [`population_size`](EvolutionAlgorithm::population_size) individuals.
    ///
    /// # Arguments
    /// * `state` - The state of the run, which holds the current population.
    /// * `offspring` - The individuals of the next generation, which may already hold its elites.
    /// * `rng` - A mutable reference to the random number generator of the run.
    ///
    /// # Note
    /// For a run to be reproducible from its seed, the offspring must depend only on the state
    /// of `rng`, for example by breeding each offspring with its own generator of [`Rng::split`],
    /// as [`Reproduction::breed_into`] does.
    fn generate_offspring(
        &self,
        state: &RunState<IndividualOf<Self>>,
        offspring: &mut Vec<IndividualOf<Self>>,
        rng: &mut Rng,
    );

    /// Replaces the current population with the new generation in `offspring`.
    ///
    /// # Note
    /// The default implementation replaces the whole population by [`Population::replace`].
    /// [`Evolution`] records the new generation in the [`RunState`] of the run once the
    /// population is replaced.
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if the new generation has no valid individual.
    #[inline]
    fn replace_population(
        &self,
        population: &mut Population<IndividualOf<Self>>,
        offspring: &mut Vec<IndividualOf<Self>>,
    ) -> Result<(), EvolutionError> {
        population.replace(offspring)
    }
}

/// Returns the number of elites for a population of the given size and elitism rate.
#[inline]
fn elite_count(population_size: usize, elitism_rate: f64) -> usize {
    interpolate_usize(0, population_size, elitism_rate)
}

/// Adds up to `count` of the fittest individuals of `population` to `elites`.
#[inline]
fn select_fittest<I: Individual>(population: &Population<I>, count: usize, elites: &mut Vec<I>) {
    let start = population.len().saturating_sub(count);
    elites.extend_from_slice(&population[start..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::one_max::*;
    use std::{
        ops::ControlFlow,
        sync::{
            Arc, Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    fn ga_parameters(population_size: usize, elitism_rate: f64) -> GeneticAlgorithmParameters {
        GeneticAlgorithmParameters {
            population_size,
            elitism_rate,
            ..Default::default()
        }
    }

    #[test]
    fn algorithms_and_evolutions_can_be_sent_to_other_threads() {
        fn assert_send<T: Send>() {}
        assert_send::<Evolution<GeneticAlgorithm<OneMaxEvolutionModel>>>();
        assert_send::<Evolution<AdaptiveGeneticAlgorithm<OneMaxEvolutionModel>>>();
    }

    #[test]
    fn ga_with_single_individual_population_evolves() {
        let mut evolution = GeneticAlgorithm::new(OneMaxEvolutionModel, ga_parameters(1, 0.0))
            .unwrap()
            .into_evolution();

        let outcome = evolution
            .evolve_population(vec![OneMax::from(vec![false; 8])], |s| {
                Limits::new().generations(50).check(s)
            })
            .unwrap();

        assert_eq!(outcome.state().generation(), 50);
        assert_eq!(outcome.state().population().len(), 1);
    }

    #[test]
    fn ga_breeds_the_last_free_slot_of_a_generation() {
        let mut ga = GeneticAlgorithm::new(OneMaxEvolutionModel, ga_parameters(4, 0.75)).unwrap();
        let initial = (0..4)
            .map(|i| OneMax::from((0..8).map(|bit| bit < i).collect::<Vec<_>>()))
            .collect();
        let state = RunState::new(ga.initialize_population(initial).unwrap());
        ga.prepare_generation(&state);

        let mut offspring = Vec::new();
        ga.select_elites(state.population(), &mut offspring);
        assert_eq!(offspring.len(), 3);

        ga.generate_offspring(&state, &mut offspring, &mut Rng::from_seed(0));
        assert_eq!(offspring.len(), 4);
    }

    #[test]
    fn ga_rejects_invalid_parameters() {
        let invalid = [
            GeneticAlgorithmParameters {
                population_size: 0,
                ..Default::default()
            },
            GeneticAlgorithmParameters {
                elitism_rate: 1.0,
                ..Default::default()
            },
            GeneticAlgorithmParameters {
                crossover_rate: 1.5,
                ..Default::default()
            },
            GeneticAlgorithmParameters {
                mutation_rate: f64::NAN,
                ..Default::default()
            },
            GeneticAlgorithmParameters {
                selection_strategy: SelectionStrategy::Tournament(0),
                ..Default::default()
            },
        ];

        for parameters in invalid {
            assert!(matches!(
                GeneticAlgorithm::new(OneMaxEvolutionModel, parameters),
                Err(EvolutionError::InvalidParameter { .. })
            ));
        }
    }

    /// An individual whose fitness is its value.
    #[derive(Debug, Clone, PartialEq)]
    struct Score(f64);

    impl Individual for Score {
        fn fitness(&self) -> f64 {
            self.0
        }
    }

    impl RandomIndividual<f64> for Score {
        fn new_random(value: &f64, _: &mut Rng) -> Self {
            Score(*value)
        }
    }

    /// A model whose offspring are always less fit than their parents.
    struct Decline;

    impl EvolutionModel for Decline {
        type Individual = Score;

        fn crossover(&self, parent1: &Score, parent2: &Score, _: &mut Rng) -> Score {
            Score(parent1.0.min(parent2.0) - 1.0)
        }

        fn mutate(&self, individual: &mut Score, _: &mut Rng) {
            individual.0 -= 10.0;
        }

        fn evaluate(&self, _: &mut Score) {}
    }

    /// An evolution without elites, whose offspring always replace their parents, so each
    /// generation is less fit than the one before.
    fn declining_evolution() -> Evolution<GeneticAlgorithm<Decline>> {
        let parameters = GeneticAlgorithmParameters {
            population_size: 4,
            elitism_rate: 0.0,
            crossover_rate: 1.0,
            mutation_rate: 1.0,
            selection_strategy: SelectionStrategy::Rank,
        };
        let mut ga = GeneticAlgorithm::new(Decline, parameters).unwrap();
        ga.reproduction_mut().set_survival(Generational);
        ga.into_evolution().with_seed(0)
    }

    #[test]
    fn run_returns_the_fittest_individual_of_any_generation() {
        let mut evolution = declining_evolution();

        let mut last_fitness = f64::NAN;
        let outcome = evolution
            .run(&100.0, |s| {
                last_fitness = s.population().fittest().fitness();
                Limits::new().generations(5).check(s)
            })
            .unwrap();

        assert!(last_fitness < 100.0);
        assert_eq!(outcome.reason(), &StopReason::GenerationLimit);
        assert_eq!(outcome.into_fittest_so_far(), Score(100.0));
    }

    #[test]
    fn evolve_individual_returns_the_fittest_individual_of_any_generation() {
        let mut evolution = declining_evolution();

        let mut last_fitness = f64::NAN;
        let best = evolution
            .evolve_individual(Score(100.0), |s| {
                last_fitness = s.population().fittest().fitness();
                Limits::new().generations(5).check(s)
            })
            .unwrap()
            .into_fittest_so_far();

        assert!(last_fitness < 100.0);
        assert_eq!(best, Score(100.0));
    }

    #[test]
    fn a_run_stops_with_the_reason_its_callback_gives() {
        for reason in [StopReason::Interrupted, StopReason::Custom("enough".into())] {
            let outcome = declining_evolution()
                .run(&100.0, |s| {
                    if s.generation() < 2 {
                        ControlFlow::Continue(())
                    } else {
                        ControlFlow::Break(reason.clone())
                    }
                })
                .unwrap();
            assert_eq!(outcome.reason(), &reason);
            assert_eq!(outcome.state().generation(), 2);
        }
    }

    /// A model of OneMax that mutates more bits in each generation of a run, and records how
    /// many bits it mutates at a time.
    struct Strengthening {
        strength: usize,
        strengths: Mutex<Vec<usize>>,
    }

    impl EvolutionModel for Strengthening {
        type Individual = OneMax;

        fn start_run(&mut self) {
            self.strength = 1;
        }

        fn prepare(&mut self, _: &RunState<OneMax>) {
            self.strength += 1;
        }

        fn crossover(&self, parent1: &OneMax, parent2: &OneMax, rng: &mut Rng) -> OneMax {
            OneMaxEvolutionModel.crossover(parent1, parent2, rng)
        }

        fn mutate(&self, individual: &mut OneMax, rng: &mut Rng) {
            self.strengths.lock().unwrap().push(self.strength);
            for _ in 0..self.strength {
                OneMaxEvolutionModel.mutate(individual, rng);
            }
        }

        fn evaluate(&self, individual: &mut OneMax) {
            OneMaxEvolutionModel.evaluate(individual);
        }
    }

    /// Asserts that two seeded runs of `algorithm` from the same individual mutate its starting
    /// copies at the strength of a new run, and end with the same population.
    fn assert_runs_start_afresh<A: EvolutionAlgorithm<Model = Strengthening>>(algorithm: A) {
        let mut evolution = algorithm.into_evolution().with_seed(7);
        let copies = evolution.algorithm().population_size() - 1;
        let mut populations = Vec::new();
        for _ in 0..2 {
            let outcome = evolution
                .evolve_individual(OneMax::from(vec![false; 8]), |s| {
                    Limits::new().generations(3).check(s)
                })
                .unwrap();
            let strengths =
                std::mem::take(&mut *evolution.algorithm().model().strengths.lock().unwrap());
            // The starting copies are mutated before the first generation is prepared.
            assert_eq!(strengths[..copies], vec![1; copies]);
            populations.push(outcome.into_state().into_population().to_vec());
        }
        assert_eq!(populations[0], populations[1]);
    }

    #[test]
    fn every_run_resets_the_algorithm_and_model_before_creating_its_starting_individuals() {
        let model = || Strengthening {
            strength: 1,
            strengths: Mutex::new(Vec::new()),
        };
        assert_runs_start_afresh(GeneticAlgorithm::new(model(), Default::default()).unwrap());
        assert_runs_start_afresh(
            AdaptiveGeneticAlgorithm::new(model(), Default::default()).unwrap(),
        );
    }

    /// A selector that selects the fittest individual and counts its selections.
    struct CountingSelector(Arc<AtomicUsize>);

    impl SelectionMechanism<OneMax> for CountingSelector {
        fn prepare(&mut self, _: &Population<OneMax>) {}

        fn select<'a>(
            &self,
            population: &'a Population<OneMax>,
            _: &mut Rng,
        ) -> (usize, &'a OneMax) {
            self.0.fetch_add(1, Ordering::Relaxed);
            let index = population.len() - 1;
            (index, &population[index])
        }
    }

    #[test]
    fn a_custom_selector_is_used_in_every_run() {
        let selections = Arc::new(AtomicUsize::new(0));
        let mut ga = GeneticAlgorithm::new(OneMaxEvolutionModel, ga_parameters(10, 0.0)).unwrap();
        ga.reproduction_mut()
            .set_selector(CountingSelector(selections.clone()));
        let mut evolution = ga.into_evolution();

        for _ in 0..2 {
            let before = selections.load(Ordering::Relaxed);
            evolution
                .run(&8, |s| Limits::new().generations(3).check(s))
                .unwrap();
            assert!(selections.load(Ordering::Relaxed) >= before + 30);
        }
    }

    #[test]
    fn runs_with_the_same_seed_are_equal_on_any_number_of_threads() {
        let run = |threads, seed| {
            let mut evolution =
                AdaptiveGeneticAlgorithm::new(OneMaxEvolutionModel, Default::default())
                    .unwrap()
                    .into_evolution()
                    .with_seed(seed);
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut individuals = Vec::new();
                    evolution
                        .run(&64, |state| {
                            individuals = state.population().to_vec();
                            Limits::new().generations(10).check(state)
                        })
                        .unwrap();
                    individuals
                })
        };

        assert_eq!(run(1, 5), run(4, 5));
        assert_ne!(run(4, 5), run(4, 6));
    }

    /// An individual whose fitness is the same everywhere, so the fitness of every population
    /// has a standard deviation of zero.
    #[derive(Debug, Clone, PartialEq)]
    struct Plateau(Vec<bool>);

    impl Individual for Plateau {
        fn fitness(&self) -> f64 {
            0.0
        }
    }

    struct PlateauModel;

    impl EvolutionModel for PlateauModel {
        type Individual = Plateau;

        fn crossover(&self, parent1: &Plateau, _: &Plateau, _: &mut Rng) -> Plateau {
            parent1.clone()
        }

        fn mutate(&self, individual: &mut Plateau, rng: &mut Rng) {
            MutationStrategy::Swap.mutate(&mut individual.0, rng);
        }

        fn evaluate(&self, _: &mut Plateau) {}
    }

    #[test]
    fn aga_keeps_mutating_when_fitness_has_no_spread() {
        let parameters = AdaptiveGeneticAlgorithmParameters {
            population_size: 2,
            elitism_rate: 0.0,
            ..Default::default()
        };
        let mut evolution = AdaptiveGeneticAlgorithm::new(PlateauModel, parameters)
            .unwrap()
            .into_evolution();
        // With two genes, every swap mutation moves the set bit.
        let mut first_bit_moved = false;
        evolution
            .evolve_population(vec![Plateau(vec![true, false])], |state| {
                first_bit_moved |= state.population().iter().any(|p| !p.0[0]);
                Limits::new().generations(5).check(state)
            })
            .unwrap();

        // Without any spread, diversity is zero: crossover is at the start of its range and
        // mutation at the end of its range (the defaults are 0.3..=1.0 and 0.1..=1.0).
        let reproduction = evolution.algorithm().reproduction();
        assert_eq!(reproduction.crossover_rate(), 0.3);
        assert!((reproduction.mutation_rate() - 1.0).abs() <= f64::EPSILON);
        assert!(first_bit_moved);
    }

    #[test]
    fn aga_rejects_invalid_parameters() {
        let invalid = [
            AdaptiveGeneticAlgorithmParameters {
                crossover_rate_range: 0.9..=0.3,
                ..Default::default()
            },
            AdaptiveGeneticAlgorithmParameters {
                mutation_rate_range: 0.1..=1.1,
                ..Default::default()
            },
            AdaptiveGeneticAlgorithmParameters {
                parameter_adjustment_interval: 0,
                ..Default::default()
            },
        ];

        for parameters in invalid {
            assert!(matches!(
                AdaptiveGeneticAlgorithm::new(OneMaxEvolutionModel, parameters),
                Err(EvolutionError::InvalidParameter { .. })
            ));
        }
    }

    #[test]
    fn evolution_fails_without_valid_initial_individual() {
        let mut evolution = GeneticAlgorithm::new(OneMaxEvolutionModel, Default::default())
            .unwrap()
            .into_evolution();

        let result =
            evolution.evolve_population(vec![OneMax::from(vec![])], |_| ControlFlow::Continue(()));

        assert_eq!(result.err(), Some(EvolutionError::EmptyPopulation));
    }
}

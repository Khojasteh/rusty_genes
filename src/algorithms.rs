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

    /// Prepares the algorithm for breeding the next generation from `population`.
    ///
    /// # Note
    /// This method is called once per generation, before any other step of the generation. When
    /// the generation number of `population` is zero, a new run starts, and the algorithm should
    /// reset any state it adapts during a run.
    ///
    /// The algorithm is also responsible for calling [`EvolutionModel::prepare`] on its model.
    fn prepare_generation(&mut self, population: &Population<IndividualOf<Self>>);

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

    /// Breeds new individuals from `population` and adds them to `offspring` until it holds
    /// [`population_size`](EvolutionAlgorithm::population_size) individuals.
    ///
    /// # Arguments
    /// * `population` - The current population.
    /// * `offspring` - The individuals of the next generation, which may already hold its elites.
    /// * `rng` - A mutable reference to the random number generator of the run.
    ///
    /// # Note
    /// For a run to be reproducible from its seed, the offspring must depend only on the state
    /// of `rng`, for example by breeding each offspring with its own generator of [`Rng::split`],
    /// as [`Reproduction::breed_into`] does.
    fn generate_offspring(
        &self,
        population: &Population<IndividualOf<Self>>,
        offspring: &mut Vec<IndividualOf<Self>>,
        rng: &mut Rng,
    );

    /// Replaces the current population with the new generation in `offspring`.
    ///
    /// # Note
    /// The default implementation replaces the whole population by [`Population::replace`].
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
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
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

        let population = evolution
            .evolve_population(vec![OneMax::from(vec![false; 8])], |p| p.generation() < 50)
            .unwrap();

        assert_eq!(population.generation(), 50);
        assert_eq!(population.len(), 1);
    }

    #[test]
    fn ga_breeds_the_last_free_slot_of_a_generation() {
        let mut ga = GeneticAlgorithm::new(OneMaxEvolutionModel, ga_parameters(4, 0.75)).unwrap();
        let initial = (0..4)
            .map(|i| OneMax::from((0..8).map(|bit| bit < i).collect::<Vec<_>>()))
            .collect();
        let population = ga.initialize_population(initial).unwrap();
        ga.prepare_generation(&population);

        let mut offspring = Vec::new();
        ga.select_elites(&population, &mut offspring);
        assert_eq!(offspring.len(), 3);

        ga.generate_offspring(&population, &mut offspring, &mut Rng::from_seed(0));
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
            evolution.run(&8, |p| p.generation() < 3).unwrap();
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
                        .run(&64, |population| {
                            individuals = population.to_vec();
                            population.generation() < 10
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
            .evolve_population(vec![Plateau(vec![true, false])], |population| {
                first_bit_moved |= population.iter().any(|p| !p.0[0]);
                population.generation() < 5
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

        let result = evolution.evolve_population(vec![OneMax::from(vec![])], |_| true);

        assert_eq!(result.err(), Some(EvolutionError::EmptyPopulation));
    }
}

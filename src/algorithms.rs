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

    /// Evaluates the starting individuals of a run, without requiring cached scores.
    ///
    /// # Note
    /// [`Evolution`] then forms the initial [`Population`] from them under the model's
    /// [`objective`](EvolutionModel::objective), discarding invalid individuals. The default
    /// implementation evaluates the score of each individual in parallel.
    fn evaluate_initial(&mut self, individuals: &mut [IndividualOf<Self>]) {
        let model = self.model();
        individuals
            .par_iter_mut()
            .for_each(|individual| model.evaluate(individual));
    }

    /// Prepares the algorithm for breeding the next generation from the population of `state`.
    ///
    /// # Note
    /// This method is called once per generation, before any other step of the generation.
    ///
    /// The algorithm is also responsible for calling [`EvolutionModel::prepare`] on its model.
    fn prepare_generation(&mut self, state: &RunState<IndividualOf<Self>>);

    /// Carries the elite individuals of `population` over to the next generation by adding them
    /// to `elites`, which is empty and holds at most
    /// [`population_size`](EvolutionAlgorithm::population_size) individuals.
    ///
    /// # Note
    /// The default implementation adds the fittest individual.
    fn select_elites(
        &self,
        population: &Population<IndividualOf<Self>>,
        elites: &mut Generation<IndividualOf<Self>>,
    ) {
        elites.insert(population.fittest().clone());
    }

    /// Breeds new individuals from the population of `state` and adds them to `offspring`, up
    /// to its [size](Generation::size), which is the
    /// [`population_size`](EvolutionAlgorithm::population_size).
    ///
    /// `offspring` may already hold elites; add only as many individuals as its capacity allows.
    ///
    /// # Note
    /// For a run to be reproducible from its seed, the offspring must depend only on the state
    /// of `rng`, for example by breeding each offspring with its own generator of [`Rng::split`],
    /// as [`Reproduction::breed_into`] does.
    fn generate_offspring(
        &self,
        state: &RunState<IndividualOf<Self>>,
        offspring: &mut Generation<IndividualOf<Self>>,
        rng: &mut Rng,
    );

    /// Replaces the current population with the next generation in `offspring`.
    ///
    /// # Note
    /// The default implementation replaces the whole population by [`Population::advance`].
    /// [`Evolution`] records the new generation in the [`RunState`] of the run once the
    /// population is replaced.
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if the next generation is empty.
    #[inline]
    fn replace_population(
        &self,
        population: &mut Population<IndividualOf<Self>>,
        offspring: &mut Generation<IndividualOf<Self>>,
    ) -> Result<(), EvolutionError> {
        population.advance(offspring)
    }
}

/// Returns the number of elites for a population of the given size and elitism rate.
#[inline]
fn elite_count(population_size: usize, elitism_rate: f64) -> usize {
    interpolate_usize(0, population_size, elitism_rate)
}

/// Adds up to `count` of the fittest individuals of `population` to `elites`.
#[inline]
fn select_fittest<I: Individual>(
    population: &Population<I>,
    count: usize,
    elites: &mut Generation<I>,
) {
    let start = population.len().saturating_sub(count);
    for elite in &population[start..] {
        elites.insert(elite.clone());
    }
}

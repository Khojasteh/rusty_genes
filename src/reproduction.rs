use super::*;
use rayon::prelude::*;

/// Breeds the next generation of a population: it selects parents, combines and varies them with
/// the operators of an [`EvolutionModel`] into candidates, and lets a [`SurvivalPolicy`] choose
/// the next generation from the population and the candidates.
///
/// Each candidate is bred as follows:
/// 1. Two parents are selected, the second among the other individuals by
///    [`Sampler::select_other`]. With the probability of their
///    [crossover rate](BreedingRates::crossover_rate), the candidate is bred by crossing them
///    over. Otherwise, and in a population of a single individual, the candidate is a copy of
///    the first parent.
/// 2. With the probability of its [mutation rate](BreedingRates::mutation_rate), the candidate
///    is mutated.
/// 3. The candidate is improved and evaluated.
///
/// The algorithm breeding with the reproduction gives it the rates. A new reproduction uses the
/// [`FamilyCompetition`] survival policy.
pub struct Reproduction<I: Individual> {
    selector: Box<dyn SelectionMechanism<I>>,
    survival: Box<dyn SurvivalPolicy<I>>,
}

impl<I: Individual> Reproduction<I> {
    /// Creates a reproduction that selects parents by `selector`.
    pub fn new(selector: impl SelectionMechanism<I> + 'static) -> Self {
        Self::with_boxed_selector(Box::new(selector))
    }

    /// Creates a reproduction that selects parents by the mechanism of `strategy`.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if a parameter of the strategy is invalid.
    pub fn from_strategy(strategy: SelectionStrategy) -> Result<Self, EvolutionError> {
        Ok(Self::with_boxed_selector(strategy.create_selector()?))
    }

    fn with_boxed_selector(selector: Box<dyn SelectionMechanism<I>>) -> Self {
        Self {
            selector,
            survival: Box::new(FamilyCompetition),
        }
    }

    /// Replaces the survival policy, and returns the reproduction.
    pub fn with_survival(mut self, survival: impl SurvivalPolicy<I> + 'static) -> Self {
        self.set_survival(survival);
        self
    }

    /// Replaces the mechanism that selects parents.
    pub fn set_selector(&mut self, selector: impl SelectionMechanism<I> + 'static) {
        self.selector = Box::new(selector);
    }

    /// Replaces the survival policy.
    pub fn set_survival(&mut self, survival: impl SurvivalPolicy<I> + 'static) {
        self.survival = Box::new(survival);
    }

    /// Returns the mechanism that selects parents.
    #[inline]
    pub fn selector(&self) -> &dyn SelectionMechanism<I> {
        &*self.selector
    }

    /// Breeds and evaluates one candidate, recording its parents' indices in the population.
    ///
    /// `parents` is a sampler created by [`SelectionMechanism::sampler`] for the population
    /// to breed from. `rates` supplies probabilities based on the selected parents; `model`
    /// supplies crossover, mutation, local search, and evaluation.
    ///
    /// The returned candidate may be invalid. This method applies no survival policy; its
    /// caller decides whether the candidate or a parent joins the next generation.
    pub fn breed<'a, M>(
        &self,
        model: &M,
        parents: &dyn Sampler<'a, I>,
        rates: &(impl BreedingRates<I> + ?Sized),
        rng: &mut Rng,
    ) -> Candidate<I>
    where
        M: EvolutionModel<Individual = I> + ?Sized,
        I: 'a,
    {
        let population = parents.population();
        let (first, parent1) = parents.select(rng);
        let second = parents
            .select_other(rng, first)
            .filter(|(_, parent2)| rng.random_bool(rates.crossover_rate(parent1, parent2)));

        let (mut individual, fittest_parent, other_parent) = match second {
            Some((second, parent2)) => {
                let child = model.crossover(parent1, parent2, rng);
                if population
                    .objective()
                    .is_fitter(parent2.score(), parent1.score())
                {
                    (child, second, Some(first))
                } else {
                    (child, first, Some(second))
                }
            }
            None => (parent1.clone(), first, None),
        };

        if rng.random_bool(rates.mutation_rate(&population[fittest_parent])) {
            model.mutate(&mut individual, rng);
        }

        model.improve(&mut individual, rng);
        model.evaluate(&mut individual);

        Candidate {
            individual,
            fittest_parent,
            other_parent,
        }
    }

    /// Breeds candidates from the population of `state` in parallel, then lets the survival
    /// policy fill `next`.
    ///
    /// The parents are selected by a sampler that the selector creates for the population and
    /// the [generation](RunState::generation) of `state`.
    /// `next` may already hold elites. As many candidates are bred as it lacks individuals to its
    /// [size](Generation::size), and the survival policy chooses the rest of the next
    /// generation from the population and the candidates. A generation holds only valid and
    /// distinct individuals, so `next` stays smaller than its size when the policy finds no more
    /// of them, such as when the problem has fewer solutions than the size.
    ///
    /// The candidates are bred in parallel, each with its own generator of [`Rng::split`], and
    /// the survival policy receives them in the order of those generators, with a generator of
    /// its own. The result therefore depends only on the state of `rng`, not on how the work is
    /// scheduled on threads, as long as the rates and the policy depend only on what they
    /// receive, and the model uses only the supplied generators for its random choices.
    pub fn breed_into<M>(
        &self,
        model: &M,
        state: &RunState<I>,
        next: &mut Generation<I>,
        rates: &(impl BreedingRates<I> + Sync + ?Sized),
        rng: &mut Rng,
    ) where
        M: EvolutionModel<Individual = I> + ?Sized,
    {
        let population = state.population();
        let parents = self.selector.sampler(population, state.generation());
        let count = next.size().saturating_sub(next.len());
        let streams = rng.split();
        let candidates = (0..count)
            .into_par_iter()
            .map(|index| self.breed(model, &*parents, rates, &mut streams.rng(index as u64)))
            .collect();
        self.survival.select_survivors(
            population,
            candidates,
            next,
            &mut streams.rng(count as u64),
        );
    }
}

/// The probabilities at which a [`Reproduction`] breeds each candidate by crossover, and mutates
/// it.
///
/// An algorithm decides the rates and gives them to the reproduction for each breeding. The
/// rates may depend on the parents of a candidate, such as the rates of the
/// [`AdaptiveGeneticAlgorithm`], or be the same for every candidate, as [`FixedRates`] are.
/// A rate is a probability between 0.0 and 1.0.
pub trait BreedingRates<I: Individual> {
    /// Returns the probability of breeding a candidate by crossing over `parent1` and `parent2`,
    /// rather than copying `parent1`.
    fn crossover_rate(&self, parent1: &I, parent2: &I) -> f64;

    /// Returns the probability of mutating a candidate whose fittest parent is `parent`: the
    /// parent it is a copy of, or the fitter of the two parents it was crossed over from.
    fn mutation_rate(&self, parent: &I) -> f64;
}

/// The same crossover and mutation rates for every candidate.
///
/// # Example
/// ```
/// use rusty_genes::*;
///
/// let rates = FixedRates::new(0.8, 0.1)?;
/// assert_eq!((rates.crossover(), rates.mutation()), (0.8, 0.1));
/// assert!(FixedRates::new(1.5, 0.1).is_err());
/// # Ok::<(), EvolutionError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedRates {
    crossover: f64,
    mutation: f64,
}

impl FixedRates {
    /// Sets the probabilities of crossover and mutation for every candidate.
    ///
    /// Crossover requires at least two individuals. Mutation can apply to either a crossed-over
    /// candidate or a copied parent, independently of whether crossover occurred.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] unless both rates are between 0.0 and 1.0,
    /// inclusive.
    pub fn new(crossover: f64, mutation: f64) -> Result<Self, EvolutionError> {
        check_probability("crossover_rate", crossover)?;
        check_probability("mutation_rate", mutation)?;
        Ok(Self::constant(crossover, mutation))
    }

    /// Creates rates that are known to be valid.
    pub(crate) const fn constant(crossover: f64, mutation: f64) -> Self {
        Self {
            crossover,
            mutation,
        }
    }

    /// Returns the probability of a candidate being bred by crossover from two parents.
    #[inline]
    pub fn crossover(&self) -> f64 {
        self.crossover
    }

    /// Returns the probability of a candidate being mutated.
    #[inline]
    pub fn mutation(&self) -> f64 {
        self.mutation
    }
}

impl<I: Individual> BreedingRates<I> for FixedRates {
    #[inline]
    fn crossover_rate(&self, _: &I, _: &I) -> f64 {
        self.crossover
    }

    #[inline]
    fn mutation_rate(&self, _: &I) -> f64 {
        self.mutation
    }
}

/// An individual bred as a candidate for the next generation, by a [`Reproduction`] or a custom
/// breeding step, with the parents it was bred from.
#[derive(Debug, Clone)]
pub struct Candidate<I: Individual> {
    individual: I,
    fittest_parent: usize,
    other_parent: Option<usize>,
}

impl<I: Individual> Candidate<I> {
    /// Creates a candidate for the next generation of a population, bred from the individuals at
    /// the given indices of that population.
    ///
    /// # Arguments
    /// * `individual` - The evaluated individual, which may be invalid.
    /// * `fittest_parent` - The index of the parent it was copied from, or of the fitter of the
    ///   two parents it was bred from.
    /// * `other_parent` - The index of the other parent of a candidate bred from two parents, or
    ///   [`None`] for a candidate copied from one parent.
    ///
    /// A custom breeding step creates its candidates by this function, so that a
    /// [`SurvivalPolicy`] can choose the next generation from them. [`FamilyCompetition`] and
    /// [`PlaceCompetition`] compare a candidate with its fittest parent, and panic if
    /// `fittest_parent` is not an index of the population.
    #[inline]
    pub fn new(individual: I, fittest_parent: usize, other_parent: Option<usize>) -> Self {
        Self {
            individual,
            fittest_parent,
            other_parent,
        }
    }

    /// Returns the evaluated individual, which may be invalid.
    #[inline]
    pub fn individual(&self) -> &I {
        &self.individual
    }

    /// Returns the index in the population of the fittest parent of the candidate: the parent
    /// it was copied from, or the fitter of the two parents it was bred from, or the first one
    /// when they are equally fit.
    #[inline]
    pub fn fittest_parent(&self) -> usize {
        self.fittest_parent
    }

    /// Returns the index in the population of the other parent of a candidate bred from two
    /// parents, or [`None`] for a candidate copied from one parent.
    #[inline]
    pub fn other_parent(&self) -> Option<usize> {
        self.other_parent
    }

    /// Returns the evaluated individual, consuming the candidate.
    #[inline]
    pub fn into_individual(self) -> I {
        self.individual
    }
}

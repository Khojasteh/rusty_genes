use super::*;
use rayon::prelude::*;

/// Breeds the next generation of a population: it selects parents, combines and varies them with
/// the operators of an [`EvolutionModel`] into candidates, and lets a [`SurvivalPolicy`] choose
/// the next generation from the population and the candidates.
///
/// # Note
/// Each candidate is bred as follows:
/// 1. A parent is selected and, with the probability of the crossover rate, a second parent is
///    selected among the other individuals by [`Sampler::select_other`] to cross over with.
///    Otherwise, and in a population of a single individual, the candidate is a copy of the
///    first parent.
/// 2. With the probability of the mutation rate, the candidate is mutated.
/// 3. The candidate is improved and evaluated.
///
/// A new reproduction uses the [`FamilyCompetition`] survival policy.
pub struct Reproduction<I: Individual> {
    selector: Box<dyn SelectionMechanism<I>>,
    survival: Box<dyn SurvivalPolicy<I>>,
    crossover_rate: f64,
    mutation_rate: f64,
}

impl<I: Individual> Reproduction<I> {
    /// Creates a reproduction that selects parents by `selector`.
    ///
    /// # Arguments
    /// * `selector` - The mechanism that selects parents.
    /// * `crossover_rate` - The probability of a candidate having two parents, in a population
    ///   of more than one individual.
    /// * `mutation_rate` - The probability of a candidate being mutated.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] unless both rates are between 0.0 and 1.0,
    /// inclusive.
    pub fn new(
        selector: impl SelectionMechanism<I> + 'static,
        crossover_rate: f64,
        mutation_rate: f64,
    ) -> Result<Self, EvolutionError> {
        Self::with_boxed_selector(Box::new(selector), crossover_rate, mutation_rate)
    }

    /// Creates a reproduction that selects parents by the mechanism of `strategy`.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] if a parameter of the strategy is invalid, or
    /// unless both rates are between 0.0 and 1.0, inclusive.
    pub fn from_strategy(
        strategy: SelectionStrategy,
        crossover_rate: f64,
        mutation_rate: f64,
    ) -> Result<Self, EvolutionError> {
        Self::with_boxed_selector(strategy.create_selector()?, crossover_rate, mutation_rate)
    }

    fn with_boxed_selector(
        selector: Box<dyn SelectionMechanism<I>>,
        crossover_rate: f64,
        mutation_rate: f64,
    ) -> Result<Self, EvolutionError> {
        check_rates(crossover_rate, mutation_rate)?;
        Ok(Self {
            selector,
            survival: Box::new(FamilyCompetition),
            crossover_rate,
            mutation_rate,
        })
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

    /// Returns the probability of a candidate having two parents, in a population of more than
    /// one individual.
    #[inline]
    pub fn crossover_rate(&self) -> f64 {
        self.crossover_rate
    }

    /// Returns the probability of a candidate being mutated.
    #[inline]
    pub fn mutation_rate(&self) -> f64 {
        self.mutation_rate
    }

    /// Sets the crossover and mutation rates.
    ///
    /// # Errors
    /// Returns [`EvolutionError::InvalidParameter`] unless both rates are between 0.0 and 1.0,
    /// inclusive. In that case, the rates are left unchanged.
    pub fn set_rates(
        &mut self,
        crossover_rate: f64,
        mutation_rate: f64,
    ) -> Result<(), EvolutionError> {
        check_rates(crossover_rate, mutation_rate)?;
        self.crossover_rate = crossover_rate;
        self.mutation_rate = mutation_rate;
        Ok(())
    }

    /// Returns the mechanism that selects parents.
    #[inline]
    pub fn selector(&self) -> &dyn SelectionMechanism<I> {
        &*self.selector
    }

    /// Breeds and evaluates one candidate, recording its parents' indices in the population.
    ///
    /// # Arguments
    /// * `model` - The model whose operators create, vary, and evaluate the candidate.
    /// * `parents` - The sampler of the parent selector for the population to breed from, as
    ///   created by [`SelectionMechanism::sampler`].
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// The returned candidate may be invalid. This method applies no survival policy; its
    /// caller decides whether the candidate or a parent joins the next generation.
    pub fn breed<'a, M>(
        &self,
        model: &M,
        parents: &dyn Sampler<'a, I>,
        rng: &mut Rng,
    ) -> Candidate<I>
    where
        M: EvolutionModel<Individual = I> + ?Sized,
        I: 'a,
    {
        let population = parents.population();
        let (first, parent1) = parents.select(rng);
        let second = if population.len() > 1 && rng.random_bool(self.crossover_rate) {
            parents.select_other(rng, first)
        } else {
            None
        };

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

        if rng.random_bool(self.mutation_rate) {
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

    /// Breeds candidates in parallel, then lets the survival policy fill `next`.
    ///
    /// # Arguments
    /// * `model` - The model whose operators create, vary, and evaluate the candidates.
    /// * `state` - The state of the run, whose population parents are selected from by a
    ///   sampler the selector creates for its generation.
    /// * `next` - The next generation, which may already hold its elites.
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// # Note
    /// As many candidates are bred as `next` lacks individuals to its
    /// [size](Generation::size), and the survival policy chooses the rest of the next
    /// generation from the population and the candidates. A generation holds only valid and
    /// distinct individuals, so `next` stays smaller than its size when the policy finds no more
    /// of them, such as when the problem has fewer solutions than the size.
    ///
    /// The candidates are bred in parallel, each with its own generator of [`Rng::split`], and
    /// the survival policy receives them in the order of those generators, with a generator of
    /// its own. The result therefore depends only on the state of `rng`, not on how the work is
    /// scheduled on threads, as long as the policy depends only on what it receives.
    pub fn breed_into<M>(
        &self,
        model: &M,
        state: &RunState<I>,
        next: &mut Generation<I>,
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
            .map(|index| self.breed(model, &*parents, &mut streams.rng(index as u64)))
            .collect();
        self.survival.select_survivors(
            population,
            candidates,
            next,
            &mut streams.rng(count as u64),
        );
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

fn check_rates(crossover_rate: f64, mutation_rate: f64) -> Result<(), EvolutionError> {
    check_probability("crossover_rate", crossover_rate)?;
    check_probability("mutation_rate", mutation_rate)
}

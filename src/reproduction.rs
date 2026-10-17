use super::*;
use rayon::prelude::*;

/// The number of attempts to select a second parent distinct from the first one.
const DISTINCT_PARENT_TRIES: usize = 3;

/// Breeds the offspring of a population: it selects parents, combines and varies them with the
/// operators of an [`EvolutionModel`], and decides by a [`SurvivalPolicy`] whether an offspring
/// or its fittest parent passes on to the next generation.
///
/// # Note
/// Each offspring is bred as follows:
/// 1. A parent is selected and, with the probability of the crossover rate, a second distinct
///    parent is selected to cross over with. Without a second parent, the offspring is a copy of
///    the first one.
/// 2. With the probability of the mutation rate, the offspring is mutated.
/// 3. The offspring is improved and evaluated.
/// 4. If the offspring is valid and survives by the survival policy, it is the result;
///    otherwise, the fittest parent is (keep-best reproduction).
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
    /// * `crossover_rate` - The probability of an offspring having two parents.
    /// * `mutation_rate` - The probability of an offspring being mutated.
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

    /// Returns the probability of an offspring having two parents.
    #[inline]
    pub fn crossover_rate(&self) -> f64 {
        self.crossover_rate
    }

    /// Returns the probability of an offspring being mutated.
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

    /// Restores the state the parent selector adapts during a run, for a new run.
    #[inline]
    pub fn reset(&mut self) {
        self.selector.reset();
    }

    /// Prepares the parent selector for breeding from `population`.
    ///
    /// # Note
    /// This method must be called for each new population before breeding from it.
    #[inline]
    pub fn prepare(&mut self, population: &Population<I>) {
        self.selector.prepare(population);
    }

    /// Breeds one offspring from `population`.
    ///
    /// # Arguments
    /// * `model` - The model whose operators create, vary, and evaluate the offspring.
    /// * `population` - The population to select parents from. It must be the population the
    ///   reproduction was last prepared for.
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// # Returns
    /// The evaluated offspring, or the fittest parent if the offspring is invalid or does not
    /// survive.
    pub fn breed<M>(&self, model: &M, population: &Population<I>, rng: &mut Rng) -> I
    where
        M: EvolutionModel<Individual = I> + ?Sized,
    {
        let (_, parent1) = self.selector.select(population, rng);
        let parent2 = if population.len() > 1 && rng.random_bool(self.crossover_rate) {
            self.selector
                .select_distinct(population, rng, DISTINCT_PARENT_TRIES, parent1)
                .map(|(_, parent)| parent)
        } else {
            None
        };

        let mut fittest_parent = parent1;
        let mut offspring = if let Some(parent2) = parent2 {
            if parent2.fitness() > parent1.fitness() {
                fittest_parent = parent2;
            }
            model.crossover(parent1, parent2, rng)
        } else {
            parent1.clone()
        };

        if rng.random_bool(self.mutation_rate) {
            model.mutate(&mut offspring, rng);
        }

        model.improve(&mut offspring, rng);
        model.evaluate(&mut offspring);

        if offspring.is_valid()
            && self
                .survival
                .survives(&offspring, fittest_parent, population, rng)
        {
            offspring
        } else {
            fittest_parent.clone()
        }
    }

    /// Breeds offspring from `population` in parallel and adds them to `offspring` until it holds
    /// `size` individuals.
    ///
    /// # Arguments
    /// * `model` - The model whose operators create, vary, and evaluate the offspring.
    /// * `population` - The population to select parents from. It must be the population the
    ///   reproduction was last prepared for.
    /// * `offspring` - The individuals of the next generation, which may already hold its elites.
    /// * `size` - The number of individuals of the next generation.
    /// * `rng` - A mutable reference to a random number generator.
    ///
    /// # Note
    /// Each offspring is bred with its own generator of [`Rng::split`], so the result depends
    /// only on the state of `rng`, not on how the work is scheduled on threads.
    pub fn breed_into<M>(
        &self,
        model: &M,
        population: &Population<I>,
        offspring: &mut Vec<I>,
        size: usize,
        rng: &mut Rng,
    ) where
        M: EvolutionModel<Individual = I> + ?Sized,
    {
        let count = size.saturating_sub(offspring.len());
        let streams = rng.split();
        offspring.par_extend((0..count).into_par_iter().map(|index| {
            let mut rng = streams.rng(index as u64);
            self.breed(model, population, &mut rng)
        }));
    }
}

fn check_rates(crossover_rate: f64, mutation_rate: f64) -> Result<(), EvolutionError> {
    check_probability("crossover_rate", crossover_rate)?;
    check_probability("mutation_rate", mutation_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Number(f64);

    impl Individual for Number {
        fn fitness(&self) -> f64 {
            self.0
        }
    }

    /// A model whose offspring is always worse than its parents.
    struct Decline;

    impl EvolutionModel for Decline {
        type Individual = Number;

        fn crossover(&self, parent1: &Number, parent2: &Number, _: &mut Rng) -> Number {
            Number(parent1.0.min(parent2.0) - 1.0)
        }

        fn mutate(&self, individual: &mut Number, _: &mut Rng) {
            individual.0 -= 10.0;
        }

        fn evaluate(&self, _: &mut Number) {}
    }

    fn bred_fitness(reproduction: Reproduction<Number>) -> Vec<f64> {
        let mut reproduction = reproduction;
        let population = Population::new(vec![Number(1.0), Number(2.0)]).unwrap();
        reproduction.prepare(&population);
        let mut rng = Rng::from_seed(0);
        (0..20)
            .map(|_| reproduction.breed(&Decline, &population, &mut rng).0)
            .collect()
    }

    #[test]
    fn family_competition_keeps_the_fittest_parent_of_a_worse_offspring() {
        let reproduction = Reproduction::new(RankSelection::new(), 1.0, 1.0).unwrap();
        assert!(bred_fitness(reproduction).iter().all(|&f| f >= 1.0));
    }

    #[test]
    fn generational_survival_keeps_a_worse_offspring() {
        let reproduction = Reproduction::new(RankSelection::new(), 1.0, 1.0)
            .unwrap()
            .with_survival(Generational);
        assert!(bred_fitness(reproduction).iter().all(|&f| f < 0.0));
    }

    #[test]
    fn invalid_rates_are_rejected_and_leave_the_rates_unchanged() {
        assert!(Reproduction::<Number>::new(RankSelection::new(), 1.5, 0.1).is_err());
        assert!(Reproduction::<Number>::new(RankSelection::new(), 0.5, f64::NAN).is_err());

        let mut reproduction = Reproduction::<Number>::new(RankSelection::new(), 0.5, 0.1).unwrap();
        assert!(reproduction.set_rates(0.2, -0.1).is_err());
        assert_eq!(
            (reproduction.crossover_rate(), reproduction.mutation_rate()),
            (0.5, 0.1)
        );
    }

    #[test]
    fn bred_offspring_depend_only_on_the_generator_state() {
        use crate::one_max::*;

        let model = OneMaxEvolutionModel;
        let mut individuals = (0..20)
            .map(|i| OneMax::from((0..16).map(|bit| (bit * i) % 3 == 0).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        individuals.iter_mut().for_each(|i| model.evaluate(i));
        let population = Population::new(individuals).unwrap();
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
}

use super::*;
use rayon::prelude::*;
use std::ops::Deref;

/// Represents a population of individuals in an evolutionary algorithm.
///
/// # Note
/// A population is never empty, and it holds only valid and distinct individuals, sorted by
/// their fitness score in ascending order. Individuals are distinct according to their
/// [`PartialEq`] implementation.
pub struct Population<I: Individual> {
    individuals: Vec<I>,
    generation: usize,
    stagnation: usize,
    stats: PopulatedStats,
}

impl<I: Individual> Population<I> {
    /// Creates a new population from the given individuals.
    ///
    /// Invalid individuals (see [`Individual::is_valid`]) and duplicates are discarded, and the
    /// remaining individuals are sorted by their fitness score.
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if no valid individual is given.
    pub fn new(mut individuals: Vec<I>) -> Result<Self, EvolutionError> {
        let stats = normalize(&mut individuals)?;
        Ok(Self {
            individuals,
            generation: 0,
            stagnation: 0,
            stats,
        })
    }

    /// Replaces the current population with a new generation.
    ///
    /// The individuals in `offspring` are normalized the same way as in [`Population::new`].
    /// On success, the generation counter is incremented, and the stagnation counter is reset
    /// if the fitness of the best individual has improved or incremented otherwise. Upon return,
    /// `offspring` contains the individuals of the previous generation.
    ///
    /// # Errors
    /// Returns [`EvolutionError::EmptyPopulation`] if `offspring` contains no valid individual.
    /// In that case, the population is left unchanged and `offspring` is left empty.
    pub fn replace(&mut self, offspring: &mut Vec<I>) -> Result<(), EvolutionError> {
        let stats = normalize(offspring)?;
        std::mem::swap(&mut self.individuals, offspring);
        let previous_best = self.stats.max();
        self.stats = stats;
        self.generation += 1;
        if self.stats.max() > previous_best {
            self.stagnation = 0;
        } else {
            self.stagnation += 1;
        }
        Ok(())
    }

    /// Returns the current generation number of the population.
    #[inline]
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Returns the number of generations since the last improvement of the best fitness score.
    #[inline]
    pub fn stagnation(&self) -> usize {
        self.stagnation
    }

    /// Returns the statistical information regarding the fitness score of the individuals
    /// in the current population.
    #[inline]
    pub fn stats(&self) -> &PopulatedStats {
        &self.stats
    }

    /// Returns the fittest individual in the population.
    #[inline]
    pub fn fittest(&self) -> &I {
        &self.individuals[self.stats.arg_max()]
    }

    /// Returns the fittest individual in the population while consuming the population.
    #[inline]
    pub fn take_fittest(mut self) -> I {
        self.individuals.swap_remove(self.stats.arg_max())
    }
}

impl<I: Individual> Deref for Population<I> {
    type Target = [I];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.individuals
    }
}

impl<I: Individual> TryFrom<Vec<I>> for Population<I> {
    type Error = EvolutionError;

    #[inline]
    fn try_from(individuals: Vec<I>) -> Result<Self, Self::Error> {
        Self::new(individuals)
    }
}

impl<I: Individual> From<Population<I>> for Vec<I> {
    #[inline]
    fn from(population: Population<I>) -> Self {
        population.individuals
    }
}

/// Removes invalid and duplicate individuals, sorts the rest by fitness, and collects the
/// statistics of their fitness scores.
fn normalize<I: Individual>(individuals: &mut Vec<I>) -> Result<PopulatedStats, EvolutionError> {
    individuals.retain(Individual::is_valid);
    // A stable sort keeps individuals with equal fitness in their given order, which keeps runs
    // reproducible.
    individuals.par_sort_by(compare_fitness);
    remove_duplicates(individuals);
    individuals
        .iter()
        .map(Individual::fitness)
        .collect::<Stats>()
        .snapshot()
        .ok_or(EvolutionError::EmptyPopulation)
}

/// Removes duplicate individuals from a vector that is sorted by fitness.
///
/// Equal individuals have equal fitness scores, so only individuals within the same run of
/// equal fitness scores are compared with each other. The cost is quadratic in the length of
/// the longest such run.
fn remove_duplicates<I: Individual>(individuals: &mut Vec<I>) {
    let mut kept = 0;
    let mut run_start = 0;
    for i in 0..individuals.len() {
        if kept != 0 && compare_fitness(&individuals[kept - 1], &individuals[i]).is_ne() {
            run_start = kept;
        }
        if !individuals[run_start..kept].contains(&individuals[i]) {
            individuals.swap(kept, i);
            kept += 1;
        }
    }
    individuals.truncate(kept);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Mock {
        id: u32,
        fitness: f64,
    }

    impl Individual for Mock {
        fn fitness(&self) -> f64 {
            self.fitness
        }
    }

    fn mock(id: u32, fitness: f64) -> Mock {
        Mock { id, fitness }
    }

    #[test]
    fn new_removes_non_adjacent_duplicates_with_equal_fitness() {
        let individuals = (0..300).map(|i| mock(i % 3, 1.0)).collect();
        let population = Population::new(individuals).unwrap();

        let mut ids = population.iter().map(|m| m.id).collect::<Vec<_>>();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn new_keeps_distinct_individuals_with_equal_fitness() {
        let individuals = vec![mock(1, 2.0), mock(2, 2.0), mock(3, 1.0)];
        let population = Population::new(individuals).unwrap();

        assert_eq!(population.len(), 3);
        assert_eq!(population[0].id, 3);
    }

    #[test]
    fn new_discards_invalid_individuals_and_sorts_by_fitness() {
        let individuals = vec![
            mock(1, 3.0),
            mock(2, f64::NAN),
            mock(3, -1.0),
            mock(4, f64::INFINITY),
        ];
        let population = Population::new(individuals).unwrap();

        let ids = population.iter().map(|m| m.id).collect::<Vec<_>>();
        assert_eq!(ids, vec![3, 1]);
        assert_eq!(population.fittest().id, 1);
    }

    #[test]
    fn new_fails_without_valid_individual() {
        assert_eq!(
            Population::<Mock>::new(vec![]).err(),
            Some(EvolutionError::EmptyPopulation)
        );
        assert_eq!(
            Population::new(vec![mock(1, f64::NAN)]).err(),
            Some(EvolutionError::EmptyPopulation)
        );
    }

    #[test]
    fn replace_tracks_generation_and_stagnation() {
        let mut population = Population::new(vec![mock(1, 1.0)]).unwrap();

        population.replace(&mut vec![mock(2, 1.0)]).unwrap();
        assert_eq!((population.generation(), population.stagnation()), (1, 1));

        population.replace(&mut vec![mock(3, 2.0)]).unwrap();
        assert_eq!((population.generation(), population.stagnation()), (2, 0));
    }

    #[test]
    fn replace_without_valid_offspring_leaves_population_unchanged() {
        let mut population = Population::new(vec![mock(1, 1.0)]).unwrap();
        let mut offspring = vec![mock(2, f64::NAN)];

        let result = population.replace(&mut offspring);

        assert_eq!(result, Err(EvolutionError::EmptyPopulation));
        assert_eq!(population.generation(), 0);
        assert_eq!(population.fittest().id, 1);
        assert!(offspring.is_empty());
    }
}

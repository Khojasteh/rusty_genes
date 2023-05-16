use rand::prelude::*;
use std::ops::{Deref, DerefMut};

/// Represents an individual in a population of an evolutionary algorithm.
///
/// # Note
/// The `EvolutionaryIndividual` trait defines the interface for an individual in an evolutionary
/// algorithm population. An individual is characterized by its fitness score, which measures how
/// well it performs the task or problem that the evolutionary algorithm is trying to optimize.
pub trait Individual: Clone + PartialEq + Send + Sync {
    /// Gets the fitness score of the individual.
    ///
    /// # Note
    /// The `fitness` method returns a floating-point value that represents the fitness
    /// score of the individual. A higher fitness score indicates that the individual is
    /// better suited to the task or problem being optimized.
    fn fitness(&self) -> f64;

    /// Returns [`true`] if the individual is an valid solution. An invalid individual will be
    /// removed from the population.
    ///
    /// # Note
    /// The default implementation checks whether the fitness score of the individual is neither
    /// infinite nor NaN. In such case, the method returns [`true`], indicating that the individual
    /// is valid and can be considered as a solution to the problem being optimized.
    ///
    /// Individual implementations of this trait may choose to override this method with a more
    /// specific definition of invalidity.
    #[inline]
    fn is_valid(&self) -> bool {
        self.fitness().is_finite()
    }
}

/// Represents an individual in a population of an evolutionary algorithm that is generated randomly.
///
/// # Note
/// The `RandomIndividual` trait defines the interface for an individual in an evolutionary algorithm
/// population that is generated randomly. An individual is characterized by its fitness score, which
/// measures how well it performs the task or problem that the evolutionary algorithm is trying to
/// optimize.
///
/// The `RandomIndividual` trait is parameterized over a type `T`, which represents domain-specific
/// arguments for generating the individual. The `new_random` method generates a new individual with
/// random characteristics using the specified random number generator and the domain-specific
/// arguments.
pub trait RandomIndividual<A>: Individual {
    /// Generates a new individual with random characteristics using the specified random number
    /// generator.
    ///
    /// # Arguments
    /// * `args` - A reference to domain-specific arguments for generating the individual.
    /// * `rng` - A mutable reference to random number generator.
    ///
    /// # Type Parameters
    /// * `R` - The type of random number generator.
    ///
    /// # Returns
    /// A new individual that is generated randomly with characteristics that are determined by the
    /// random number generator and the domain-specific arguments.
    fn new_random<R: Rng>(args: &A, rng: &mut R) -> Self;
}

/// Wrapper type for objects implementing the [`Individual`] trait, allowing them to be placed
/// on a heap.
///
/// # Note
/// This struct serves as a wrapper for objects implementing the [`Individual`] trait. The purpose
/// of this wrapper is to enable these objects to be allocated on the heap, which may be necessary
/// for certain operations or data structures. The wrapper provides a convenient way to store and
/// manage these objects without losing their original type information.
pub struct IndividualWrapper<I: Individual> {
    i: I,
}

impl<I: Individual> IndividualWrapper<I> {
    /// Consumes the `IndividualWrapper` and returns the wrapped [`Individual`] object.
    ///
    /// # Note
    /// This method allows you to retrieve the underlying [`Individual`] object from the wrapper
    /// and take ownership of it. The `IndividualWrapper` will be consumed in the process.
    #[inline]
    pub fn take(self) -> I {
        self.i
    }
}

impl<I: Individual> Deref for IndividualWrapper<I> {
    type Target = I;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.i
    }
}

impl<I: Individual> DerefMut for IndividualWrapper<I> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.i
    }
}

impl<I: Individual> PartialEq for IndividualWrapper<I> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.i.fitness() == other.i.fitness()
    }
}

impl<I: Individual> Eq for IndividualWrapper<I> {}

impl<I: Individual> PartialOrd for IndividualWrapper<I> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.i.fitness().partial_cmp(&other.i.fitness())
    }
}

impl<I: Individual> Ord for IndividualWrapper<I> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl<I: Individual> From<I> for IndividualWrapper<I> {
    #[inline]
    fn from(i: I) -> Self {
        Self { i }
    }
}

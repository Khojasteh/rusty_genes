use super::*;
use std::{
    collections::{HashMap, hash_map::Entry},
    hash::BuildHasherDefault,
    ops::Deref,
};

/// The next generation of a population while a [`SurvivalPolicy`] chooses its individuals.
///
/// Like a [`Population`], a generation holds only valid and distinct individuals, and it holds
/// at most its [size](Generation::size). It keeps the individuals in the order they were added;
/// the population that the generation becomes sorts them by score.
///
/// # Example
/// ```
/// use rusty_genes::*;
/// use std::hash::{Hash, Hasher};
///
/// #[derive(Clone, PartialEq)]
/// struct Number(f64);
///
/// impl Hash for Number {
///     fn hash<H: Hasher>(&self, state: &mut H) {
///         hash_f64(self.0, state);
///     }
/// }
///
/// impl Individual for Number {
///     fn score(&self) -> f64 {
///         self.0
///     }
/// }
///
/// let mut generation = Generation::new(2);
/// assert!(generation.insert(Number(1.0)));
/// assert!(!generation.insert(Number(1.0))); // A duplicate.
/// assert!(!generation.insert(Number(f64::NAN))); // Invalid.
/// assert!(generation.insert(Number(2.0)));
/// assert!(!generation.insert(Number(3.0))); // The generation is full.
/// assert!(generation.is_full());
/// assert_eq!(generation.len(), 2);
/// ```
pub struct Generation<I: Individual> {
    individuals: Vec<I>,
    index: HashIndex,
    size: usize,
}

impl<I: Individual> Generation<I> {
    /// Creates an empty generation that holds at most `size` individuals.
    /// A size of zero is allowed and makes it full immediately.
    pub fn new(size: usize) -> Self {
        Self {
            individuals: Vec::with_capacity(size),
            index: HashIndex::default(),
            size,
        }
    }

    /// Returns the number of individuals in a full generation.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Returns [`true`] if the generation holds [`size`](Generation::size) individuals.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.individuals.len() >= self.size
    }

    /// Returns [`true`] if the generation holds an individual equal to `individual`.
    pub fn contains(&self, individual: &I) -> bool {
        self.position_of(individual, hash_of(individual)).is_some()
    }

    /// Adds `individual` and returns [`true`] if it is valid, distinct, and there is room.
    /// Otherwise returns [`false`] without changing the generation.
    pub fn insert(&mut self, individual: I) -> bool {
        let hash = hash_of(&individual);
        self.insert_hashed(individual, hash)
    }

    /// Adds `individual`, whose hash by [`hash_of`] is `hash`, as [`insert`](Self::insert) does.
    pub(crate) fn insert_hashed(&mut self, individual: I, hash: u64) -> bool {
        let accepted = self.accepts(&individual, hash);
        if accepted {
            self.push(individual, hash);
        }
        accepted
    }

    /// Adds a clone of `individual`, whose hash by [`hash_of`] is `hash`, as
    /// [`insert`](Self::insert) adds `individual`, cloning it only when it is added.
    pub(crate) fn insert_clone_hashed(&mut self, individual: &I, hash: u64) -> bool {
        let accepted = self.accepts(individual, hash);
        if accepted {
            self.push(individual.clone(), hash);
        }
        accepted
    }

    /// Returns whether [`insert`](Self::insert) would add `individual`, whose hash by
    /// [`hash_of`] is `hash`.
    fn accepts(&self, individual: &I, hash: u64) -> bool {
        !self.is_full() && individual.is_valid() && self.position_of(individual, hash).is_none()
    }

    fn push(&mut self, individual: I, hash: u64) {
        self.index.insert(hash, self.individuals.len());
        self.individuals.push(individual);
    }

    /// Removes every individual from the generation, keeping its size.
    pub fn clear(&mut self) {
        self.individuals.clear();
        self.index.clear();
    }

    /// Moves the individuals of the generation into `individuals`, replacing its content, and
    /// leaves the generation empty. Both allocations are retained for reuse.
    pub(crate) fn move_into(&mut self, individuals: &mut Vec<I>) {
        std::mem::swap(&mut self.individuals, individuals);
        self.clear();
    }

    fn position_of(&self, individual: &I, hash: u64) -> Option<usize> {
        self.index
            .find(hash, |position| self.individuals[position] == *individual)
    }
}

impl<I: Individual> Deref for Generation<I> {
    type Target = [I];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.individuals
    }
}

impl<I: Individual> From<Generation<I>> for Vec<I> {
    #[inline]
    fn from(generation: Generation<I>) -> Self {
        generation.individuals
    }
}

/// Finds the positions of equal individuals by their hashes.
///
/// Equal individuals have equal hashes, so only the individuals at positions with the same hash
/// are compared.
#[derive(Debug, Default)]
pub(crate) struct HashIndex {
    // The first position with each hash, and the later positions with the same hashes, which
    // are rare unless the individuals share a constant hash.
    first: HashMap<u64, usize, BuildHasherDefault<GeneHasher>>,
    others: Vec<(u64, usize)>,
}

impl HashIndex {
    /// Returns the first position with `hash` for which `is_equal` returns [`true`].
    pub(crate) fn find(&self, hash: u64, mut is_equal: impl FnMut(usize) -> bool) -> Option<usize> {
        let &first = self.first.get(&hash)?;
        if is_equal(first) {
            return Some(first);
        }
        self.others
            .iter()
            .filter(|&&(other, _)| other == hash)
            .map(|&(_, position)| position)
            .find(|&position| is_equal(position))
    }

    /// Adds `position` with `hash`.
    pub(crate) fn insert(&mut self, hash: u64, position: usize) {
        match self.first.entry(hash) {
            Entry::Vacant(entry) => {
                entry.insert(position);
            }
            Entry::Occupied(_) => self.others.push((hash, position)),
        }
    }

    /// Removes every position.
    pub(crate) fn clear(&mut self) {
        self.first.clear();
        self.others.clear();
    }
}

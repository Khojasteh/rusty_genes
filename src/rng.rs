use rand::{
    Rng as _, RngExt, SeedableRng,
    distr::{Distribution, Uniform},
    rngs::Xoshiro256PlusPlus,
    seq::{IndexedRandom, SliceRandom},
};
use std::ops::{Range, RangeInclusive};

/// The random number generator of the evolutionary algorithms and their operators.
///
/// The generator is not cryptographically secure. Two generators created from the same
/// seed produce the same sequence of values with the same version of this library; the sequence
/// may change between versions.
///
/// With the `rand-0_10` feature, the generator also implements the random number generator traits
/// of `rand` 0.10, so it can be used with the distributions and other functions of that crate.
///
/// # Example
/// With the `rand-0_10` feature, the generator shuffles a slice and samples a value through `rand`.
/// ```
/// # #[cfg(feature = "rand-0_10")]
/// # {
/// use rand::{RngExt as _, seq::SliceRandom};
/// use rusty_genes::Rng;
///
/// let mut rng = Rng::from_seed(3);
/// let mut values = (0..10).collect::<Vec<_>>();
/// values.shuffle(&mut rng);
/// values.sort_unstable();
///
/// assert_eq!(values, (0..10).collect::<Vec<_>>());
/// assert!((0.0..1.0).contains(&rng.random::<f64>()));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct Rng(Xoshiro256PlusPlus);

impl Rng {
    /// Creates a generator from a seed.
    ///
    /// # Example
    /// ```
    /// use rusty_genes::Rng;
    ///
    /// let mut a = Rng::from_seed(42);
    /// let mut b = Rng::from_seed(42);
    ///
    /// assert!((0..10).all(|_| a.next_u64() == b.next_u64()));
    /// ```
    pub fn from_seed(seed: u64) -> Self {
        Self(Xoshiro256PlusPlus::seed_from_u64(seed))
    }

    /// Creates a generator seeded from the operating system's source of randomness.
    pub fn from_entropy() -> Self {
        Self(rand::make_rng())
    }

    /// Returns a set of independent generators for parallel work, which are determined by this
    /// generator's state.
    ///
    /// Taking the generator of each unit of work by its index from the returned [`RngStreams`]
    /// makes the result of parallel work independent of how the work is scheduled.
    ///
    /// # Example
    /// The generator of each index depends only on the state of the split generator and the
    /// index.
    /// ```
    /// use rusty_genes::Rng;
    ///
    /// let streams = Rng::from_seed(7).split();
    /// let again = Rng::from_seed(7).split();
    ///
    /// assert_eq!(streams.rng(3).next_u64(), again.rng(3).next_u64());
    /// assert_ne!(streams.rng(3).next_u64(), streams.rng(4).next_u64());
    /// ```
    pub fn split(&mut self) -> RngStreams {
        RngStreams {
            seed: self.next_u64(),
        }
    }

    /// Returns a random `u64` value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    /// Returns a random `f64` value between 0.0 (inclusive) and 1.0 (exclusive).
    #[inline]
    pub fn random_f64(&mut self) -> f64 {
        self.0.random()
    }

    /// Returns [`true`] with the probability `p`.
    ///
    /// A probability of 0.0 or less, or NaN, never returns [`true`], and a probability of 1.0 or
    /// more always does.
    ///
    /// # Example
    /// ```
    /// use rusty_genes::Rng;
    ///
    /// let mut rng = Rng::from_seed(0);
    ///
    /// assert!((0..100).all(|_| !rng.random_bool(0.0)));
    /// assert!((0..100).all(|_| !rng.random_bool(f64::NAN)));
    /// assert!((0..100).all(|_| rng.random_bool(1.0)));
    /// ```
    #[inline]
    pub fn random_bool(&mut self, p: f64) -> bool {
        self.random_f64() < p
    }

    /// Returns a value uniformly distributed in `range`.
    ///
    /// # Panics
    /// Panics if `range` is empty.
    #[inline]
    pub fn random_range(&mut self, range: Range<usize>) -> usize {
        self.0.random_range(range)
    }

    /// Samples inclusive index bounds without requiring a representable exclusive end.
    #[inline]
    pub(crate) fn random_range_inclusive(&mut self, range: RangeInclusive<usize>) -> usize {
        self.0.random_range(range)
    }

    /// Shuffles the elements of `slice` in place, so that every order is equally likely.
    #[inline]
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        slice.shuffle(&mut self.0);
    }

    /// Returns a uniformly chosen element of `slice`, or [`None`] if it is empty.
    #[inline]
    pub fn choose<'a, T>(&mut self, slice: &'a [T]) -> Option<&'a T> {
        slice.choose(&mut self.0)
    }
}

/// Reuses a configured uniform index distribution across repeated draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UniformIndices(Uniform<usize>);

impl UniformIndices {
    /// Returns a sampler for `range`, or [`None`] if `range` is empty.
    pub(crate) fn new(range: Range<usize>) -> Option<Self> {
        Uniform::new(range.start, range.end).ok().map(Self)
    }

    #[inline]
    pub(crate) fn sample(&self, rng: &mut Rng) -> usize {
        self.0.sample(&mut rng.0)
    }
}

/// A set of independent random number generators, identified by index, for parallel work.
///
/// It is created by [`Rng::split`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RngStreams {
    seed: u64,
}

impl RngStreams {
    /// Returns the generator with the given index. The same index always returns a generator
    /// with the same state.
    #[inline]
    pub fn rng(&self, index: u64) -> Rng {
        // Seeds from consecutive indices would share most of their state, so the index is
        // mixed into the seed first.
        Rng::from_seed(split_mix_64(self.seed ^ split_mix_64(index)))
    }
}

/// The output function of the SplitMix64 generator, which scrambles the bits of `x`.
#[inline]
fn split_mix_64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(feature = "rand-0_10")]
impl rand_core_0_10::TryRng for Rng {
    type Error = rand_core_0_10::Infallible;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.0.next_u32())
    }

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.0.next_u64())
    }

    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        self.0.fill_bytes(dst);
        Ok(())
    }
}

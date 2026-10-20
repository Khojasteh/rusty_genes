/// A streaming summary of sample extrema, mean, and sample variance, without retaining samples.
///
/// Extrema and moments return [`None`] until the first sample. Extrema indices refer to
/// recording order and keep the first occurrence of a tie. Variance divides by `count - 1`;
/// a single sample has variance and standard deviation zero. Non-finite samples are not
/// discarded; use finite inputs for meaningful statistics.
///
/// This struct uses Welford's online algorithm with power-of-two scaling to avoid overflow in
/// intermediate calculations for finite samples. Variance can still overflow or underflow if
/// its result cannot be represented by `f64`; standard deviation is computed independently of
/// that rounding, so it can remain representable in those cases.
///
/// # Example
/// ```
/// use rusty_genes::Stats;
///
/// let stats = [1.0, 2.0, 3.0, 4.0, 5.0].into_iter().collect::<Stats>();
///
/// assert_eq!(stats.count(), 5);
/// assert_eq!(stats.min(), Some(1.0));
/// assert_eq!(stats.arg_min(), Some(0));
/// assert_eq!(stats.max(), Some(5.0));
/// assert_eq!(stats.arg_max(), Some(4));
/// assert_eq!(stats.avg(), Some(3.0));
/// assert_eq!(stats.var(), Some(2.5));
/// assert!((stats.std().unwrap() - 1.58113883).abs() < 1e-6);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Stats {
    count: usize,
    min: f64,
    arg_min: usize,
    max: f64,
    arg_max: usize,
    scaled_mean: f64,
    scaled_m2: f64,
    // scaled_mean is in units of scale; scaled_m2 is in units of scale squared.
    // scale is always a positive, normal power of two.
    scale: f64,
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            count: 0,
            min: 0.0,
            arg_min: 0,
            max: 0.0,
            arg_max: 0,
            scaled_mean: 0.0,
            scaled_m2: 0.0,
            scale: f64::MIN_POSITIVE,
        }
    }
}

impl Stats {
    /// Returns the total number of samples.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the smallest sample value.
    #[inline]
    pub fn min(&self) -> Option<f64> {
        match self.count {
            0 => None,
            _ => Some(self.min),
        }
    }

    /// Returns the zero-based index of the smallest sample value.
    #[inline]
    pub fn arg_min(&self) -> Option<usize> {
        match self.count {
            0 => None,
            _ => Some(self.arg_min),
        }
    }

    /// Returns the largest sample value.
    #[inline]
    pub fn max(&self) -> Option<f64> {
        match self.count {
            0 => None,
            _ => Some(self.max),
        }
    }

    /// Returns the zero-based index of the largest sample value.
    #[inline]
    pub fn arg_max(&self) -> Option<usize> {
        match self.count {
            0 => None,
            _ => Some(self.arg_max),
        }
    }

    /// Returns the average (mean) of the sample values.
    #[inline]
    pub fn avg(&self) -> Option<f64> {
        match self.count {
            0 => None,
            _ => Some(self.scaled_mean * self.scale),
        }
    }

    /// Returns sample variance using `count - 1`, zero for one sample, or [`None`] when empty.
    #[inline]
    pub fn var(&self) -> Option<f64> {
        match self.count {
            0 => None,
            1 => Some(0.0),
            _ => Some((self.scaled_variance() * self.scale) * self.scale),
        }
    }

    /// Returns sample standard deviation, zero for one sample, or [`None`] when empty.
    ///
    /// It is computed independently of [`var`](Self::var), whose rounding may overflow or
    /// underflow even when the standard deviation is representable.
    #[inline]
    pub fn std(&self) -> Option<f64> {
        match self.count {
            0 => None,
            _ => Some(self.scaled_variance().sqrt() * self.scale),
        }
    }

    /// Returns the power of two by which `sample` is scaled.
    #[inline]
    fn scale_of(sample: f64) -> f64 {
        // Clearing the sign and significand gives a power of two. Zero and subnormal samples use
        // the smallest normal scale so their squared normalized values remain representable.
        f64::from_bits(sample.to_bits() & 0x7ff0_0000_0000_0000).max(f64::MIN_POSITIVE)
    }

    #[inline]
    fn scaled_variance(&self) -> f64 {
        if self.count > 1 {
            self.scaled_m2 / (self.count - 1) as f64
        } else {
            0.0
        }
    }

    /// Records a sample, assigning it the next zero-based index and updating the summary.
    ///
    /// # Example
    /// ```
    /// use rusty_genes::Stats;
    ///
    /// let mut stats = Stats::default();
    /// stats.record(2.0);
    /// stats.record(5.0);
    /// stats.record(1.0);
    /// stats.record(4.0);
    /// stats.record(3.0);
    ///
    /// assert_eq!(stats.count(), 5);
    /// assert_eq!(stats.min(), Some(1.0));
    /// assert_eq!(stats.arg_min(), Some(2));
    /// assert_eq!(stats.max(), Some(5.0));
    /// assert_eq!(stats.arg_max(), Some(1));
    /// assert_eq!(stats.avg(), Some(3.0));
    /// assert_eq!(stats.var(), Some(2.5));
    /// assert!((stats.std().unwrap() - 1.58113883).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn record(&mut self, sample: f64) {
        if sample.is_finite() && sample != 0.0 {
            let scale = Self::scale_of(sample);
            if scale > self.scale {
                let ratio = self.scale / scale;
                self.scaled_mean *= ratio;
                self.scaled_m2 *= ratio * ratio;
                self.scale = scale;
            }
        }
        let scaled_sample = sample / self.scale;
        if self.count != 0 {
            if sample < self.min {
                self.min = sample;
                self.arg_min = self.count;
            }
            if sample > self.max {
                self.max = sample;
                self.arg_max = self.count;
            }

            // Welford's online algorithm
            self.count += 1;
            let delta = scaled_sample - self.scaled_mean;
            self.scaled_mean += delta / self.count as f64;
            self.scaled_m2 += delta * (scaled_sample - self.scaled_mean);
        } else {
            self.count = 1;
            self.min = sample;
            self.arg_min = 0;
            self.max = sample;
            self.arg_max = 0;
            self.scaled_mean = scaled_sample;
            self.scaled_m2 = 0.0;
        }
    }

    /// Copies the current summary, or returns [`None`] before any sample has been recorded.
    ///
    /// Later calls to [`record`](Self::record) do not change the snapshot.
    ///
    /// # Example
    /// ```
    /// use rusty_genes::Stats;
    ///
    /// let mut samples = Stats::default();
    /// samples.record(3.0);
    /// samples.record(5.0);
    ///
    /// let s1 = samples.snapshot().unwrap();
    ///
    /// samples.record(2.0);
    /// samples.record(4.0);
    ///
    /// let s2 = samples.snapshot().unwrap();
    ///
    /// samples.record(1.0);
    ///
    /// assert_eq!(
    ///     format!("{s1:.2}"),
    ///     "Samples: 2; Min: 3.00; Max: 5.00; Average: 4.00; Variance: 2.00; STD: 1.41"
    /// );
    /// assert_eq!(
    ///     format!("{s2:.2}"),
    ///     "Samples: 4; Min: 2.00; Max: 5.00; Average: 3.50; Variance: 1.67; STD: 1.29"
    /// );
    /// assert_eq!(
    ///     format!("{samples:.2}"),
    ///     "Samples: 5; Min: 1.00; Max: 5.00; Average: 3.00; Variance: 2.50; STD: 1.58"
    /// );
    /// ```
    pub fn snapshot(&self) -> Option<PopulatedStats> {
        if self.count != 0 {
            Some(PopulatedStats { stats: *self })
        } else {
            None
        }
    }
}

impl FromIterator<f64> for Stats {
    fn from_iter<I: IntoIterator<Item = f64>>(iter: I) -> Self {
        let mut stats = Self::default();
        for sample in iter {
            stats.record(sample);
        }
        stats
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.snapshot().map(|s| s.fmt(f)).unwrap_or_else(|| Ok(()))
    }
}

/// A snapshot of a non-empty [`Stats`] summary, with values available without [`Option`].
///
/// [`Stats::snapshot`] and conversion from [`Stats`] require at least one sample. Subsequent
/// recording in the original summary does not change this copy. Variance uses `count - 1`,
/// with zero variance and standard deviation for one sample.
///
/// # Example
/// ```
/// use rusty_genes::{Stats, PopulatedStats};
///
/// let stats: PopulatedStats = [1.0, 2.0, 3.0, 4.0, 5.0]
///     .into_iter()
///     .collect::<Stats>()
///     .try_into()
///     .unwrap();
///
/// assert_eq!(stats.count(), 5);
/// assert_eq!(stats.min(), 1.0);
/// assert_eq!(stats.arg_min(), 0);
/// assert_eq!(stats.max(), 5.0);
/// assert_eq!(stats.arg_max(), 4);
/// assert_eq!(stats.avg(), 3.0);
/// assert_eq!(stats.var(), 2.5);
/// assert!((stats.std() - 1.58113883).abs() < 1e-6);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct PopulatedStats {
    stats: Stats,
}

impl PopulatedStats {
    /// Summarizes the finite scores of `items`, which are sorted by score in ascending or
    /// descending order, in two passes over them.
    ///
    /// The extrema and their indices are those [`Stats::record`] finds, because they lie at the
    /// ends of the order. Knowing the largest magnitude in advance, the scores are scaled once,
    /// and their variance is accumulated from deviations from their mean, with the correction of
    /// the corrected two-pass algorithm.
    pub(crate) fn of_sorted<T>(items: &[T], score: impl Fn(&T) -> f64) -> Option<Self> {
        let (first, last) = (score(items.first()?), score(items.last()?));
        // Recording keeps the first of equal extrema, which begins the run of scores equal to
        // the last one.
        let run = items.partition_point(|item| score(item) != last);
        let ((min, arg_min), (max, arg_max)) = if first <= last {
            ((first, 0), (score(&items[run]), run))
        } else {
            ((score(&items[run]), run), (first, 0))
        };

        let count = items.len();
        let scale = Stats::scale_of(min).max(Stats::scale_of(max));
        let scaled_mean = items.iter().map(|item| score(item) / scale).sum::<f64>() / count as f64;
        let (sum, sum_of_squares) = items.iter().fold((0.0, 0.0), |(sum, squares), item| {
            let deviation = score(item) / scale - scaled_mean;
            (sum + deviation, squares + deviation * deviation)
        });
        let scaled_m2 = (sum_of_squares - sum * sum / count as f64).max(0.0);

        Some(Self {
            stats: Stats {
                count,
                min,
                arg_min,
                max,
                arg_max,
                scaled_mean,
                scaled_m2,
                scale,
            },
        })
    }

    /// Returns the total number of samples.
    #[inline]
    pub fn count(&self) -> usize {
        self.stats.count
    }

    /// Returns the smallest sample value.
    #[inline]
    pub fn min(&self) -> f64 {
        self.stats.min
    }

    /// Returns the zero-based index of the smallest sample value.
    #[inline]
    pub fn arg_min(&self) -> usize {
        self.stats.arg_min
    }

    /// Returns the largest sample value.
    #[inline]
    pub fn max(&self) -> f64 {
        self.stats.max
    }

    /// Returns the zero-based index of the largest sample value.
    #[inline]
    pub fn arg_max(&self) -> usize {
        self.stats.arg_max
    }

    /// Returns the average (mean) of the sample values.
    #[inline]
    pub fn avg(&self) -> f64 {
        self.stats.avg().unwrap()
    }

    /// Returns sample variance using `count - 1`, or zero for one sample.
    #[inline]
    pub fn var(&self) -> f64 {
        self.stats.var().unwrap()
    }

    /// Returns sample standard deviation, or zero for one sample.
    #[inline]
    pub fn std(&self) -> f64 {
        self.stats.std().unwrap()
    }

    /// Remaps the minimum's index after the caller reorders the original samples.
    ///
    /// Only the index changes; the caller must ensure it still identifies the minimum.
    ///
    /// # Panics
    /// Panics if `arg_min` is out of bounds.
    ///
    #[inline]
    pub fn set_arg_min(&mut self, arg_min: usize) {
        assert!(arg_min < self.count());
        self.stats.arg_min = arg_min;
    }

    /// Remaps the maximum's index after the caller reorders the original samples.
    ///
    /// Only the index changes; the caller must ensure it still identifies the maximum.
    ///
    /// # Panics
    /// Panics if `arg_max` is out of bounds.
    ///
    #[inline]
    pub fn set_arg_max(&mut self, arg_max: usize) {
        assert!(arg_max < self.count());
        self.stats.arg_max = arg_max;
    }
}

impl TryFrom<Stats> for PopulatedStats {
    type Error = ();

    #[inline]
    fn try_from(stats: Stats) -> Result<Self, Self::Error> {
        if stats.count != 0 {
            Ok(Self { stats })
        } else {
            Err(())
        }
    }
}

impl std::fmt::Display for PopulatedStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(p) = f.precision() {
            write!(
                f,
                "Samples: {}; Min: {:.p$}; Max: {:.p$}; Average: {:.p$}; Variance: {:.p$}; STD: {:.p$}",
                self.count(),
                self.min(),
                self.max(),
                self.avg(),
                self.var(),
                self.std()
            )
        } else {
            write!(
                f,
                "Samples: {}; Min: {}; Max: {}; Average: {}; Variance: {}; STD: {}",
                self.count(),
                self.min(),
                self.max(),
                self.avg(),
                self.var(),
                self.std()
            )
        }
    }
}

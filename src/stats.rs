/// The `Stats` struct represents a collection of statistical data derived from
/// a set of unsorted numerical samples.
///
/// It provides information such as:
/// - The number of samples (`count`)
/// - The minimum value (`min`)
/// - The zero-based index of the minimum value (`arg_min`)
/// - The maximum value (`max`)
/// - The zero-based index of the maximum value (`arg_max`)
/// - The average value (`avg`)
/// - The variance (`var`)
/// - The standard deviation (`std`)
///
/// This struct uses Welford's online algorithm to provide an efficient and
/// numerically stable calculation of the variance.
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
#[derive(Debug, Default, Copy, Clone)]
pub struct Stats {
    count: usize,
    min: f64,
    arg_min: usize,
    max: f64,
    arg_max: usize,
    avg: f64,
    sum_squared: f64,
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
            _ => Some(self.avg),
        }
    }

    /// Returns the variance of the sample values.
    #[inline]
    pub fn var(&self) -> Option<f64> {
        match self.count {
            0 => None,
            1 => Some(0.0),
            _ => Some(self.sum_squared / (self.count - 1) as f64),
        }
    }

    /// Returns the standard deviation of the sample values.
    #[inline]
    pub fn std(&self) -> Option<f64> {
        self.var().map(f64::sqrt)
    }

    /// Records a new sample and updates all statistical data accordingly.
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
            let delta = sample - self.avg;
            self.avg += delta / self.count as f64;
            self.sum_squared += delta * (sample - self.avg);
        } else {
            self.count = 1;
            self.min = sample;
            self.arg_min = 0;
            self.max = sample;
            self.arg_max = 0;
            self.avg = sample;
            self.sum_squared = 0.0;
        }
    }

    /// Returns a snapshot of the [`Stats`] object at its current state, if it is
    /// not empty.
    ///
    /// This can be useful to record the statistics at a certain point in time,
    /// while allowing the original [`Stats`] object to continue being updated with
    /// new samples.
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
        let mut samples = iter.into_iter();
        let Some (first_sample) = samples.next() else {
            return Default::default()
        };

        let mut count = 1;
        let mut min = first_sample;
        let mut arg_min = 0;
        let mut max = first_sample;
        let mut arg_max = 0;
        let mut avg = first_sample;
        let mut sum_squared = 0.0;

        for sample in samples {
            if sample < min {
                min = sample;
                arg_min = count;
            }
            if sample > max {
                max = sample;
                arg_max = count;
            }

            count += 1;
            let delta = sample - avg;
            avg += delta / count as f64;
            sum_squared += delta * (sample - avg);
        }

        Self {
            count,
            min,
            arg_min,
            max,
            arg_max,
            avg,
            sum_squared,
        }
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.snapshot().map(|s| s.fmt(f)).unwrap_or_else(|| Ok(()))
    }
}

/// The [`PopulatedStats`] struct represents a statistical summary of a non-empty
/// set of numerical samples. This structure is similar to the [`Stats`] struct,
/// but it contains additional constraints ensuring that the set of samples it
/// represents is not empty.
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
        self.stats.avg
    }

    /// Returns the variance of the sample values.
    #[inline]
    pub fn var(&self) -> f64 {
        if self.stats.count > 1 {
            self.stats.sum_squared / (self.stats.count - 1) as f64
        } else {
            0.0
        }
    }

    /// Returns the standard deviation of the sample values.
    #[inline]
    pub fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// Updates the `arg_min` value. This method is useful when the order of
    /// elements in the original sampled data is changed.
    ///
    /// # Arguments
    /// * `arg_min` - The new value for `arg_min`, which must be within the bounds
    ///   of the sampled data.
    ///
    /// # Panics
    /// This method will panic if `arg_min` is out of bounds.
    ///
    /// # Note
    /// This method should be called with caution to ensure that the updated `arg_min`
    /// value remains valid and consistent with the sampled data.
    #[inline]
    pub fn set_arg_min(&mut self, arg_min: usize) {
        assert!(arg_min < self.count());
        self.stats.arg_min = arg_min;
    }

    /// Updates the `arg_max` value.
    ///
    /// This method is useful when the order of elements in the original sampled
    /// data is changed.
    ///
    /// # Arguments
    /// * `arg_max` - The new value for `arg_max`, which must be within the bounds
    ///   of the sampled data.
    ///
    /// # Panics
    /// This method will panic if `arg_max` is out of bounds.
    ///
    /// # Note
    /// This method should be called with caution to ensure that the updated `arg_max`
    /// value remains valid and consistent with the sampled data.
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
            write!(f,
                "Samples: {}; Min: {:.p$}; Max: {:.p$}; Average: {:.p$}; Variance: {:.p$}; STD: {:.p$}",
                self.count(),
                self.min(),
                self.max(),
                self.avg(),
                self.var(),
                self.std())
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

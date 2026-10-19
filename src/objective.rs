use std::cmp::Ordering;

/// Whether lower or higher scores are better, without changing the scores themselves.
///
/// Models select their objective through [`EvolutionModel::objective`](crate::EvolutionModel::objective).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    /// Higher scores are better. This is the default for existing models.
    #[default]
    Maximize,
    /// Lower scores are better, for example when minimizing distance or cost.
    Minimize,
}

impl Objective {
    /// Returns whether `score` is fitter than `other`: higher for maximization, and lower for
    /// minimization.
    ///
    /// Equal numerical scores, including signed zeros, are equally fit.
    #[inline]
    pub fn is_fitter(self, score: f64, other: f64) -> bool {
        self.cmp_fitness(score, other).is_gt()
    }

    /// Returns whether `score` is at least as fit as `other`: at least as high for
    /// maximization, and at most as high for minimization.
    ///
    /// Equal numerical scores, including signed zeros, are equally fit.
    #[inline]
    pub fn is_at_least_as_fit(self, score: f64, other: f64) -> bool {
        self.cmp_fitness(score, other).is_ge()
    }

    /// Orders scores by fitness, so that [`Ordering::Greater`] means that `left` is fitter than
    /// `right`. Sorting scores by it orders them from the least fit to the fittest.
    ///
    /// Equal numerical scores, including signed zeros, are equally fit. Otherwise this follows
    /// [`f64::total_cmp`], reversed for minimization. Population scores must be finite.
    #[inline]
    pub fn cmp_fitness(self, left: f64, right: f64) -> Ordering {
        // `partial_cmp` agrees with `total_cmp` except that it treats signed zeros as equal and
        // has no order for NaN, which falls back to `total_cmp`.
        let ordering = left
            .partial_cmp(&right)
            .unwrap_or_else(|| left.total_cmp(&right));
        match self {
            Self::Maximize => ordering,
            Self::Minimize => ordering.reverse(),
        }
    }

    /// Returns `score` as a value to maximize, which is the score itself for maximization and
    /// its negation for minimization.
    #[inline]
    pub(crate) fn as_maximized(self, score: f64) -> f64 {
        match self {
            Self::Maximize => score,
            Self::Minimize => -score,
        }
    }
}

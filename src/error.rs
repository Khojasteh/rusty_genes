use std::fmt;

/// The error type for configuring and running an evolutionary algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EvolutionError {
    /// A parameter is outside of its valid domain.
    InvalidParameter {
        /// The name of the offending parameter.
        name: &'static str,
        /// The constraint the parameter violates.
        constraint: &'static str,
    },

    /// There is no valid individual to form a population from.
    ///
    /// This happens when a population is created from, or replaced by, a set of individuals
    /// in which no individual is valid according to [`Individual::is_valid`](crate::Individual::is_valid).
    EmptyPopulation,
}

impl EvolutionError {
    pub(crate) fn invalid_parameter(name: &'static str, constraint: &'static str) -> Self {
        Self::InvalidParameter { name, constraint }
    }
}

impl fmt::Display for EvolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter { name, constraint } => {
                write!(f, "invalid parameter `{name}`: {constraint}")
            }
            Self::EmptyPopulation => f.write_str("no valid individual to form a population"),
        }
    }
}

impl std::error::Error for EvolutionError {}

/// Returns an error unless `value` is a probability, i.e. a number between 0.0 and 1.0, inclusive.
pub(crate) fn check_probability(name: &'static str, value: f64) -> Result<(), EvolutionError> {
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(EvolutionError::invalid_parameter(
            name,
            "must be between 0.0 and 1.0, inclusive",
        ))
    }
}

/// Returns an error unless `value` is a ratio between 0.0 (inclusive) and 1.0 (exclusive).
pub(crate) fn check_ratio(name: &'static str, value: f64) -> Result<(), EvolutionError> {
    if (0.0..1.0).contains(&value) {
        Ok(())
    } else {
        Err(EvolutionError::invalid_parameter(
            name,
            "must be between 0.0 (inclusive) and 1.0 (exclusive)",
        ))
    }
}

// Curve fitting: using a genetic algorithm to estimate four real-valued parameters.
//
// The measurements follow a damped oscillation, `y = a * exp(-b * x) * cos(c * x + d)`, with some
// noise. Each individual is one proposed set of amplitude, decay, frequency, and phase values.
// Together these values form its genome. Its score is the root mean square error (RMSE) at the
// measured points: lower scores mean a closer fit, and zero would mean an exact fit.
//
// These genomes are coordinates, not permutations, so the model defines operators for real
// numbers. Crossover blends the parents' values, and mutation moves one parameter a small step.
// Both keep the parameters inside the allowed bounds. A population holds several candidate fits.
//
// The diversity-guided genetic algorithm switches between crossover and mutation according to
// how spread out those fits are. Low diversity starts mutation; sufficiently high diversity
// resumes crossover. Between the thresholds it keeps its current mode. Its binary tournaments
// choose the better of two random candidates as a parent.
//
// The run stops at a generation limit, at an optional time limit, or when its best score stops
// improving. The parameters that generated the data are a reference, not a known optimum: noise
// can make another fit have a smaller error on these measurements. Run it with:
//
//     cargo run --release --example curve_fitting
//
// Try a larger dataset once you have followed the default run:
//
//     cargo run --release --example curve_fitting -- --points 10000 --population 500

use rusty_genes::*;
use std::{
    hash::{Hash, Hasher},
    ops::RangeInclusive,
    time::Duration,
};

/// The number of parameters of the curve: its amplitude, decay, frequency, and phase.
const PARAMETER_COUNT: usize = 4;

/// The names of the parameters, for the output.
const PARAMETER_NAMES: [&str; PARAMETER_COUNT] = ["amplitude", "decay", "frequency", "phase"];

/// The parameters that generate the measurements, which the fitted parameters are compared with.
const TRUE_PARAMETERS: [f64; PARAMETER_COUNT] = [3.0, 0.3, 2.0, 0.5];

/// Builds noisy measurements, searches for a fit, and compares it with the generating curve.
fn main() -> Result<(), EvolutionError> {
    let Some(Options {
        point_count,
        population_size,
        generation_limit,
        stagnation_limit,
        time_limit,
        seed,
    }) = Options::parse()
    else {
        return Ok(());
    };

    let measurements = sample_measurements(point_count);
    // The four intervals constrain the parameters that random creation and breeding may use.
    let bounds: Bounds = [0.0..=5.0, 0.0..=1.0, 0.5..=4.0, -3.2..=3.2];
    let model = CurveFittingModel {
        measurements,
        bounds: bounds.clone(),
    };
    // Score the generating parameters as a reference; noise means their RMSE need not be zero.
    let mut generating_fit = Fit::from(TRUE_PARAMETERS);
    model.evaluate(&mut generating_fit);

    // Diversity describes how spread out the candidate fits are. When they crowd together,
    // this algorithm switches to mutation to explore; once they spread out again, it resumes
    // crossover. The separate thresholds prevent a small fluctuation from switching it back.
    // Its default binary tournaments pick the better of two random candidates as a parent.
    let mut parameters = DiversityGuidedGeneticAlgorithmParameters::default();
    parameters.population_size = population_size;
    let mut evolution = DiversityGuidedGeneticAlgorithm::new(model, parameters)?
        .into_evolution()
        .with_seed(seed);

    // An exact fit is not expected for noisy data. Stagnation counts generations since the
    // best error of the entire run last decreased. This limits effort without improvement,
    // without proving that the best fit has been found. Also cap generations, and optionally time.
    // Time is checked between generations, so a generation can run past the budget.
    let mut limits = Limits::new()
        .generations(generation_limit)
        .stagnation(stagnation_limit);
    if let Some(time_limit) = time_limit {
        limits = limits.time(time_limit);
    }
    // `run` samples starting parameters inside the bounds and evaluates them as generation
    // zero. The callback reports progress and decides whether another generation may be bred.
    let outcome = evolution.run(&bounds, |state| {
        if state.generation() % 100 == 0 {
            println!(
                "Generation {:4}: smallest error {:.4}",
                state.generation(),
                state.fittest_so_far().score()
            );
        }
        limits.check(state)
    })?;

    // Report the best fit from any generation. Compare errors as well as parameters, since
    // fitting the noisy measurements does not require recovering the generating values exactly.
    let state = outcome.state();
    let best = state.fittest_so_far();
    println!("Smallest error: {:.4}", best.score());
    println!(
        "Error of the generating parameters: {:.4}",
        generating_fit.score()
    );
    for ((name, fitted), actual) in PARAMETER_NAMES
        .iter()
        .zip(best.parameters)
        .zip(TRUE_PARAMETERS)
    {
        println!("{name:>9}: {fitted:7.4} (generated with {actual})");
    }
    println!(
        "{} generations in {:.3} seconds ({})",
        state.generation(),
        state.elapsed().as_secs_f64(),
        outcome.reason()
    );
    Ok(())
}

/// Evaluates the curve at `x`, using amplitude `a`, decay `b`, frequency `c`, and phase `d`.
///
/// Decay reduces the amplitude as `x` increases; frequency and phase set the cosine's argument.
fn curve([a, b, c, d]: &[f64; PARAMETER_COUNT], x: f64) -> f64 {
    a * (-b * x).exp() * (c * x + d).cos()
}

/// One candidate fit, with four parameters and their cached error on the measurements.
#[derive(Debug, Clone)]
struct Fit {
    /// Amplitude, decay, frequency, and phase, in that order.
    parameters: [f64; PARAMETER_COUNT],
    /// The cached RMSE, or NaN until these parameters have been evaluated.
    error: f64,
}

impl PartialEq for Fit {
    /// Compares the parameters exactly, ignoring the cached error when finding duplicate fits.
    fn eq(&self, other: &Self) -> bool {
        self.parameters == other.parameters
    }
}

impl Hash for Fit {
    /// Hashes the same parameters equality compares, treating positive and negative zero alike.
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_f64_slice(&self.parameters, state);
    }
}

impl From<[f64; PARAMETER_COUNT]> for Fit {
    /// Wraps parameters with an invalid score; the engine evaluates them before admission.
    fn from(parameters: [f64; PARAMETER_COUNT]) -> Self {
        Self {
            parameters,
            error: f64::NAN,
        }
    }
}

impl Individual for Fit {
    /// Returns the cached RMSE, in the same units as the measured `y` values.
    fn score(&self) -> f64 {
        self.error
    }
}

/// Allowed intervals for amplitude, decay, frequency, and phase, in genome order.
type Bounds = [RangeInclusive<f64>; PARAMETER_COUNT];

/// Limits a parameter to its allowed interval after crossover or mutation.
fn clamp(value: f64, range: &RangeInclusive<f64>) -> f64 {
    value.clamp(*range.start(), *range.end())
}

/// Interpolates across `range` using a uniformly sampled fraction in `[0, 1)`.
fn random_in(range: &RangeInclusive<f64>, rng: &mut Rng) -> f64 {
    range.start() + rng.random_f64() * (range.end() - range.start())
}

impl RandomIndividual<Bounds> for Fit {
    /// Creates a starting fit by drawing each parameter independently within its interval.
    fn new_random(bounds: &Bounds, rng: &mut Rng) -> Self {
        Self::from(std::array::from_fn(|i| random_in(&bounds[i], rng)))
    }
}

/// The measurements, search bounds, and rules for breeding and scoring candidate fits.
struct CurveFittingModel {
    /// The measured points, as `(x, y)` pairs.
    measurements: Vec<(f64, f64)>,
    /// The allowed search space, also used to normalize the population's diversity.
    bounds: Bounds,
}

impl EvolutionModel for CurveFittingModel {
    /// This model breeds and scores real-valued parameter sets.
    type Individual = Fit;

    /// Makes a smaller fitting error a better score without changing the error's sign.
    fn objective(&self) -> Objective {
        Objective::Minimize
    }

    /// Picks each parameter uniformly from the range between the parents' values, extended by
    /// half its width on both sides, and limited to the parameter's bounds.
    fn crossover(&self, parent1: &Fit, parent2: &Fit, rng: &mut Rng) -> Fit {
        Fit::from(std::array::from_fn(|i| {
            let (a, b) = (parent1.parameters[i], parent2.parameters[i]);
            // Allow values just outside the parents' interval to explore nearby alternatives.
            let extension = 0.5 * (a - b).abs();
            let blend = a.min(b) - extension..=a.max(b) + extension;
            // Clamping keeps that exploration inside the problem's search space.
            clamp(random_in(&blend, rng), &self.bounds[i])
        }))
    }

    /// Moves one randomly chosen parameter by up to a tenth of its range, within its bounds.
    fn mutate(&self, fit: &mut Fit, rng: &mut Rng) {
        // Vary one coordinate at a time, with a step scaled to that parameter's own range.
        let i = rng.random_range(0..PARAMETER_COUNT);
        let range = &self.bounds[i];
        let step = 0.1 * (range.end() - range.start()) * (2.0 * rng.random_f64() - 1.0);
        fit.parameters[i] = clamp(fit.parameters[i] + step, range);
        // The cached error belongs to the old parameters and must be recomputed.
        fit.error = f64::NAN;
    }

    /// Caches the RMSE of the curve at every measured point; the algorithm minimizes it.
    fn evaluate(&self, fit: &mut Fit) {
        // Squaring penalizes large deviations and prevents positive and negative ones canceling.
        let squared_error = self
            .measurements
            .iter()
            .map(|&(x, y)| (curve(&fit.parameters, x) - y).powi(2))
            .sum::<f64>();
        // Averaging and taking the square root returns the error to the measured y units.
        fit.error = (squared_error / self.measurements.len() as f64).sqrt();
    }
}

impl DiversityModel for CurveFittingModel {
    /// Measures how far the parameter sets lie from their average, relative to the search space.
    fn diversity(&self, population: &Population<Fit>) -> f64 {
        // The diagonal combines the widths of the four parameter intervals into one length.
        let diagonal = self
            .bounds
            .iter()
            .map(|range| (range.end() - range.start()).powi(2))
            .sum::<f64>()
            .sqrt();
        // Each fit is a point in four dimensions. A tight cluster has low diversity, which
        // tells the algorithm to switch to mutation when it crosses the low threshold.
        distance_to_average_point(population.iter().map(|fit| fit.parameters), diagonal)
    }
}

/// Samples the generating curve at evenly spaced points and adds uniform noise in [-0.2, 0.2).
/// A separate fixed seed keeps the data unchanged when the search's --seed changes.
fn sample_measurements(point_count: usize) -> Vec<(f64, f64)> {
    let mut rng = Rng::from_seed(7);
    (0..point_count)
        .map(|i| {
            let x = 10.0 * i as f64 / point_count as f64;
            let noise = 0.4 * rng.random_f64() - 0.2;
            (x, curve(&TRUE_PARAMETERS, x) + noise)
        })
        .collect()
}

/// Dataset size and search budget supplied on the command line.
struct Options {
    point_count: usize,
    population_size: usize,
    generation_limit: usize,
    stagnation_limit: usize,
    time_limit: Option<Duration>,
    seed: u64,
}

impl Options {
    /// Reads overrides for the defaults, or returns None after printing --help.
    fn parse() -> Option<Self> {
        let mut options = Self {
            point_count: 60,
            population_size: 100,
            generation_limit: 2000,
            stagnation_limit: 100,
            time_limit: None,
            seed: 42,
        };
        let mut args = std::env::args().skip(1);
        while let Some(option) = args.next() {
            match option.as_str() {
                "--points" => options.point_count = option_value(&option, args.next()),
                "--population" => options.population_size = option_value(&option, args.next()),
                "--generations" => options.generation_limit = option_value(&option, args.next()),
                "--stagnation" => options.stagnation_limit = option_value(&option, args.next()),
                "--time" => options.time_limit = Some(seconds_value(&option, args.next())),
                "--seed" => options.seed = option_value(&option, args.next()),
                "--help" => {
                    println!("{USAGE}");
                    return None;
                }
                _ => exit_with_usage(&format!("unknown option {option}")),
            }
        }
        if options.point_count == 0 {
            exit_with_usage("there must be at least one measurement");
        }
        Some(options)
    }
}

/// The usage of the example's options.
const USAGE: &str = "Options:
  --points <N>       The number of measurements to fit [default: 60]
  --population <N>   The number of individuals in a generation [default: 100]
  --generations <N>  The most generations to run [default: 2000]
  --stagnation <N>   The generations without improvement that stop the run [default: 100]
  --time <SECONDS>   The most seconds to run, which makes the result depend on the machine
                     [default: no limit]
  --seed <N>         The seed of the random number generator [default: 42]";

/// Parses an option value, reporting usage and exiting with status 2 if it is missing or invalid.
fn option_value<T: std::str::FromStr>(option: &str, value: Option<String>) -> T {
    value
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| exit_with_usage(&format!("{option} needs a valid value")))
}

/// Parses a number of seconds, reporting usage and exiting with status 2 if it is missing or is
/// not a valid duration.
fn seconds_value(option: &str, value: Option<String>) -> Duration {
    Duration::try_from_secs_f64(option_value(option, value))
        .unwrap_or_else(|_| exit_with_usage(&format!("{option} needs a valid number of seconds")))
}

/// Reports an input error with the available options, then exits with status 2.
fn exit_with_usage(error: &str) -> ! {
    eprintln!("{error}\n\n{USAGE}");
    std::process::exit(2)
}

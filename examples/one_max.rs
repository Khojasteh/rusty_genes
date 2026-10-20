// OneMax: learning a genetic algorithm by evolving a string of bits towards all ones.
//
// A candidate solution is an individual, and its bit string is its genome. The score counts
// the ones: higher is better, and a string of n bits has the known optimum n. A population is a
// collection of these candidates that the algorithm replaces generation by generation.
//
// The individual defines how to create a random bit string; the model defines how to combine,
// change, and score bit strings. The standard genetic algorithm chooses parents by tournaments
// of three: draw three candidates and take the best.
// Crossover joins parts of two parents; mutation flips a bit, so it can introduce ones that
// neither parent has. Their probabilities stay fixed during the run.
//
// Evolution starts with random bit strings, evaluates them, and breeds new generations. The
// model supplies the optimum as a goal; a generation limit, and an optional time limit, also
// stop unsuccessful searches.
// The returned solution is the best found across all generations. Run it with:
//
//     cargo run --example one_max
//
// Try a larger problem once you have followed the default run:
//
//     cargo run --release --example one_max -- --bits 10000 --population 500

use rusty_genes::*;
use std::{
    hash::{Hash, Hasher},
    time::Duration,
};

/// Configures the search, evolves random bit strings, and reports the best score found.
fn main() -> Result<(), EvolutionError> {
    let Some(Options {
        genome_length,
        population_size,
        generation_limit,
        time_limit,
        seed,
    }) = Options::parse()
    else {
        return Ok(());
    };

    // The population explores many solutions at once. The defaults use tournament selection
    // (pick the best of three random candidates), 80% crossover, and 10% mutation. These
    // probabilities apply to a whole candidate, not to each bit; the model defines the changes.
    let mut parameters = GeneticAlgorithmParameters::default();
    parameters.population_size = population_size;
    let mut evolution = GeneticAlgorithm::new(OneMaxEvolutionModel, parameters)?
        .into_evolution()
        .with_seed(seed);

    // Generation zero is the random, evaluated starting population. Limits::check lets the
    // next generation be bred until a budget is reached. The model's all-ones goal can stop
    // the search sooner, before the callback sees that generation.
    // A time limit is checked between generations, so it may be exceeded by one generation.
    let mut limits = Limits::new().generations(generation_limit);
    if let Some(time_limit) = time_limit {
        limits = limits.time(time_limit);
    }
    let outcome = evolution.run(&genome_length, |state| limits.check(state))?;

    // The outcome keeps the best solution from the entire run, even if a later generation
    // loses it. Compare its count of ones with the known optimum, the genome length.
    let state = outcome.state();
    println!(
        "Best score after {} generations: {} of {genome_length}",
        state.generation(),
        state.fittest_so_far().score()
    );
    println!("Stopped: {}", outcome.reason());
    println!("Elapsed: {:.3} seconds", state.elapsed().as_secs_f64());
    Ok(())
}

/// An individual of the OneMax problem.
///
/// The genome is a binary string represented as a vector of booleans, and the score of an
/// evaluated individual is the number of `true` bits. An optimal genome of length `n` has a score
/// of `n`.
///
/// Equality and hashing use only the genome, so an individual equals its evaluated copy, and a
/// population keeps one copy of each genome.
#[derive(Debug, Clone)]
pub struct OneMax {
    /// The candidate solution; each position can be either zero (`false`) or one (`true`).
    genome: Vec<bool>,
    /// The cached number of ones, or NaN until the genome has been evaluated.
    score: f64,
}

impl PartialEq for OneMax {
    /// Compares solutions by their bits, so evaluating a genome does not change its identity.
    fn eq(&self, other: &Self) -> bool {
        self.genome == other.genome
    }
}

impl Eq for OneMax {}

impl Hash for OneMax {
    /// Hashes the same bits equality compares, allowing a population to find duplicate genomes.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.genome.hash(state);
    }
}

impl OneMax {
    /// Returns the binary string of the individual.
    pub fn genome(&self) -> &[bool] {
        &self.genome
    }
}

impl From<Vec<bool>> for OneMax {
    /// Wraps a bit string with an invalid score; the engine evaluates it before admission.
    fn from(genome: Vec<bool>) -> Self {
        Self {
            genome,
            score: f64::NAN,
        }
    }
}

impl Individual for OneMax {
    /// Returns the cached number of ones, in the problem's own units.
    fn score(&self) -> f64 {
        self.score
    }

    /// A valid individual has a non-empty genome and a finite score.
    fn is_valid(&self) -> bool {
        !self.genome.is_empty() && self.score.is_finite()
    }
}

impl RandomIndividual<usize> for OneMax {
    /// Creates a starting candidate with `length` independently chosen bits.
    fn new_random(length: &usize, rng: &mut Rng) -> Self {
        // Equal chances of zero and one give the initial population a variety of bit strings.
        let genome = (0..*length)
            .map(|_| rng.random_bool(0.5))
            .collect::<Vec<_>>();
        Self::from(genome)
    }
}

/// The problem's operators and scoring rule, shared by every individual in the run.
///
/// The default objective maximizes the score. Parent selection, survival, and the probabilities
/// of applying crossover and mutation belong to the algorithm rather than to this model.
/// By default, a child survives if it is at least as fit as its fitter parent. Elitism also
/// carries a small number of the best individuals unchanged into the next generation.
#[derive(Debug, Default)]
pub struct OneMaxEvolutionModel;

impl EvolutionModel for OneMaxEvolutionModel {
    /// This model breeds and scores the bit-string representation defined above.
    type Individual = OneMax;

    /// The goal is to set every bit, so it equals the genome length of the run. All individuals
    /// in a run must have the same genome length.
    fn goal_score(&self, initial_population: &Population<OneMax>) -> Option<f64> {
        Some(initial_population.fittest().genome.len() as f64)
    }

    /// Joins a prefix of the first parent's bits to a suffix of the second parent's bits.
    ///
    /// The parents must have the same length. Converting the result to `OneMax` leaves its
    /// score invalid until the engine evaluates it.
    fn crossover(&self, parent1: &OneMax, parent2: &OneMax, rng: &mut Rng) -> OneMax {
        CrossoverStrategy::SinglePoint
            .crossover(&parent1.genome, &parent2.genome, rng)
            .into()
    }

    /// Flips one randomly chosen bit, allowing new set bits even in an all-zero genome. The
    /// individual remains invalid until it is evaluated again.
    fn mutate(&self, individual: &mut OneMax, rng: &mut Rng) {
        if !individual.genome.is_empty() {
            let index = rng.random_range(0..individual.genome.len());
            individual.genome[index] = !individual.genome[index];
            // The cached count no longer describes the changed genome.
            individual.score = f64::NAN;
        }
    }

    /// Scores the individual by the number of `true` bits in its genome. Empty genomes remain
    /// invalid.
    fn evaluate(&self, individual: &mut OneMax) {
        let one_bits = individual.genome.iter().filter(|&&one| one).count();
        individual.score = one_bits as f64;
    }
}

/// Problem size and search budget supplied on the command line.
struct Options {
    genome_length: usize,
    population_size: usize,
    generation_limit: usize,
    time_limit: Option<Duration>,
    seed: u64,
}

impl Options {
    /// Reads overrides for the defaults, or returns None after printing --help.
    fn parse() -> Option<Self> {
        let mut options = Self {
            genome_length: 100,
            population_size: 100,
            generation_limit: 200,
            time_limit: None,
            seed: 42,
        };
        let mut args = std::env::args().skip(1);
        while let Some(option) = args.next() {
            match option.as_str() {
                "--bits" => options.genome_length = option_value(&option, args.next()),
                "--population" => options.population_size = option_value(&option, args.next()),
                "--generations" => options.generation_limit = option_value(&option, args.next()),
                "--time" => options.time_limit = Some(seconds_value(&option, args.next())),
                "--seed" => options.seed = option_value(&option, args.next()),
                "--help" => {
                    println!("{USAGE}");
                    return None;
                }
                _ => exit_with_usage(&format!("unknown option {option}")),
            }
        }
        Some(options)
    }
}

/// The usage of the example's options.
const USAGE: &str = "Options:
  --bits <N>         The length of the genome [default: 100]
  --population <N>   The number of individuals in a generation [default: 100]
  --generations <N>  The most generations to run [default: 200]
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

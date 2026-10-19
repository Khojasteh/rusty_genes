// Flow shop scheduling: using a genetic algorithm to find a good order for a set of jobs.
//
// Every job is processed by every machine, in the same order of machines, and each machine
// processes one job at a time. All machines process the jobs in the same order, which is the
// solution we search for. A job must wait if its next machine is busy. The score is the
// makespan: the time when the last machine finishes the last job. Lower is better.
//
// Each individual stores a permutation of the job IDs, so every job occurs exactly once. The
// model combines parents with order crossover and mutates an order by moving one job. Both
// operators preserve the permutation. A population explores several such orders at once.
//
// The adaptive genetic algorithm of Srinivas and Patnaik sets crossover and mutation rates
// from the parents' scores: it protects better orders and changes weaker ones more readily.
// Its default roulette-wheel selector favors orders in proportion to how much shorter they
// are than the longest order in the current population.
//
// We do not compute an exact optimum here. The run stops at a generation limit, at an optional
// time limit, or when the best score stops improving, and we compare the result with a lower
// bound that no order can beat. A gap above that bound does not tell us whether the best order
// is optimal. Run it with:
//
//     cargo run --release --example flow_shop
//
// Try a larger problem once you have followed the default run:
//
//     cargo run --release --example flow_shop -- --jobs 500 --machines 20

use rusty_genes::*;
use std::{
    hash::{Hash, Hasher},
    time::Duration,
};

/// Builds a flow shop instance, searches its job orders, and compares the best with a bound.
fn main() -> Result<(), EvolutionError> {
    let Some(Options {
        job_count,
        machine_count,
        population_size,
        generation_limit,
        stagnation_limit,
        time_limit,
        seed,
    }) = Options::parse()
    else {
        return Ok(());
    };

    let model = sample_problem(job_count, machine_count);
    let lower_bound = model.lower_bound();

    // Adaptive rates protect promising orders: candidates from fitter parents are disrupted
    // less often than those from weaker parents. The model below defines what crossover and
    // mutation do to a job order; the algorithm decides how often to use them.
    // Its default survival rule compares each candidate with its fitter parent.
    let mut parameters = AdaptiveGeneticAlgorithmParameters::default();
    parameters.population_size = population_size;
    let mut evolution = AdaptiveGeneticAlgorithm::new(model, parameters)?
        .into_evolution()
        .with_seed(seed);

    // We have no known optimum to use as a goal. Stagnation means that the best score of the
    // entire run has not improved for this many generations; it does not prove optimality.
    // The generation limit is a second stopping rule. The optional time budget is checked
    // between generations, so a generation can run past it.
    let mut limits = Limits::new()
        .generations(generation_limit)
        .stagnation(stagnation_limit);
    if let Some(time_limit) = time_limit {
        limits = limits.time(time_limit);
    }
    // `run` asks Schedule::new_random for the initial orders and evaluates them as generation
    // zero. The callback reports progress and decides whether another generation may be bred.
    let outcome = evolution.run(&job_count, |state| {
        if state.generation() % 100 == 0 {
            println!(
                "Generation {:4}: shortest makespan {} minutes",
                state.generation(),
                state.fittest_so_far().score()
            );
        }
        limits.check(state)
    })?;

    // Report the best order from any generation, alongside a bound rather than a known optimum.
    let state = outcome.state();
    let best = state.fittest_so_far();
    println!("Shortest makespan: {} minutes", best.score());
    println!("Lower bound: {lower_bound} minutes");
    println!("Order of the jobs: {:?}", best.jobs);
    println!(
        "{} generations in {:.3} seconds ({})",
        state.generation(),
        state.elapsed().as_secs_f64(),
        outcome.reason()
    );
    Ok(())
}

/// One candidate schedule: the common job order used on every machine, with its cached score.
#[derive(Debug, Clone)]
struct Schedule {
    /// A permutation of the job IDs, in processing order.
    jobs: Vec<usize>,
    /// The completion time in minutes, or NaN until this order has been evaluated.
    makespan: f64,
}

impl PartialEq for Schedule {
    /// Compares job orders, ignoring the cached score so repeated schedules are duplicates.
    fn eq(&self, other: &Self) -> bool {
        self.jobs == other.jobs
    }
}

impl Hash for Schedule {
    /// Hashes the job order used by equality, allowing the population to find duplicates.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.jobs.hash(state);
    }
}

impl From<Vec<usize>> for Schedule {
    /// Wraps a job order with an invalid score; the engine evaluates it before admission.
    fn from(jobs: Vec<usize>) -> Self {
        Self {
            jobs,
            makespan: f64::NAN,
        }
    }
}

impl Individual for Schedule {
    /// Returns the cached makespan in minutes; the model tells the algorithm to minimize it.
    fn score(&self) -> f64 {
        self.makespan
    }
}

impl RandomIndividual<usize> for Schedule {
    /// Creates a starting schedule by shuffling every job ID in `0..job_count` exactly once.
    fn new_random(job_count: &usize, rng: &mut Rng) -> Self {
        // Shuffling a complete list keeps every job, without introducing duplicates.
        let mut jobs = (0..*job_count).collect::<Vec<_>>();
        rng.shuffle(&mut jobs);
        jobs.into()
    }
}

/// The fixed processing times and the rules for breeding and evaluating schedules.
struct FlowShopModel {
    /// The processing time of each job on each machine, in the order of the machines.
    processing_times: Vec<Vec<f64>>,
    /// The positive number of machines, also the number of times stored for each job.
    machine_count: usize,
}

impl FlowShopModel {
    /// Returns the time at which the last machine finishes the last job when the jobs are
    /// processed in the given order.
    fn makespan(&self, jobs: &[usize]) -> f64 {
        // Track when each machine becomes free after processing the preceding jobs.
        let mut finished = vec![0.0; self.machine_count];
        for &job in jobs {
            let times = &self.processing_times[job];
            // The first machine has no earlier machine to wait for.
            finished[0] += times[0];
            for machine in 1..self.machine_count {
                // A machine starts a job when it is free and the previous machine has finished it.
                finished[machine] = finished[machine].max(finished[machine - 1]) + times[machine];
            }
        }
        // The last job's departure from the last machine completes the whole schedule.
        finished[self.machine_count - 1]
    }

    /// Returns a lower bound in minutes, for judging the quality of a candidate schedule.
    ///
    /// Every job needs its total processing time. Every machine needs its total processing time,
    /// after the shortest time any job takes on the machines before it, and before the shortest
    /// time any job takes on the machines after it. The strongest of these bounds applies to
    /// every order, but need not be attainable by any order.
    fn lower_bound(&self) -> f64 {
        // Even with no waiting, the longest job must complete all its own operations.
        let job_bound = self
            .processing_times
            .iter()
            .map(|times| times.iter().sum::<f64>())
            .fold(0.0, f64::max);
        // Each machine must process its full workload, with a possible lead-in and tail.
        let machine_bound = (0..self.machine_count)
            .map(|machine| {
                // Ignore contention to find the shortest possible time before or after this
                // machine. These can come from different jobs; that only weakens the bound.
                let shortest = |machines: std::ops::Range<usize>| {
                    self.processing_times
                        .iter()
                        .map(|times| times[machines.clone()].iter().sum::<f64>())
                        .fold(f64::INFINITY, f64::min)
                };
                let load = self
                    .processing_times
                    .iter()
                    .map(|times| times[machine])
                    .sum::<f64>();
                shortest(0..machine) + load + shortest(machine + 1..self.machine_count)
            })
            .fold(0.0, f64::max);
        job_bound.max(machine_bound)
    }
}

impl EvolutionModel for FlowShopModel {
    /// This model breeds and scores permutations of the job IDs.
    type Individual = Schedule;

    /// Treats a shorter completion time as a fitter schedule, without negating the score.
    fn objective(&self) -> Objective {
        Objective::Minimize
    }

    /// Keeps a segment of one order and fills the rest in the other parent's relative order.
    ///
    /// Order crossover keeps every job exactly once, as a valid schedule requires.
    fn crossover(&self, parent1: &Schedule, parent2: &Schedule, rng: &mut Rng) -> Schedule {
        CrossoverStrategy::Order
            .crossover(&parent1.jobs, &parent2.jobs, rng)
            .into()
    }

    /// Moves one job to another position, exploring a neighboring order without losing a job.
    fn mutate(&self, schedule: &mut Schedule, rng: &mut Rng) {
        MutationStrategy::Insertion.mutate(&mut schedule.jobs, rng);
        // The changed order needs a fresh makespan before it can compete with its parents.
        schedule.makespan = f64::NAN;
    }

    /// Simulates the machines in this job order and caches the resulting makespan.
    fn evaluate(&self, schedule: &mut Schedule) {
        schedule.makespan = self.makespan(&schedule.jobs);
    }
}

/// Builds synthetic processing times in whole minutes, with a separate fixed seed.
/// Changing the search's --seed changes its choices, keeping this problem instance the same.
fn sample_problem(job_count: usize, machine_count: usize) -> FlowShopModel {
    let mut rng = Rng::from_seed(7);
    let processing_times = (0..job_count)
        .map(|_| {
            (0..machine_count)
                .map(|_| rng.random_range(1..100) as f64)
                .collect()
        })
        .collect();
    FlowShopModel {
        processing_times,
        machine_count,
    }
}

/// Problem dimensions and search budget supplied on the command line.
struct Options {
    job_count: usize,
    machine_count: usize,
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
            job_count: 20,
            machine_count: 5,
            population_size: 100,
            generation_limit: 3000,
            stagnation_limit: 200,
            time_limit: None,
            seed: 42,
        };
        let mut args = std::env::args().skip(1);
        while let Some(option) = args.next() {
            match option.as_str() {
                "--jobs" => options.job_count = option_value(&option, args.next()),
                "--machines" => options.machine_count = option_value(&option, args.next()),
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
        if options.job_count == 0 || options.machine_count == 0 {
            exit_with_usage("there must be at least one job and one machine");
        }
        Some(options)
    }
}

/// The usage of the example's options.
const USAGE: &str = "Options:
  --jobs <N>         The number of jobs to schedule [default: 20]
  --machines <N>     The number of machines that process every job [default: 5]
  --population <N>   The number of individuals in a generation [default: 100]
  --generations <N>  The most generations to run [default: 3000]
  --stagnation <N>   The generations without improvement that stop the run [default: 200]
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

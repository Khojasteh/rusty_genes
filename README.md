# RustyGenes

RustyGenes is a Rust library for building and running evolutionary algorithms, with a focus on
permutation-based problems, where a solution is an ordering of items. You describe your problem
as a model, and the library evolves a population of candidate solutions towards a better
score, breeding each generation in parallel. Models can maximize rewards or minimize costs
using scores in their own units.

The library provides building blocks rather than a ready-made solver: you choose the operators
that suit your problem, and you can replace any step of an algorithm with your own.

## Features

- **Algorithms:** a standard genetic algorithm, and an adaptive genetic algorithm that adjusts
  its crossover and mutation rates to the diversity of the population. Runs stop automatically
  when the model's goal score is reached.
- **Selection:** tournament, roulette wheel, Boltzmann, rank, linear, and elitist.
- **Crossover:** common partition, successor, edge recombination (ERX), partially mapped (PMX),
  order (OX), cycle (CX), and single-point (SPX). Through
  `CrossoverStrategy::crossover_with_repetition`, the permutation crossovers also combine
  genomes in which genes repeat, such as schedules that list each job once per operation.
- **Mutation:** swap, scramble, inversion, insertion, and duplicate.
- **Extensibility:** custom models, selection mechanisms, survival policies, and algorithms
  through traits.
- **Reproducibility:** a seeded run gives the same result on any number of threads.

## Installation

```sh
cargo add rusty_genes
```

RustyGenes requires Rust 1.85 or later.

Models use the library's own random number generator, `rusty_genes::Rng`, so they need no
other dependency. To use the distributions of `rand` 0.10 with it, enable the `rand-0_10`
feature.

## Quick start

The [OneMax example](examples/one_max.rs) evolves strings of bits towards one in which every bit
is `1`. Its score is the number of set bits, so a 100-bit solution has a maximum score of 100.
The example defines the problem's individual and model, and runs the standard genetic algorithm
on them with its default parameters, which select parents by tournaments of three:

```sh
cargo run --example one_max
```

It runs the algorithm with `Evolution::run`, which creates a random initial population and calls
a closure before each generation. The closure receives the `RunState` of the run, with its
population, generation number, stagnation, fittest individual so far, and elapsed time, and
returns `ControlFlow::Break` with a `StopReason` to stop the run; `Limits` stops it after a
number of generations, generations without improvement, or a time. The model's goal also stops
the run automatically. OneMax's goal is the genome length, so no score check is needed in the
closure. `run` returns an `Outcome` with the final state of the run, including the best
individual of any generation, and the reason it stopped. With a seed, every run gives the same
result; without it, each run is seeded from the operating system.

## Defining your own problem

A problem consists of an individual, which holds a candidate solution and its score, and a
model, which creates, combines, varies, and evaluates individuals. This example evolves a
permutation of the numbers `0..n` towards ascending order by minimizing the number of misplaced
items. Its score is a count, without normalization or negation:

```rust
use rusty_genes::*;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
struct Arrangement {
    order: Vec<usize>,
    score: f64,
}

// An arrangement is identified by its order alone, not by its cached score.
impl PartialEq for Arrangement {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
    }
}

impl Hash for Arrangement {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.order.hash(state);
    }
}

impl Individual for Arrangement {
    fn score(&self) -> f64 {
        self.score
    }
}

// The argument is the number of items to arrange.
impl RandomIndividual<usize> for Arrangement {
    fn new_random(len: &usize, rng: &mut Rng) -> Self {
        let mut order = (0..*len).collect::<Vec<_>>();
        rng.shuffle(&mut order);
        Self { order, score: f64::NAN }
    }
}

struct Sorting;

impl EvolutionModel for Sorting {
    type Individual = Arrangement;

    fn objective(&self) -> Objective {
        Objective::Minimize
    }

    fn goal_score(&self, _initial_population: &Population<Arrangement>) -> Option<f64> {
        Some(0.0) // No misplaced items.
    }

    fn crossover(&self, a: &Arrangement, b: &Arrangement, rng: &mut Rng) -> Arrangement {
        let order = CrossoverStrategy::PartiallyMapped.crossover(&a.order, &b.order, rng);
        Arrangement { order, score: f64::NAN }
    }

    fn mutate(&self, individual: &mut Arrangement, rng: &mut Rng) {
        MutationStrategy::Swap.mutate(&mut individual.order, rng);
        individual.score = f64::NAN;
    }

    // Lower scores mean fewer misplaced numbers.
    fn evaluate(&self, individual: &mut Arrangement) {
        let misplaced = individual.order.iter().enumerate().filter(|(i, n)| i != *n).count();
        individual.score = misplaced as f64;
    }
}

let mut evolution = GeneticAlgorithm::new(Sorting, Default::default())
    .expect("the parameters are valid")
    .into_evolution();
let limits = Limits::new().generations(100);
let solution = evolution
    .run(&20, |state| limits.check(state))
    .expect("random arrangements are valid")
    .into_fittest_so_far();

println!("{:?} has score {}", solution.order, solution.score());
```

Keep these in mind when you define a model:

- Models maximize scores by default. Override `EvolutionModel::objective` to return
  `Objective::Minimize` when lower scores are better. `Individual::score` returns the raw
  score in either case; any finite `f64` is supported, including negative values. OneMax uses
  the number of set bits, while the sorting example uses the number of misplaced items.
- Override `EvolutionModel::goal_score` to return an optional finite target. It receives the
  evaluated initial population and is called once per run. Maximization stops at or above the
  target; minimization stops at or below it. The default `None` leaves stopping to the callback.
  Keep a generation or stagnation limit for problems that may never reach their goal.
- Populations are ordered from worst to best under their objective. `fittest`, `fittest_so_far`,
  elites, selection, and survival follow that direction. `stats().min()`, `max()`, and `avg()`
  always describe the numerical scores in their original units.
- Roulette-wheel and Boltzmann selection derive probabilities internally. For minimization,
  roulette weights are proportional to `max(0, maximum score) - score`, with uniform selection
  when all weights are zero. Boltzmann probabilities are proportional to `exp(-score / T)`,
  where the temperature `T` is expressed in the score's units. Tournament and rank-based
  selectors use ordering alone.
- An individual whose score is not finite is invalid and never joins the population. Override
  `Individual::is_valid` to reject other infeasible solutions.
- Implement `PartialEq` and `Hash` on the genome alone, excluding the cached score, as in the
  example. Equal individuals must have equal hashes. Populations remove duplicates by comparing
  only individuals with equal hashes; when equal genomes have different scores, they keep the
  better evaluation under the population's objective. `f64` has no `Hash`; hash floating-point
  genes with `hash_f64` or `hash_f64_slice`, which agree with `==`. A constant hash is valid for
  a genome that cannot be hashed, but deduplication then compares every pair, in quadratic time.
- Use an invalid score until a new or changed genome has been evaluated; the engine evaluates
  the initial population and every offspring before admitting them.
- Individuals are bred and evaluated on several threads, so the model must be `Sync`.
- The model describes the problem only; the algorithm decides how parents are selected and
  which offspring survive.

## Customizing breeding

The algorithms breed offspring through a `Reproduction`, which selects parents, applies the
model's operators, and keeps an offspring only if its survival policy accepts it. Both parts can
be replaced, including by your own implementations of `SelectionMechanism` and
`SurvivalPolicy`:

```rust
use rusty_genes::*;

/// Creates a genetic algorithm for `model` that selects parents by tournaments of four, and in
/// which every valid offspring survives, instead of only those at least as fit as their fittest
/// parent.
fn generational_tournament<M: EvolutionModel>(
    model: M,
) -> Result<GeneticAlgorithm<M>, EvolutionError> {
    let mut ga = GeneticAlgorithm::new(model, Default::default())?;
    ga.reproduction_mut().set_survival(Generational);
    ga.reproduction_mut().set_selector(TournamentSelection::new(4)?);
    Ok(ga)
}
```

## Documentation

The [API documentation](https://docs.rs/rusty_genes) describes every algorithm, operator, and
parameter, and the [changelog](CHANGELOG.md) lists the changes in each release.

## License

RustyGenes is licensed under the [MIT License](LICENSE).

## Disclaimer

RustyGenes is still evolving, and its API may change between releases. Please use it at your own
risk.

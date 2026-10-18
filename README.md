# RustyGenes

RustyGenes is a Rust library for building and running evolutionary algorithms, with a focus on
permutation-based problems, where a solution is an ordering of items. You describe your problem
as a model, and the library evolves a population of candidate solutions towards a higher
fitness score, breeding each generation in parallel.

The library provides building blocks rather than a ready-made solver: you choose the operators
that suit your problem, and you can replace any step of an algorithm with your own.

## Features

- **Algorithms:** a standard genetic algorithm, and an adaptive genetic algorithm that adjusts
  its crossover and mutation rates to the diversity of the population.
- **Selection:** tournament, roulette wheel, Boltzmann, rank, linear, and elitist.
- **Crossover:** generalized partition (GPX2), sequential constructive (SCX), edge recombination
  (ERX), partially mapped (PMX), order (OX), cycle (CX), and single-point (SPX).
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

The library includes the OneMax problem, which looks for a string of bits that are all `1`.
This runs the genetic algorithm on it with the default parameters:

```rust
use rusty_genes::one_max::*;
use rusty_genes::*;
use std::ops::ControlFlow;

let mut evolution = GeneticAlgorithm::new(OneMaxEvolutionModel, Default::default())
    .expect("the default parameters are valid")
    .into_evolution()
    .with_seed(42);

// Evolve 100-bit strings until one has only '1' bits, or for at most 200 generations.
let limits = Limits::new().generations(200);
let outcome = evolution
    .run(&100, |state| {
        if state.fittest_so_far().fitness() < 1.0 {
            limits.check(state)
        } else {
            ControlFlow::Break(StopReason::Custom("every bit is set".into()))
        }
    })
    .expect("random OneMax individuals are valid");

println!("Stopped: {}", outcome.reason());
println!("Best fitness: {}", outcome.into_fittest_so_far().fitness());
```

`run` creates a random initial population and calls the closure before each generation. The
closure receives the `RunState` of the run, with its population, generation number,
stagnation, fittest individual so far, and elapsed time, and returns `ControlFlow::Break` with
a `StopReason` to stop the run; `Limits` stops it after a number of generations, generations
without improvement, or a time. `run` returns an `Outcome` with the final state of the run,
including the fittest individual of any generation, and the reason it stopped. With the
seed, every run gives the same result; without it, each run is seeded from the operating
system.

## Defining your own problem

A problem consists of an individual, which holds a candidate solution and its fitness, and a
model, which creates, combines, varies, and evaluates individuals. This example evolves a
permutation of the numbers `0..n` towards ascending order:

```rust
use rusty_genes::*;

#[derive(Debug, Clone, PartialEq)]
struct Arrangement {
    order: Vec<usize>,
    fitness: f64,
}

impl Individual for Arrangement {
    fn fitness(&self) -> f64 {
        self.fitness
    }
}

// The argument is the number of items to arrange.
impl RandomIndividual<usize> for Arrangement {
    fn new_random(len: &usize, rng: &mut Rng) -> Self {
        let mut order = (0..*len).collect::<Vec<_>>();
        rng.shuffle(&mut order);
        Self { order, fitness: 0.0 }
    }
}

struct Sorting;

impl EvolutionModel for Sorting {
    type Individual = Arrangement;

    fn crossover(&self, a: &Arrangement, b: &Arrangement, rng: &mut Rng) -> Arrangement {
        let order = CrossoverStrategy::PartiallyMapped.crossover(&a.order, &b.order, rng);
        Arrangement { order, fitness: 0.0 }
    }

    fn mutate(&self, individual: &mut Arrangement, rng: &mut Rng) {
        MutationStrategy::Swap.mutate(&mut individual.order, rng);
    }

    // The fitness is the share of the numbers that are in their place.
    fn evaluate(&self, individual: &mut Arrangement) {
        let in_place = individual.order.iter().enumerate().filter(|(i, n)| i == *n).count();
        individual.fitness = in_place as f64 / individual.order.len() as f64;
    }
}

let mut parameters = GeneticAlgorithmParameters::default();
parameters.selection_strategy = SelectionStrategy::Tournament(3);
let mut evolution = GeneticAlgorithm::new(Sorting, parameters)
    .expect("the parameters are valid")
    .into_evolution();
let limits = Limits::new().generations(100);
let solution = evolution
    .run(&20, |state| limits.check(state))
    .expect("random arrangements are valid")
    .into_fittest_so_far();

println!("{:?} has fitness {}", solution.order, solution.fitness());
```

Keep these in mind when you define a model:

- The library maximizes fitness. For a minimization problem, give better solutions a higher
  score, for example by negating the cost.
- An individual whose fitness is not finite is invalid and never joins the population. Override
  `Individual::is_valid` to reject other infeasible solutions.
- Individuals are bred and evaluated on several threads, so the model must be `Sync`.
- The model describes the problem only; the algorithm decides how parents are selected and
  which offspring survive.

## Customizing breeding

The algorithms breed offspring through a `Reproduction`, which selects parents, applies the
model's operators, and keeps an offspring only if its survival policy accepts it. Both parts can
be replaced, including by your own implementations of `SelectionMechanism` and
`SurvivalPolicy`:

```rust
use rusty_genes::one_max::*;
use rusty_genes::*;

let mut ga = GeneticAlgorithm::new(OneMaxEvolutionModel, Default::default())
    .expect("the default parameters are valid");

// Every valid offspring survives, instead of only those at least as fit as their fittest parent.
ga.reproduction_mut().set_survival(Generational);
ga.reproduction_mut()
    .set_selector(TournamentSelection::new(4).expect("the size is positive"));

let limits = Limits::new().generations(50);
let solution = ga
    .into_evolution()
    .run(&100, |state| limits.check(state))
    .expect("random OneMax individuals are valid")
    .into_fittest_so_far();

println!("Best fitness: {}", solution.fitness());
```

## Documentation

The [API documentation](https://docs.rs/rusty_genes) describes every algorithm, operator, and
parameter, and the [changelog](CHANGELOG.md) lists the changes in each release.

## License

RustyGenes is licensed under the [MIT License](LICENSE).

## Disclaimer

RustyGenes is still evolving, and its API may change between releases. Please use it at your own
risk.

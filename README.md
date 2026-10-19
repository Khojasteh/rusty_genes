# RustyGenes

**Describe a solution. Give it a score. Let evolution search for something better.**

RustyGenes is a Rust library for genetic algorithms (GAs). Use it to explore job schedules,
find useful orderings, or fit numerical parameters. It is especially well equipped for
permutations, where a solution is an ordering of items, and you can also define your own
representation.

Start with an algorithm's defaults, then try different ways to combine and vary solutions.
RustyGenes handles the generation loop and parallel breeding, so you can focus on your problem.

## Features

- **Three ways to search.** Choose a standard GA with fixed crossover and mutation probabilities,
  an adaptive GA that adjusts them to the parents' scores, or a diversity-guided GA that switches
  between combining solutions and introducing variation.
- **Plenty of operators to experiment with.** Pick from seven crossover strategies, five mutation
  strategies, and six ways to select parents. Permutation crossovers also support repeated genes,
  such as schedules that list each job once per operation.
- **Scores that make sense.** Maximize rewards or minimize costs, distances, and errors directly
  in their original units.
- **Parallel breeding built in.** Candidates are bred and evaluated across threads with Rayon.
- **Control when to stop.** Set a goal, limit generations, stop after a stretch without
  improvement, or set a time budget. Every run returns its best solution so far and why it stopped.
- **Repeatable experiments.** Set a seed to compare changes using repeatable random choices.
  See [reproducibility](#stopping-and-reproducibility) for the conditions.
- **Room to make it yours.** Supply your own solution type and model, replace parent selection
  or survival, add local search, or implement a whole algorithm through traits.

## Try an example

With Rust 1.85 or later, run this **from a checkout of this repository**:

```sh
cargo run --example one_max
```

The [OneMax example](examples/one_max.rs) starts with random strings of 100 bits and searches
for one in which every bit is `1`. Its score is the number of ones, so you know what success
looks like: a score of 100. The output shows the best score, the number of generations, and
why the search stopped.

The examples each show a different representation and algorithm. Read them in this order:

| Example | What you search for | What it teaches |
| --- | --- | --- |
| [OneMax](examples/one_max.rs) | A bit string with as many ones as possible | The basic GA loop, maximization, and a known goal |
| [Flow shop](examples/flow_shop.rs) | A job order with a short completion time | Permutations, minimization, and adaptive breeding rates |
| [Curve fitting](examples/curve_fitting.rs) | Four parameters that fit noisy measurements | Real-valued genomes, custom operators, and diversity-guided search |

For the larger examples, use a release build:

```sh
cargo run --release --example flow_shop
cargo run --release --example curve_fitting
```

Each file starts with the search setup in `main`, followed by the solution representation and
model. Command-line parsing lives at the end. Use `--help` to see the problem sizes, search
limits, and seed you can change:

```sh
cargo run --example one_max -- --help
cargo run --release --example flow_shop -- --jobs 50 --machines 10 --population 200
```

A GA searches for good solutions; it does not generally prove they are optimal. The flow shop
example compares its result with a lower bound. The curve fitting example compares with the
parameters used to generate the noisy data, which need not be the best fit to that data.

## How the pieces fit

An **individual** is one candidate solution. Its **genome** is the data describing that solution:
a bit string, a job order, or a set of numbers. A **population** is a group of distinct, valid
individuals. A score tells the algorithm which ones are better.

A genetic algorithm repeats a few steps to build each new **generation**:

1. **Select parents:** favor promising solutions from the current population.
2. **Breed candidates:** crossover combines parts of two parents; mutation changes a candidate
   to introduce variation.
3. **Evaluate and choose survivors:** score the candidates and decide which solutions continue.

Your `EvolutionModel` defines crossover, mutation, and scoring for your problem.
The algorithm decides when to apply these operations and which solutions survive.
By default, a candidate competes with its fitter parent; a valid candidate that is at least as
fit can survive. **Elitism** also reserves places for copies of the best current individuals.

`Evolution` runs this loop. A progress callback sees its `RunState`, and the returned `Outcome`
holds the final state, the best individual from any generation, and the stopping reason.

## Use it in your project

Add the library to your Cargo project:

```sh
cargo add rusty_genes
```

Models use the library's `Rng`, so you do not need a separate random-number dependency.
To use distributions from `rand` 0.10, enable the `rand-0_10` feature.

Here is a complete model that searches for an ascending order of the numbers `0..n`.
Each misplaced number adds one to the score, so a perfectly sorted order has a score of zero:

```rust
use rusty_genes::*;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
struct Arrangement {
    order: Vec<usize>,
    score: f64,
}

// The genome is the order itself. Scoring it must not change its identity.
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

// Start with a random ordering of all the numbers, each appearing once.
impl RandomIndividual<usize> for Arrangement {
    fn new_random(len: &usize, rng: &mut Rng) -> Self {
        let mut order = (0..*len).collect::<Vec<_>>();
        rng.shuffle(&mut order);
        Self {
            order,
            score: f64::NAN,
        }
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
        // This crossover preserves a permutation: no number is lost or repeated.
        let order = CrossoverStrategy::PartiallyMapped.crossover(&a.order, &b.order, rng);
        Arrangement {
            order,
            score: f64::NAN,
        }
    }

    fn mutate(&self, individual: &mut Arrangement, rng: &mut Rng) {
        MutationStrategy::Swap.mutate(&mut individual.order, rng);
        individual.score = f64::NAN;
    }

    // Lower scores mean fewer misplaced numbers.
    fn evaluate(&self, individual: &mut Arrangement) {
        let misplaced = individual
            .order
            .iter()
            .enumerate()
            .filter(|(i, n)| i != *n)
            .count();
        individual.score = misplaced as f64;
    }
}

fn main() -> Result<(), EvolutionError> {
    let mut evolution = GeneticAlgorithm::new(Sorting, Default::default())?
        .into_evolution()
        .with_seed(42);

    // Stop after 100 generations, or sooner if the model reaches its zero-error goal.
    let limits = Limits::new().generations(100);
    let outcome = evolution.run(&20, |state| limits.check(state))?;

    println!("Stopped: {}", outcome.reason());
    let solution = outcome.into_fittest_so_far();
    println!("{:?} has score {}", solution.order, solution.score());
    Ok(())
}
```

When adapting this to your problem:

- Put the candidate's data in your individual type and return its cached score from
  `Individual::score`. A new or changed genome can use `f64::NAN` until it is evaluated.
- Use `Objective::Minimize` for costs or errors. The default is `Objective::Maximize`.
  Any finite score is supported, including negative values.
- Define equality and hashing using the genome, excluding its cached score. This lets the
  population remove duplicates. For floating-point genomes, `hash_f64` and `hash_f64_slice`
  provide hashes consistent with `==`.
- Keep scores non-finite for invalid solutions, or override `Individual::is_valid` to reject
  other infeasible candidates. Models are shared during parallel breeding and must be `Sync`.

## Stopping and reproducibility

`Limits` can combine a generation limit, a stagnation limit, and a time limit. **Stagnation**
counts generations since the best score of the entire run last improved. Generation zero is
the initial population; a limit of 100 allows up to 100 generations to be bred after it.

Set `EvolutionModel::goal_score` when you have a target: maximization stops at or above it,
minimization at or below it. The goal is checked before the progress callback, so the callback
does not see the generation that reaches it. You can inspect that generation in the outcome.

Time limits are checked between generations, so a generation can run past the budget.
With the same library version, input, and seed, runs give the same result across thread counts
when the model and algorithm use the supplied generators for random choices, reset any adapted
state in their `start_run` hooks, and are otherwise deterministic. The callback must also make
the same stopping decisions. A run stopped by elapsed time may differ because its stopping
point depends on the machine.

## Make the search your own

The sorting example uses the algorithm's default settings. Once your model is working, you
can tune the search through its parameters: change the population size, crossover and mutation
probabilities, elitism, or the way parents are selected.

For example, to select parents using tournaments of four, start with the defaults and change
the selection strategy:

```rust
use rusty_genes::*;

let mut parameters = GeneticAlgorithmParameters::default();
parameters.selection_strategy = SelectionStrategy::Tournament(4);
```

Pass these parameters to `GeneticAlgorithm::new(model, parameters)`. The other settings keep
their default values, and the model stays the same.

When you need behavior beyond the built-in settings, you can supply custom parent selectors
and survival policies through `Reproduction`, or add local search with `EvolutionModel::improve`.
For control over the steps of each generation, implement `EvolutionAlgorithm`.

The [API documentation](https://docs.rs/rusty_genes) covers the strategies and their parameters,
and the [changelog](CHANGELOG.md) explains changes between versions.

If you are new to genetic algorithms, start with OneMax, change one setting, and see how the
search responds. The same building blocks are ready when you move to your own problem.

## License

RustyGenes is licensed under the [MIT License](LICENSE). The library is still evolving;
its API may change between releases.

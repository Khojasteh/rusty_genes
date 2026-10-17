# Changelog

All notable changes to RustyGenes are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-10-17

This release redesigns the public API, so code written for 0.1 needs changes. The model now
describes only the problem, the algorithms own how offspring are bred, and the library owns its
random number generator.

### Changed

- **Breaking:** Rust 1.85 or later is required, and the crate uses the 2024 edition.
- **Breaking:** The library's own `Rng` replaces the random number generators of `rand` in the
  public API. `EvolutionModel` no longer has the `Rng` associated type and the `rng` function;
  its operators, `RandomIndividual::new_random`, the selection mechanisms, and the operator
  functions take `&mut Rng`. `rand` is no longer part of the public API, so models need no
  dependency on it; the new `rand-0_10` feature makes `Rng` usable with `rand` 0.10.
- **Breaking:** `EvolutionModel` describes only the problem: its crossover, mutation, local
  search, and evaluation. Breeding moved to the algorithms (see `Reproduction` below):
  - `breed` is replaced by `Reproduction::breed`.
  - `can_survive` is replaced by the `SurvivalPolicy` trait.
  - `pre_generation` is replaced by `prepare`, which receives the population but not the
    algorithm's breeding state.
  - `extend_population` and `extend_population_with` are removed; `Evolution` creates the
    initial population.
  - `improve` also receives a random number generator.
- **Breaking:** Algorithms are created from a model and validated parameters, then run through
  `Evolution`:
  `GeneticAlgorithm::new(model, parameters)?.into_evolution()`, and likewise for
  `AdaptiveGeneticAlgorithm`. `Evolution::run`, `evolve_individual`, and `evolve_population` no
  longer take parameters, take `&mut self`, and return `Result<_, EvolutionError>` instead of
  panicking.
- **Breaking:** `EvolutionAlgorithm` names its model with the associated type `Model` instead of
  a generic parameter. For custom algorithms:
  - `prepare_for_next_population` is renamed `prepare_generation`, and is responsible for
    calling `EvolutionModel::prepare`.
  - `generate_new_population` is renamed `generate_offspring`. It is required, and it receives
    the run's random number generator instead of a pool of generators.
  - `reproduction_parameters` is removed; the population size comes from the new
    `population_size` method.
  - `initialize_population` takes `&mut self`, and it and `replace_population` return `Result`.
- **Breaking:** `SelectionMechanism` is no longer generic over the random number generator, and
  it requires `Send` in addition to `Sync`, so that algorithms and `Evolution` can be moved to
  other threads.
- **Breaking:** `GeneticAlgorithmParameters`, `AdaptiveGeneticAlgorithmParameters`,
  `SelectionStrategy`, `CrossoverStrategy`, and `MutationStrategy` are `#[non_exhaustive]`, so
  parameters and strategies can be added without breaking changes. Create parameters with
  `Default::default()` and change their fields, instead of using a struct expression.
- **Breaking:** `Population::new` and `Population::try_from` return `Result<_, EvolutionError>`
  and discard invalid individuals; `Population::replace` returns `Result` and leaves the
  population unchanged on failure.
- **Breaking:** The constructors and setters of the selection mechanisms that take a value,
  `SelectionStrategy::create_selector`, and the `validate` methods of the algorithm parameters
  return `Result` instead of panicking on invalid values.
- **Breaking:** The parameter of `LinearSelection` and `SelectionStrategy::Linear` is the
  selective pressure, which must be between 1.0 and 2.0; `bias` and `set_bias` are renamed
  `pressure` and `set_pressure`.
- **Breaking:** The crossover and mutation rate ranges of `AdaptiveGeneticAlgorithmParameters`
  are inclusive ranges (`0.3..=1.0`), and `parameter_adjustment_interval` must be positive.
- **Breaking:** `partially_mapped_crossover`, `order_crossover`, and `cycle_crossover` require
  genes that implement `Eq + Hash`.
- **Breaking:** `TournamentSelection` and `ElitistSelection` must be prepared for a population
  before selecting from it, as the `SelectionMechanism` trait documents.
- The algorithms keep their parent selector across runs instead of creating it anew, and reset
  it through `SelectionMechanism::reset` when a run starts.
- Roulette-wheel selection shifts negative fitness scores by the lowest one, and selects
  uniformly when all scores are zero.
- Boltzmann selection selects from the first population of a run at the initial temperature and
  cools down afterwards.
- Edge recombination crossover breaks ties between neighbors by the order of the parents rather
  than at random, and individuals with equal fitness keep their order in a population.
- `compare_fitness` follows `f64::total_cmp`.
- The permutation crossovers take time linear in the genome length instead of quadratic; at
  1,000 genes they are about 5 to 80 times faster. Tournament and elitist selection are faster.

### Added

- `Rng`, the library's random number generator, and `RngStreams`, which `Rng::split` returns for
  breeding in parallel independently of thread scheduling.
- Reproducible runs: `Evolution::with_seed`, `seed`, and `set_seed`. Runs with the same seed give
  the same result on any number of threads.
- `Reproduction`, which breeds offspring with a parent selector, crossover and mutation rates,
  and a survival policy, and is available to custom algorithms.
- The `SurvivalPolicy` trait, with the `FamilyCompetition` (the default) and `Generational`
  policies.
- `GeneticAlgorithm::reproduction` and `reproduction_mut`, and likewise for
  `AdaptiveGeneticAlgorithm`, to replace the parent selector or the survival policy.
- `SelectionMechanism::reset` and `BoltzmannSelection::initial_temperature`.
- `EvolutionError`, which reports invalid parameters and populations without a valid
  individual.
- `EvolutionAlgorithm::into_evolution`, and `Evolution::new`, `algorithm`, `algorithm_mut`, and
  `into_algorithm`.
- `GeneticAlgorithm::parameters`, `AdaptiveGeneticAlgorithm::parameters`, and
  `SelectionStrategy::validate`.
- The `IndividualOf<A>` type alias for the individuals an algorithm evolves.

### Removed

- **Breaking:** `ReproductionParameters`, which `Reproduction` replaces.
- **Breaking:** `EvolutionAlgorithm::build_evolution`, `EvolutionAlgorithm::configure`, and the
  `EvolutionParameters` trait, which the new construction replaces.
- **Breaking:** The `From<M>` and `Default` implementations of `GeneticAlgorithm` and
  `AdaptiveGeneticAlgorithm`, and the `Default` implementation of `Evolution`. Create an
  algorithm with `new(model, parameters)` instead, and turn it into an `Evolution` with
  `into_evolution` or `Evolution::from`.
- **Breaking:** `IndividualWrapper`, `interpolate_f64`, and `interpolate_usize`.
- **Breaking:** The `Copy` implementations of `RouletteWheelSelection`, `BoltzmannSelection`,
  `RankSelection`, and `LinearSelection`, and the `Eq` implementation of `RankSelection`, since
  they now hold the weights they select by. `RankSelection` is created with `new` or `default`
  instead of `RankSelection {}`.
- The `object-pool` dependency.

### Fixed

- An algorithm with a population of one panicked, and a generation in which exactly one
  individual was left to breed after the elites got no new individual.
- The adaptive genetic algorithm stopped crossover and mutation for good when the fitness scores
  of the initial population had no spread.
- Boltzmann selection overflowed after cooling down and then favored weak individuals.
- Linear selection with a bias of 1.0 always selected the fittest individual.
- A population could keep duplicate individuals with equal fitness scores.
- `order_crossover` with an empty range returned an invalid genome or never returned.
- `common_partitions` split a chain of common neighbors that passes through the first gene of
  the first genome, so GPX2 crossover could break an edge that both parents share.
- `CrossoverStrategy::crossover` panicked on empty genomes.
- Documentation that described parameters and behavior the library does not have.

## [0.1.3] - 2023-05-22

### Changed

- **Breaking:** `Population` dereferences to a read-only slice of individuals instead of a
  mutable `Vec`, so a population can no longer be modified outside of the algorithm.

## [0.1.2] - 2023-05-21

### Fixed

- Generations grew beyond the population size, because the elites were added on top of a full
  set of offspring.

## [0.1.1] - 2023-05-17

### Changed

- **Breaking:** `EvolutionModel::can_survive` receives the current population instead of the
  statistics of its fitness scores, and its parent argument is named `fittest_parent`.
- By default, an offspring survives when its fitness equals that of its fittest parent, not only
  when it is higher.

## [0.1.0] - 2023-05-16

### Added

- Standard and adaptive genetic algorithms over customizable evolutionary models.
- Tournament, roulette-wheel, Boltzmann, rank, linear, and elitist selection.
- GPX2, SCX, ERX, PMX, OX, CX, and single-point crossover.
- Swap, scramble, inversion, insertion, and duplicate mutation.
- An implementation of the OneMax problem.

[0.2.0]: https://github.com/Khojasteh/rusty_genes/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/Khojasteh/rusty_genes/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Khojasteh/rusty_genes/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Khojasteh/rusty_genes/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Khojasteh/rusty_genes/releases/tag/v0.1.0

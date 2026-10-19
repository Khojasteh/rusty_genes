# Changelog

All notable changes to RustyGenes are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-10-21

This release rewrites much of the library, replacing nearly half of the lines of 0.2, to make
it faster, faithful to the published algorithms, and easier to use and adapt:

- Models score individuals in their own units, to be minimized or maximized, and can stop at a
  goal.
- The adaptive genetic algorithm is now the published algorithm of Srinivas and Patnaik, the
  diversity-guided algorithm of Ursem is new, and edge recombination, cycle, and single-point
  crossover behave as their published operators do. Crossovers that only bore the names of
  published operators are renamed after what they do.
- Breeding is separate from survival, so generations stay full, and custom selection, breeding,
  and survival steps work with the same parts that the library's own algorithms use.
- Populations find duplicates by their hashes instead of comparing every pair, and compute
  their statistics only when they are requested.

Code written for 0.2 needs changes where an entry is marked **Breaking**.

### Added

- `Objective::Maximize` and `Objective::Minimize`, selected by `EvolutionModel::objective`.
  Models can minimize native costs without negating or normalizing their stored scores.
  Existing models maximize by default.
- `EvolutionModel::goal_score` defines an optional goal in the score's original units. Runs
  stop automatically with `StopReason::GoalReached` when the best score reaches or passes it in
  the optimization direction, including in the initial population. The progress callback does
  not observe the generation that reaches the goal; the outcome of the run holds it.
- `Population::with_objective`, `Population::objective`, `Objective::is_fitter`,
  `Objective::is_at_least_as_fit`, `Objective::cmp_fitness`, and `Objective::is_goal_reached`
  for custom algorithms and operators. Populations are ordered from worst to best for their
  objective; their statistics remain in the original score units.
- `RunState`, the state of a run: its current population, generation number, stagnation, the
  fittest individual of the current and every earlier generation, and the time the run has taken.
  Progress callbacks, algorithms, and models observe it, and `RunState::advance` records each
  new generation.
- `StopReason`, why a run stopped: its goal, a limit on generations, stagnation, or time, an
  interruption, or a reason of the callback's own. `Limits` stops a run after a number of
  generations, of generations without improvement, or a time, and reports which limit it reached.
  A time limit makes a run depend on the speed of the machine, so it is not reproducible from its
  seed.
- `Outcome`, the final `RunState` of a run and its `StopReason`.
- `EvolutionAlgorithm::start_run` and `EvolutionModel::start_run`, in which an algorithm and its
  model reset the state they adapt during a run. Every run calls them before it creates its
  starting individuals, so that it starts as if the algorithm and the model were new; the
  algorithm calls the model's.
- `DiversityGuidedGeneticAlgorithm`, the diversity-guided genetic algorithm of Ursem (2002). It
  breeds every candidate by crossover while the genetic diversity of the population is above a
  low threshold, and mutates every candidate, without crossover, once the diversity falls below
  it, until the diversity rises above a high threshold. The model measures the diversity as a
  `DiversityModel`, such as by `distance_to_average_point`, the measure of the paper, whose
  thresholds for real-valued genomes are the defaults. Genomes of bits or permutations keep a
  diversity above the default low threshold, so they need a higher one for the algorithm to
  mutate, as its documentation explains. Parents are selected by binary tournament by default.
- `hash_f64` and `hash_f64_slice` hash floating-point genes consistently with `==`, for the
  `Hash` implementations that `Individual` now requires.
- `CrossoverStrategy::crossover_with_repetition` combines genomes in which genes repeat, such as
  schedules that list each job once per operation, with any crossover strategy. The k-th copy
  of a gene in one parent stands for its k-th copy in the other.
- `Sampler`, through which a selection mechanism selects from one population, and `Generation`
  and `Candidate`, through which a survival policy chooses the next generation. A custom breeding
  step creates its candidates by `Candidate::new`, so that a survival policy can choose from them.
  `Reproduction::selector` returns the parent selector, `BoltzmannSelection::temperature_at` the
  temperature of a generation, and `Population::advance` replaces a population with a
  `Generation`.
- The `PlaceCompetition` survival policy, in which each candidate competes for the place of its
  fittest parent in the population, and takes it if it is valid, new, and at least as fit as the
  individual holding it. Unlike in `FamilyCompetition`, individuals that are not selected as
  parents keep their places.
- Two examples, each demonstrating another algorithm with its default selection: a flow shop
  scheduling example in `examples/flow_shop.rs`, which orders jobs through machines with the
  adaptive genetic algorithm to minimize the time at which the last job leaves the last machine,
  and stops when the best order stops improving; and a curve fitting example in
  `examples/curve_fitting.rs`, which fits real-valued parameters with its own crossover and
  mutation, using the diversity-guided genetic algorithm. The OneMax example, which uses the
  standard genetic algorithm, is described under Removed.
- The examples take command-line options that size their problems and set the population, the
  generation limits, an optional time limit in seconds, and the seed, and report how long a run
  takes and why it stopped. Without options, the flow shop schedules 20 jobs on 5 machines.

### Changed

- **Breaking:** The progress callback of `Evolution::run`, `evolve_individual`, and
  `evolve_population` receives the `RunState` of the run instead of the population, and returns
  `ControlFlow::Continue(())` to continue or `ControlFlow::Break` with a `StopReason` to stop,
  instead of a `bool`. Replace `|population| population.generation() < 100` with
  `|state| Limits::new().generations(100).check(state)`, or test `state.generation()` and
  return the `ControlFlow` yourself. The three methods return an `Outcome`: use
  `Outcome::into_fittest_so_far` for the fittest individual of any generation, which `run` and
  `evolve_individual` returned instead of the fittest individual of the last generation, and
  `Outcome::state` for the last population, which `evolve_population` returned.
- **Breaking:** A `Population` holds only the individuals of one generation and its objective.
  `Population::generation` and `Population::stagnation` are replaced by `RunState::generation`
  and `RunState::stagnation`, and `Population::replace` no longer counts generations.
- **Breaking:** `EvolutionAlgorithm::prepare_generation`, `EvolutionAlgorithm::generate_offspring`,
  `EvolutionModel::prepare`, and `Reproduction::breed_into` take the `RunState` of the run instead
  of its population, which `RunState::population` returns.
  `EvolutionAlgorithm::initialize_population` is replaced by `evaluate_initial`, which evaluates
  the starting individuals; `Evolution` forms the initial population from them under the model's
  objective. A custom algorithm implements `EvolutionAlgorithm::start_run`, which resets its
  adapted state instead of preparing a generation numbered zero, and calls
  `EvolutionModel::start_run` on its model from it.
- **Breaking:** `AdaptiveGeneticAlgorithm` is now the adaptive genetic algorithm of Srinivas and
  Patnaik (1994). Each candidate is bred at crossover and mutation rates that depend on the
  scores of its parents: the fittest individuals are protected from disruption, and those less
  fit than the average are disrupted. `AdaptiveGeneticAlgorithmParameters` sets the rates by
  `crossover_rate`, `subaverage_crossover_rate`, `mutation_rate`, `subaverage_mutation_rate`, and
  `minimum_mutation_rate`, whose defaults are the values of the paper; `crossover_rate_range`,
  `mutation_rate_range`, and `parameter_adjustment_interval` are removed. The algorithm of 0.2
  adjusted the rates of a whole generation by the standard deviation of the scores relative to
  that of the initial generation. That spread shrinks with the scores themselves, whatever the
  diversity of the genomes, and its rate ranges and adjustment interval had no basis; to adapt
  breeding to the diversity of the genomes, use `DiversityGuidedGeneticAlgorithm`.
- **Breaking:** A `Reproduction` no longer holds crossover and mutation rates; the algorithm
  breeding with it gives them to `Reproduction::breed` and `Reproduction::breed_into` as
  `BreedingRates`, which may depend on the parents of each candidate. `FixedRates` are the same
  rates for every candidate. `Reproduction::new` and `Reproduction::from_strategy` take only the
  parent selector, and `Reproduction::crossover_rate`, `Reproduction::mutation_rate`, and
  `Reproduction::set_rates` are removed. `GeneticAlgorithm::rates` and
  `DiversityGuidedGeneticAlgorithm::rates` return the rates of the current generation; the
  rates of a `GeneticAlgorithm` are those of its parameters and no longer change after it is
  created.
- **Breaking:** A `SelectionMechanism` selects through the `Sampler` that its `sampler` method
  creates for a population and its generation number, so it can no longer select from a
  population it was not prepared for. `sampler` replaces `prepare`, `select`, `select_distinct`,
  and `reset`: a custom mechanism returns a sampler that holds what `prepare` computed and
  selects as `select` did.
  `Sampler::select_other` excludes an individual by its index and takes no number of tries.
  `Reproduction::prepare` and `Reproduction::reset` are removed, since breeding creates its own
  sampler.
- **Breaking:** Boltzmann selection takes its temperature from the generation number its sampler
  is created for: the initial temperature at generation zero, cooled once for each later
  generation, so a new run starts at the initial temperature without a reset.
  `BoltzmannSelection::temperature` and `set_temperature` are replaced by `temperature_at`, and
  clones and comparisons consider only the parameters. Seeded runs give the same results.
- **Breaking:** Breeding and survival are separate steps. `Reproduction::breed` takes the sampler
  of the parent selector and returns a `Candidate`, the evaluated individual with the indices of
  its parents, instead of the offspring or a copy of its fittest parent. `SurvivalPolicy::survives`
  is replaced by `select_survivors`, which adds the survivors among the population and the
  candidates to the next `Generation`; a custom policy finds the fittest parent of a candidate by
  `Candidate::fittest_parent`.
- **Breaking:** `EvolutionAlgorithm::select_elites`, `generate_offspring`, and
  `replace_population` take the next generation as a `Generation` instead of a vector, and
  `Reproduction::breed_into` takes it instead of a vector and a size. The default
  `replace_population` replaces the population by the new `Population::advance`, which sorts the
  generation without checking it for invalid and duplicate individuals again. A custom algorithm
  adds elites by `Generation::insert` and replaces the population by `Population::advance`.
- **Breaking:** `Individual::fitness` is renamed `Individual::score`. Update trait
  implementations and score access in custom models, operators, callbacks, and examples.
  Replace `compare_fitness(a, b)` with `objective.cmp_fitness(a.score(), b.score())` so custom
  comparisons respect the optimization direction.
- **Breaking:** `Individual` requires `Hash`, consistent with `PartialEq`: equal individuals
  must have equal hashes. Implement it on the solution alone, excluding a cached score.
  Populations compare only individuals with equal hashes when removing duplicates, which takes
  `O(n log n)` time when few hashes are equal. A constant hash remains valid, but compares every
  pair.
- **Breaking:** `edge_recombination_crossover` takes a random number generator, with which it
  breaks ties between neighbors and chooses the gene after a dead end. `cycle_crossover` takes
  the position whose cycle comes from the first parent; `CrossoverStrategy::Cycle` chooses it at
  random.
- **Breaking:** The crossovers named after Generalized Partition Crossover 2 (GPX2) and
  Sequential Constructive Crossover (SCX) are renamed after what they do, since those published
  operators choose by edge costs, which these crossovers do not use.
  `CrossoverStrategy::GeneralizedPartition2` and `generalized_partition_2_crossover` are renamed
  `CrossoverStrategy::CommonPartition` and `common_partition_crossover`, and
  `CrossoverStrategy::SequentialConstructive` and `sequential_constructive_crossover` are renamed
  `CrossoverStrategy::Successor` and `successor_crossover`. Their behavior is unchanged.
- `GeneticAlgorithmParameters` select parents by a tournament of three by default, instead of
  by roulette wheel, which depends on the scale and offset of the scores. Seeded runs with the
  default parameters give different results.
- `FamilyCompetition` keeps a fittest parent once, however many of its candidates lose to it,
  and `Generational` survival no longer keeps the fittest parent of an invalid candidate. With
  both policies, the fittest individuals of the population that the next generation lacks take
  the places left by invalid and repeated candidates, instead of copies of parents. Seeded runs
  give different results.
- Tournament, rank, linear, and elitist selection give individuals with equal scores equal
  chances of being selected, instead of favoring those later in the population. Rank and linear
  selection give them the average of their probabilities, a tournament is won by the first drawn
  of its best candidates, and elitist selection shares the chances of the tied individuals that
  remain selectable among all of them. Seeded runs that use these strategies give different
  results.
- A reproduction selects both parents of a candidate before deciding whether to cross them over,
  so that the crossover rate can depend on them. Seeded runs give different results.
- `Population::stats` computes the statistics when they are first requested, so creating or
  replacing a population no longer pays for them. They are computed in two passes over the
  sorted scores, so averages, variances, and standard deviations can differ from earlier
  versions in their last digits.

### Removed

- **Breaking:** The `one_max` module is removed from the library. OneMax is now the example in
  `examples/one_max.rs`, which shows how an individual and a model are defined; copy its
  `OneMax` and `OneMaxEvolutionModel` into your project to keep using them. Unlike the module of
  0.2.0, the example scores the number of set bits rather than their fraction of the genome,
  supplies the genome length as its goal, compares and hashes genomes without their cached
  scores, and mutates by flipping one random bit.

### Fixed

- Candidates had two parents less often than the crossover rate, because a candidate became a
  copy of its first parent when three selections of the second parent all returned the first.
  With elitist selection that left only the fittest individual selectable, or roulette-wheel
  selection in which only the fittest had a positive score, no candidate had two parents. In a
  population of more than one individual, a candidate now has two parents with the probability
  of the crossover rate, and the selection mechanisms of the library select the second parent
  among the others in proportion to their chances of being selected. Seeded runs give different
  results.
- Generations could hold far fewer individuals than the population size, because an offspring
  that copies a parent, or a parent kept in place of a rejected or invalid offspring, repeats an
  individual that the population keeps only once; in the flow shop example, a generation of 100
  fell to 3 individuals. The survival policies now choose the next generation from the
  population and the candidates, so a generation is smaller than the population size only when
  they hold too few valid and distinct individuals, such as when the problem has fewer solutions
  than the population size.
- Roulette-wheel selection could discard the fittest individual's weight or select uniformly
  when shifting or summing large finite scores overflowed. Score differences are now computed
  before scaling to preserve selection probabilities for adjacent representable scores.
- Boltzmann selection could lose valid probabilities when score differences overflowed before
  division by a large temperature, or when subnormal score units and temperatures were rounded
  up to the smallest normal temperature.
- Score statistics could overflow intermediate calculations for large finite scores, so the
  standard deviation could become infinite or NaN. It now remains available whenever it is
  representable, even when the variance alone is not.
- Population deduplication missed equal individuals with different cached scores.
  Equality is now checked across score groups and retains the best evaluation of each genome.
- Signed zeros compare equally for improvement tracking and goal checks.
- The stagnation of a run, now `RunState::stagnation`, was reset when a generation regained the
  score of a fitter generation it had lost, rather than counting the generations since the best
  score improved.
- Common-neighbor detection discarded two-gene genomes depending on their argument position.
- `bounded_random_range` panicked when `max_len` was `usize::MAX`, and when `n` was
  `usize::MAX` and the range could be empty, although both are valid bounds.
- Insertion mutation always moved the later of two random genes to the position of the earlier
  one, so the first gene never moved and no gene moved towards the end. It now moves a random
  gene to any other position, in either direction. Seeded runs that use it give different
  results.
- Edge recombination crossover broke ties between neighbors by the order of the parents, and
  continued from the first parent's first unused gene after a dead end. As in the published
  operator, it now chooses at random in both cases. Seeded runs that use it give different
  results.
- Cycle crossover took alternate cycles from the two parents, starting at the first position. As
  in the published operator, the offspring now takes the cycle through a random position from
  the first parent and every other position from the second. Seeded runs that use it give
  different results.
- Single-point crossover could choose a point before the first gene, which copied the second
  parent, and never copied the first. The point is now between two genes, so the offspring has
  genes of both parents. Seeded runs that use it give different results.
- Partially mapped crossover never returned when the parents were not permutations of the same
  genes and the mapping returned to a gene. It now panics, like cycle crossover.
- Order crossover never returned when every gene of the second parent was in the range copied
  from the first, which can happen when the parents are not permutations of the same genes. It
  now panics, like partially mapped crossover.
- Common partition crossover, GPX2 in 0.2, returned an offspring shorter than its parents when
  the first parent had a gene more than once. It now panics, since the parents are not
  permutations of the same genes.

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

[0.3.0]: https://github.com/Khojasteh/rusty_genes/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Khojasteh/rusty_genes/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/Khojasteh/rusty_genes/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Khojasteh/rusty_genes/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Khojasteh/rusty_genes/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Khojasteh/rusty_genes/releases/tag/v0.1.0

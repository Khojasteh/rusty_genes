use criterion::{
    BatchSize, BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main,
    measurement::*,
};
use rusty_genes::*;
use std::hint::black_box;

pub fn benchmark_crossover_strategies<M: Measurement>(
    mut group: BenchmarkGroup<'_, M>,
    genome_length: usize,
) {
    let mut genome1 = Vec::from_iter(0..genome_length);
    let mut genome2 = Vec::from_iter(0..genome_length);
    {
        let mut rng = Rng::from_seed(0);
        rng.shuffle(&mut genome1);
        rng.shuffle(&mut genome2);
    }

    group.throughput(Throughput::Elements(genome_length as u64));

    group.bench_function("GPX2", |b| {
        let mut rng = Rng::from_seed(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::GeneralizedPartition2.crossover(&p1, &p2, &mut rng),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("SCX", |b| {
        let mut rng = Rng::from_seed(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::SequentialConstructive.crossover(&p1, &p2, &mut rng),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("ERX", |b| {
        let mut rng = Rng::from_seed(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::EdgeRecombination.crossover(&p1, &p2, &mut rng),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("PMX", |b| {
        let mut rng = Rng::from_seed(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::PartiallyMapped.crossover(&p1, &p2, &mut rng),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("OX", |b| {
        let mut rng = Rng::from_seed(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::Order.crossover(&p1, &p2, &mut rng),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("CX", |b| {
        let mut rng = Rng::from_seed(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::Cycle.crossover(&p1, &p2, &mut rng),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("SPX", |b| {
        let mut rng = Rng::from_seed(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::SinglePoint.crossover(&p1, &p2, &mut rng),
            BatchSize::SmallInput,
        );
    });
}

pub fn benchmark_mutation_strategies<M: Measurement>(
    mut group: BenchmarkGroup<'_, M>,
    genome_length: usize,
) {
    group.throughput(Throughput::Elements(genome_length as u64));

    group.bench_function("Swap", |b| {
        let mut rng = Rng::from_seed(0);
        let mut genome = Vec::from_iter(0..genome_length);
        b.iter(|| MutationStrategy::Swap.mutate(&mut genome, &mut rng));
        black_box(genome);
    });

    group.bench_function("Duplicate", |b| {
        let mut rng = Rng::from_seed(0);
        let mut genome = Vec::from_iter(0..genome_length);
        b.iter(|| MutationStrategy::Duplicate.mutate(&mut genome, &mut rng));
        black_box(genome);
    });

    group.bench_function("Insertion", |b| {
        let mut rng = Rng::from_seed(0);
        let mut genome = Vec::from_iter(0..genome_length);
        b.iter(|| MutationStrategy::Insertion.mutate(&mut genome, &mut rng));
        black_box(genome);
    });

    group.bench_function("Inversion", |b| {
        let mut rng = Rng::from_seed(0);
        let mut genome = Vec::from_iter(0..genome_length);
        b.iter(|| MutationStrategy::Inversion.mutate(&mut genome, &mut rng));
        black_box(genome);
    });

    group.bench_function("Scramble", |b| {
        let mut rng = Rng::from_seed(0);
        let mut genome = Vec::from_iter(0..genome_length);
        b.iter(|| MutationStrategy::Scramble.mutate(&mut genome, &mut rng));
        black_box(genome);
    });
}

/// Measures the helpers that analyze or sample genomes for the operators.
pub fn benchmark_operator_helpers<M: Measurement>(
    mut group: BenchmarkGroup<'_, M>,
    genome_length: usize,
) {
    let mut genome1 = Vec::from_iter(0..genome_length);
    let mut genome2 = Vec::from_iter(0..genome_length);
    {
        let mut rng = Rng::from_seed(0);
        rng.shuffle(&mut genome1);
        rng.shuffle(&mut genome2);
    }

    group.throughput(Throughput::Elements(genome_length as u64));

    group.bench_function("common_neighbors", |b| {
        b.iter(|| common_neighbors(black_box(&genome1), black_box(&genome2)));
    });

    group.bench_function("common_partitions", |b| {
        b.iter(|| common_partitions(black_box(&genome1), black_box(&genome2)));
    });

    // The random helpers take constant time, so each iteration draws one value per gene.
    group.bench_function("random_pair", |b| {
        let mut rng = Rng::from_seed(0);
        b.iter(|| {
            for _ in 0..genome_length {
                black_box(random_pair(black_box(genome_length), &mut rng));
            }
        });
    });

    group.bench_function("random_range", |b| {
        let mut rng = Rng::from_seed(0);
        b.iter(|| {
            for _ in 0..genome_length {
                black_box(random_range(black_box(genome_length), &mut rng));
            }
        });
    });

    group.bench_function("bounded_random_range", |b| {
        let mut rng = Rng::from_seed(0);
        b.iter(|| {
            for _ in 0..genome_length {
                black_box(bounded_random_range(
                    black_box(genome_length),
                    2,
                    genome_length / 2,
                    &mut rng,
                ));
            }
        });
    });
}

/// An individual with a given identity and fitness, for benchmarks that need no genome.
#[derive(Debug, Clone, PartialEq)]
struct Scored {
    id: usize,
    fitness: f64,
}

impl Individual for Scored {
    fn fitness(&self) -> f64 {
        self.fitness
    }
}

/// Returns `size` individuals with distinct fitness scores, in random order.
fn shuffled_individuals(size: usize) -> Vec<Scored> {
    let mut individuals = (0..size)
        .map(|id| Scored {
            id,
            fitness: id as f64 / size as f64,
        })
        .collect::<Vec<_>>();
    Rng::from_seed(0).shuffle(&mut individuals);
    individuals
}

/// Measures the selection work of one generation: preparing the mechanism for a population and
/// selecting as many parents as the population has individuals.
pub fn benchmark_selection_strategies<M: Measurement>(
    mut group: BenchmarkGroup<'_, M>,
    population_size: usize,
) {
    let population = Population::new(shuffled_individuals(population_size)).unwrap();
    let strategies = [
        ("Tournament", SelectionStrategy::Tournament(4)),
        ("RouletteWheel", SelectionStrategy::RouletteWheel),
        ("Boltzmann", SelectionStrategy::Boltzmann(1.0, 0.99)),
        ("Rank", SelectionStrategy::Rank),
        ("Linear", SelectionStrategy::Linear(1.5)),
        ("Elitist", SelectionStrategy::Elitist(0.5)),
    ];

    group.throughput(Throughput::Elements(population_size as u64));

    for (name, strategy) in strategies {
        group.bench_function(name, |b| {
            let mut rng = Rng::from_seed(0);
            let mut selector = strategy.create_selector::<Scored>().unwrap();
            b.iter(|| {
                selector.prepare(&population);
                for _ in 0..population_size {
                    black_box(selector.select(&population, &mut rng));
                }
            });
        });
    }
}

/// Measures creating a population, which validates, sorts, and deduplicates the individuals and
/// collects their statistics, for different shapes of the individuals' fitness scores.
pub fn benchmark_population<M: Measurement>(
    mut group: BenchmarkGroup<'_, M>,
    population_size: usize,
) {
    let distinct = shuffled_individuals(population_size);
    let equal_fitness = (0..population_size)
        .map(|id| Scored { id, fitness: 1.0 })
        .collect::<Vec<_>>();
    let duplicates = (0..population_size)
        .map(|i| Scored {
            id: i % 10,
            fitness: (i % 10) as f64,
        })
        .collect::<Vec<_>>();

    group.throughput(Throughput::Elements(population_size as u64));

    for (name, individuals) in [
        ("Distinct fitness", &distinct),
        ("Equal fitness", &equal_fitness),
        ("Duplicates", &duplicates),
    ] {
        group.bench_function(name, |b| {
            b.iter_batched(
                || individuals.clone(),
                Population::new,
                BatchSize::SmallInput,
            );
        });
    }
}

/// Measures the steps of a generation that the model and the algorithm provide by default, on
/// the OneMax problem.
pub fn benchmark_generation_steps<M: Measurement>(
    mut group: BenchmarkGroup<'_, M>,
    genome_length: usize,
    population_size: usize,
) {
    use rusty_genes::one_max::*;

    let model = OneMaxEvolutionModel;
    let mut parameters = GeneticAlgorithmParameters::default();
    parameters.population_size = population_size;
    let mut ga = GeneticAlgorithm::new(OneMaxEvolutionModel, parameters).unwrap();
    let mut rng = Rng::from_seed(0);
    let individuals = (0..population_size)
        .map(|_| OneMax::new_random(&genome_length, &mut rng))
        .collect::<Vec<_>>();
    let population = ga.initialize_population(individuals.clone()).unwrap();
    ga.prepare_generation(&population);

    group.throughput(Throughput::Elements(population_size as u64));

    group.bench_function("initialize_population", |b| {
        b.iter_batched(
            || individuals.clone(),
            |individuals| ga.initialize_population(individuals),
            BatchSize::SmallInput,
        );
    });

    // One offspring per individual, bred sequentially on one thread.
    group.bench_function("breed", |b| {
        let mut rng = Rng::from_seed(0);
        let reproduction = ga.reproduction();
        b.iter(|| {
            for _ in 0..population_size {
                black_box(reproduction.breed(&model, &population, &mut rng));
            }
        });
    });

    // A full generation of offspring, bred in parallel.
    group.bench_function("generate_offspring", |b| {
        let mut rng = Rng::from_seed(0);
        b.iter_batched_ref(
            || Vec::with_capacity(population_size),
            |offspring| ga.generate_offspring(&population, offspring, &mut rng),
            BatchSize::SmallInput,
        );
    });
}

/// Measures complete runs of the evolutionary algorithms on the OneMax problem, from the same
/// random initial population, for a fixed number of generations.
pub fn benchmark_evolution<M: Measurement>(
    mut group: BenchmarkGroup<'_, M>,
    genome_length: usize,
    population_size: usize,
    generations: usize,
) {
    use rusty_genes::one_max::*;

    let mut initial_population = Vec::with_capacity(population_size);
    {
        let mut rng = Rng::from_seed(0);
        initial_population
            .extend((0..population_size).map(|_| OneMax::new_random(&genome_length, &mut rng)));
    }

    group.sample_size(20);
    group.throughput(Throughput::Elements(generations as u64));

    group.bench_function("GA", |b| {
        let mut parameters = GeneticAlgorithmParameters::default();
        parameters.population_size = population_size;
        let mut evolution = GeneticAlgorithm::new(OneMaxEvolutionModel, parameters)
            .unwrap()
            .into_evolution()
            .with_seed(0);
        b.iter_batched(
            || initial_population.clone(),
            |individuals| {
                evolution.evolve_population(individuals, |population| {
                    population.generation() < generations
                })
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("AGA", |b| {
        let mut parameters = AdaptiveGeneticAlgorithmParameters::default();
        parameters.population_size = population_size;
        let mut evolution = AdaptiveGeneticAlgorithm::new(OneMaxEvolutionModel, parameters)
            .unwrap()
            .into_evolution()
            .with_seed(0);
        b.iter_batched(
            || initial_population.clone(),
            |individuals| {
                evolution.evolve_population(individuals, |population| {
                    population.generation() < generations
                })
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn benchmark(c: &mut Criterion) {
    c.bench_function("Stats (1M samples)", |b| {
        b.iter(|| Stats::from_iter((0..1000000).map(|i| i as f64)))
    });

    c.bench_function("Stats record (1M samples)", |b| {
        b.iter(|| {
            let mut stats = Stats::default();
            for i in 0..1000000 {
                stats.record(black_box(i as f64));
            }
            stats
        })
    });

    benchmark_crossover_strategies(c.benchmark_group("Crossover (1K genes)"), 1000);
    benchmark_crossover_strategies(c.benchmark_group("Crossover (100 genes)"), 100);
    benchmark_mutation_strategies(c.benchmark_group("Mutation (1K genes)"), 1000);
    benchmark_operator_helpers(c.benchmark_group("Operator helpers (1K genes)"), 1000);
    benchmark_selection_strategies(c.benchmark_group("Selection (100 individuals)"), 100);
    benchmark_selection_strategies(c.benchmark_group("Selection (1K individuals)"), 1000);
    benchmark_population(c.benchmark_group("Population (1K individuals)"), 1000);
    benchmark_generation_steps(
        c.benchmark_group("Generation steps (OneMax 100 bits, 100 individuals)"),
        100,
        100,
    );
    benchmark_evolution(
        c.benchmark_group("Evolution (OneMax 100 bits, 100 individuals, 20 generations)"),
        100,
        100,
        20,
    );
}

criterion_group!(benches, benchmark);
criterion_main!(benches);

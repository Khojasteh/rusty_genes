// The benchmarks run the model of the OneMax example; its `main` is not used here.
#[allow(dead_code)]
#[path = "../examples/one_max.rs"]
mod one_max;

use criterion::{
    BatchSize, Bencher, BenchmarkGroup, Criterion, Throughput, criterion_group, criterion_main,
    measurement::*,
};
use one_max::*;
use rusty_genes::*;
use std::{
    hash::{Hash, Hasher},
    hint::black_box,
};

/// Measures the diversity of the OneMax genomes for the diversity-guided algorithm, taking each
/// bit as a coordinate between 0 and 1.
impl DiversityModel for OneMaxEvolutionModel {
    fn diversity(&self, population: &Population<OneMax>) -> f64 {
        let length = population[0].genome().len();
        let bits = population.iter().map(|individual| {
            individual
                .genome()
                .iter()
                .map(|&bit| f64::from(u8::from(bit)))
        });
        distance_to_average_point(bits, (length as f64).sqrt())
    }
}

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

    group.bench_function("CommonPartition", |b| {
        let mut rng = Rng::from_seed(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::CommonPartition.crossover(&p1, &p2, &mut rng),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Successor", |b| {
        let mut rng = Rng::from_seed(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::Successor.crossover(&p1, &p2, &mut rng),
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

/// Measures crossing over genomes that hold each of `gene_count` genes `copies` times, such as
/// schedules that list each job once per operation. The copies of each gene are numbered so
/// that the strategies for permutations apply.
pub fn benchmark_crossover_with_repetition<M: Measurement>(
    mut group: BenchmarkGroup<'_, M>,
    gene_count: usize,
    copies: usize,
) {
    let mut genome1 =
        Vec::from_iter((0..gene_count).flat_map(|gene| std::iter::repeat_n(gene, copies)));
    let mut genome2 = genome1.clone();
    {
        let mut rng = Rng::from_seed(0);
        rng.shuffle(&mut genome1);
        rng.shuffle(&mut genome2);
    }
    let strategies = [
        ("CommonPartition", CrossoverStrategy::CommonPartition),
        ("Successor", CrossoverStrategy::Successor),
        ("ERX", CrossoverStrategy::EdgeRecombination),
        ("PMX", CrossoverStrategy::PartiallyMapped),
        ("OX", CrossoverStrategy::Order),
        ("CX", CrossoverStrategy::Cycle),
        ("SPX", CrossoverStrategy::SinglePoint),
    ];

    group.throughput(Throughput::Elements(genome1.len() as u64));

    for (name, strategy) in strategies {
        group.bench_function(name, |b| {
            let mut rng = Rng::from_seed(0);
            b.iter(|| {
                strategy.crossover_with_repetition(
                    black_box(&genome1),
                    black_box(&genome2),
                    &mut rng,
                )
            });
        });
    }
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

/// An individual with a given identity and score, for benchmarks that need no genome. It hashes
/// its identity, or nothing when `hashed` is false, which makes its hash constant.
#[derive(Debug, Clone, PartialEq)]
struct Scored {
    id: usize,
    score: f64,
    hashed: bool,
}

impl Hash for Scored {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if self.hashed {
            self.id.hash(state);
        }
    }
}

impl Individual for Scored {
    fn score(&self) -> f64 {
        self.score
    }
}

/// Returns `size` individuals with distinct scores, in random order.
fn shuffled_individuals(size: usize) -> Vec<Scored> {
    let mut individuals = (0..size)
        .map(|id| Scored {
            id,
            score: id as f64 / size as f64,
            hashed: true,
        })
        .collect::<Vec<_>>();
    Rng::from_seed(0).shuffle(&mut individuals);
    individuals
}

/// Measures the selection work of one generation: creating the sampler of a population and
/// selecting as many pairs of parents as the population has individuals, each pair as
/// `Reproduction::breed` selects it, with the second parent among the other individuals.
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
            let selector = strategy.create_selector::<Scored>().unwrap();
            b.iter(|| {
                let sampler = selector.sampler(&population, 0);
                for _ in 0..population_size {
                    let (first, _) = sampler.select(&mut rng);
                    black_box(sampler.select_other(&mut rng, first));
                }
            });
        });
    }
}

/// Measures creating a population, which validates, sorts, and deduplicates the individuals, for
/// different shapes of the individuals' scores, and for individuals with a constant hash. Also
/// measures computing the statistics of a new population, which happens when they are first
/// requested.
pub fn benchmark_population<M: Measurement>(
    mut group: BenchmarkGroup<'_, M>,
    population_size: usize,
) {
    let distinct = shuffled_individuals(population_size);
    let equal_score = (0..population_size)
        .map(|id| Scored {
            id,
            score: 1.0,
            hashed: true,
        })
        .collect::<Vec<_>>();
    let duplicates = (0..population_size)
        .map(|i| Scored {
            id: i % 10,
            score: (i % 10) as f64,
            hashed: true,
        })
        .collect::<Vec<_>>();
    let constant_hash = distinct
        .iter()
        .map(|individual| Scored {
            hashed: false,
            ..individual.clone()
        })
        .collect::<Vec<_>>();

    group.throughput(Throughput::Elements(population_size as u64));

    for (name, individuals) in [
        ("Distinct score", &distinct),
        ("Equal score", &equal_score),
        ("Duplicates", &duplicates),
        ("Distinct score, constant hash", &constant_hash),
    ] {
        group.bench_function(name, |b| {
            b.iter_batched(
                || individuals.clone(),
                Population::new,
                BatchSize::SmallInput,
            );
        });
    }

    group.bench_function("Statistics", |b| {
        b.iter_batched_ref(
            || Population::new(distinct.clone()).unwrap(),
            |population| *population.stats(),
            BatchSize::SmallInput,
        );
    });
}

/// Measures the steps of a generation that the model and the algorithm provide by default, on
/// the OneMax problem: those of the standard genetic algorithm, the survival policies, and the
/// steps in which the adaptive and diversity-guided algorithms differ from it.
pub fn benchmark_generation_steps<M: Measurement>(
    mut group: BenchmarkGroup<'_, M>,
    genome_length: usize,
    population_size: usize,
) {
    let model = OneMaxEvolutionModel;
    let mut parameters = GeneticAlgorithmParameters::default();
    parameters.population_size = population_size;
    let mut ga = GeneticAlgorithm::new(OneMaxEvolutionModel, parameters).unwrap();
    let mut rng = Rng::from_seed(0);
    let individuals = (0..population_size)
        .map(|_| OneMax::new_random(&genome_length, &mut rng))
        .collect::<Vec<_>>();
    let state = {
        let mut individuals = individuals.clone();
        ga.evaluate_initial(&mut individuals);
        RunState::new(Population::new(individuals).unwrap())
    };
    let population = state.population();
    ga.prepare_generation(&state);

    group.throughput(Throughput::Elements(population_size as u64));

    group.bench_function("evaluate_initial", |b| {
        b.iter_batched_ref(
            || individuals.clone(),
            |individuals| ga.evaluate_initial(individuals),
            BatchSize::SmallInput,
        );
    });

    // One candidate per individual, bred sequentially on one thread.
    group.bench_function("breed", |b| {
        let mut rng = Rng::from_seed(0);
        let reproduction = ga.reproduction();
        let rates = ga.rates();
        let parents = reproduction.selector().sampler(population, 0);
        b.iter(|| {
            for _ in 0..population_size {
                black_box(reproduction.breed(&model, &*parents, rates, &mut rng));
            }
        });
    });

    group.bench_function("generate_offspring", |b| {
        bench_generate_offspring(b, &ga, &state)
    });

    // Each policy chooses the next generation from the population and one candidate per
    // individual.
    let candidates = {
        let mut rng = Rng::from_seed(0);
        let reproduction = ga.reproduction();
        let parents = reproduction.selector().sampler(population, 0);
        (0..population_size)
            .map(|_| reproduction.breed(&model, &*parents, ga.rates(), &mut rng))
            .collect::<Vec<_>>()
    };
    let policies: [(&str, &dyn SurvivalPolicy<OneMax>); 3] = [
        ("FamilyCompetition", &FamilyCompetition),
        ("PlaceCompetition", &PlaceCompetition),
        ("Generational", &Generational),
    ];
    for (name, policy) in policies {
        group.bench_function(format!("select_survivors ({name})"), |b| {
            let mut rng = Rng::from_seed(0);
            b.iter_batched(
                || (candidates.clone(), Generation::new(population_size)),
                |(candidates, mut next)| {
                    policy.select_survivors(population, candidates, &mut next, &mut rng);
                    next
                },
                BatchSize::SmallInput,
            );
        });
    }

    let mut parameters = AdaptiveGeneticAlgorithmParameters::default();
    parameters.population_size = population_size;
    let mut aga = AdaptiveGeneticAlgorithm::new(OneMaxEvolutionModel, parameters).unwrap();
    aga.prepare_generation(&state);

    group.bench_function("generate_offspring (AGA)", |b| {
        bench_generate_offspring(b, &aga, &state)
    });

    // The diversity of the population decides whether the algorithm exploits or explores. A low
    // threshold of 1.0 makes it explore any population.
    let dgga = |low_diversity: f64| {
        let mut parameters = DiversityGuidedGeneticAlgorithmParameters::default();
        parameters.population_size = population_size;
        parameters.low_diversity = low_diversity;
        parameters.high_diversity = parameters.high_diversity.max(low_diversity);
        let mut dgga =
            DiversityGuidedGeneticAlgorithm::new(OneMaxEvolutionModel, parameters).unwrap();
        dgga.prepare_generation(&state);
        dgga
    };
    let mut exploiting = dgga(DiversityGuidedGeneticAlgorithmParameters::default().low_diversity);
    let exploring = dgga(1.0);
    assert!(!exploiting.is_exploring() && exploring.is_exploring());

    group.bench_function("prepare_generation (DGGA)", |b| {
        b.iter(|| {
            exploiting.prepare_generation(black_box(&state));
            exploiting.is_exploring()
        });
    });

    group.bench_function("generate_offspring (DGGA exploiting)", |b| {
        bench_generate_offspring(b, &exploiting, &state)
    });

    group.bench_function("generate_offspring (DGGA exploring)", |b| {
        bench_generate_offspring(b, &exploring, &state)
    });
}

/// Measures breeding a full generation of the population of `state` by `algorithm`, in
/// parallel, and choosing it by the survival policy.
fn bench_generate_offspring<A: EvolutionAlgorithm, M: Measurement>(
    b: &mut Bencher<'_, M>,
    algorithm: &A,
    state: &RunState<IndividualOf<A>>,
) {
    let mut rng = Rng::from_seed(0);
    b.iter_batched_ref(
        || Generation::new(algorithm.population_size()),
        |offspring| algorithm.generate_offspring(state, offspring, &mut rng),
        BatchSize::SmallInput,
    );
}

/// Measures complete runs of the evolutionary algorithms on the OneMax problem, from the same
/// random initial population, up to a generation limit or the model's goal.
pub fn benchmark_evolution<M: Measurement>(
    mut group: BenchmarkGroup<'_, M>,
    genome_length: usize,
    population_size: usize,
    generations: usize,
) {
    let mut initial_population = Vec::with_capacity(population_size);
    {
        let mut rng = Rng::from_seed(0);
        initial_population
            .extend((0..population_size).map(|_| OneMax::new_random(&genome_length, &mut rng)));
    }

    let limits = Limits::new().generations(generations);
    group.sample_size(20);

    group.bench_function("GA", |b| {
        let mut parameters = GeneticAlgorithmParameters::default();
        parameters.population_size = population_size;
        let mut evolution = GeneticAlgorithm::new(OneMaxEvolutionModel, parameters)
            .unwrap()
            .into_evolution()
            .with_seed(0);
        b.iter_batched(
            || initial_population.clone(),
            |individuals| evolution.evolve_population(individuals, |state| limits.check(state)),
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
            |individuals| evolution.evolve_population(individuals, |state| limits.check(state)),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("DGGA", |b| {
        let mut parameters = DiversityGuidedGeneticAlgorithmParameters::default();
        parameters.population_size = population_size;
        // The diversity of the run falls from about 0.5 while it exploits, so it switches to
        // exploring about halfway through.
        parameters.low_diversity = 0.35;
        parameters.high_diversity = 0.4;
        let mut evolution = DiversityGuidedGeneticAlgorithm::new(OneMaxEvolutionModel, parameters)
            .unwrap()
            .into_evolution()
            .with_seed(0);
        // Every run starts by exploiting, so a run that ends exploring has been through both.
        evolution
            .evolve_population(initial_population.clone(), |state| limits.check(state))
            .unwrap();
        assert!(evolution.algorithm().is_exploring());
        b.iter_batched(
            || initial_population.clone(),
            |individuals| evolution.evolve_population(individuals, |state| limits.check(state)),
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
    benchmark_crossover_with_repetition(
        c.benchmark_group("Crossover with repetition (100 genes, 10 copies)"),
        100,
        10,
    );
    benchmark_mutation_strategies(c.benchmark_group("Mutation (1K genes)"), 1000);
    benchmark_operator_helpers(c.benchmark_group("Operator helpers (1K genes)"), 1000);
    benchmark_selection_strategies(c.benchmark_group("Parent selection (100 individuals)"), 100);
    benchmark_selection_strategies(c.benchmark_group("Parent selection (1K individuals)"), 1000);
    benchmark_population(c.benchmark_group("Population (1K individuals)"), 1000);
    benchmark_generation_steps(
        c.benchmark_group("Generation steps (OneMax 100 bits, 100 individuals)"),
        100,
        100,
    );
    benchmark_evolution(
        c.benchmark_group("Evolution (OneMax 100 bits, 100 individuals, up to 20 generations)"),
        100,
        100,
        20,
    );
}

criterion_group!(benches, benchmark);
criterion_main!(benches);

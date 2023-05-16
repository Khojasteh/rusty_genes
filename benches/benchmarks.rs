use criterion::{measurement::*, *};
use rand::prelude::*;
use rusty_genes::*;

pub fn benchmark_crossover_strategies<M: Measurement>(
    mut group: BenchmarkGroup<'_, M>,
    genome_length: usize,
) {
    let mut genome1 = Vec::from_iter(0..genome_length);
    let mut genome2 = Vec::from_iter(0..genome_length);
    {
        let mut rng = SmallRng::seed_from_u64(0);
        genome1.shuffle(&mut rng);
        genome2.shuffle(&mut rng);
    }

    group.throughput(Throughput::Elements(genome_length as u64));

    group.bench_function("GPX2", |b| {
        let mut rng = SmallRng::seed_from_u64(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::GeneralizedPartition2.crossover(&p1, &p2, &mut rng),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("SCX", |b| {
        let mut rng = SmallRng::seed_from_u64(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::SequentialConstructive.crossover(&p1, &p2, &mut rng),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("ERX", |b| {
        let mut rng = SmallRng::seed_from_u64(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::EdgeRecombination.crossover(&p1, &p2, &mut rng),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("PMX", |b| {
        let mut rng = SmallRng::seed_from_u64(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::PartiallyMapped.crossover(&p1, &p2, &mut rng),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("OX", |b| {
        let mut rng = SmallRng::seed_from_u64(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::Order.crossover(&p1, &p2, &mut rng),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("CX", |b| {
        let mut rng = SmallRng::seed_from_u64(0);
        b.iter_batched(
            || (genome1.clone(), genome2.clone()),
            |(p1, p2)| CrossoverStrategy::Cycle.crossover(&p1, &p2, &mut rng),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("SPX", |b| {
        let mut rng = SmallRng::seed_from_u64(0);
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
        let mut rng = SmallRng::seed_from_u64(0);
        let mut genome = Vec::from_iter(0..genome_length);
        b.iter(|| MutationStrategy::Swap.mutate(&mut genome, &mut rng));
        black_box(genome);
    });

    group.bench_function("Duplicate", |b| {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut genome = Vec::from_iter(0..genome_length);
        b.iter(|| MutationStrategy::Duplicate.mutate(&mut genome, &mut rng));
        black_box(genome);
    });

    group.bench_function("Insertion", |b| {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut genome = Vec::from_iter(0..genome_length);
        b.iter(|| MutationStrategy::Insertion.mutate(&mut genome, &mut rng));
        black_box(genome);
    });

    group.bench_function("Inversion", |b| {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut genome = Vec::from_iter(0..genome_length);
        b.iter(|| MutationStrategy::Inversion.mutate(&mut genome, &mut rng));
        black_box(genome);
    });

    group.bench_function("Scramble", |b| {
        let mut rng = SmallRng::seed_from_u64(0);
        let mut genome = Vec::from_iter(0..genome_length);
        b.iter(|| MutationStrategy::Scramble.mutate(&mut genome, &mut rng));
        black_box(genome);
    });
}

pub fn benchmark(c: &mut Criterion) {
    c.bench_function("Stats (1M samples)", |b| {
        b.iter(|| Stats::from_iter((0..1000000).map(|i| i as f64)))
    });

    benchmark_crossover_strategies(c.benchmark_group("Crossover Strategies (1K genes)"), 1000);
    benchmark_mutation_strategies(c.benchmark_group("Mutation Strategies (1K genes)"), 1000);
}

criterion_group!(benches, benchmark);
criterion_main!(benches);

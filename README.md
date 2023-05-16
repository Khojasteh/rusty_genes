# RustyGenes

RustyGenes is a Rust library designed to facilitate the implementation and experimentation with evolutionary algorithms, especially those focused on permutation-based problems. This library is not intended for simple drop-in usage but rather provides a set of modular components that can be customized and assembled to match the requirements of your specific problem. It encourages the user to actively participate in the construction of their algorithm, promoting flexibility and extensibility.

One of the key features of RustyGenes is its support for creating custom algorithms by implementing a few core methods. This extensibility allows users to experiment with novel evolutionary strategies and techniques, pushing the boundaries of what's possible with genetic algorithms.

RustyGenes provides the fundamental components of an evolutionary algorithm as separate modules, allowing for a high level of customization. These components include a variety of selection strategies, crossover strategies, and mutation strategies, as well as the structure for defining evolutionary models and individuals.

The library supports only maximization problems. If your use case involves minimization, you would need to convert it into a maximization problem by adjusting the fitness function or the problem representation.

## Features

The library currently implements two evolutionary algorithms:
- Standard Genetic Algorithm (GA)
- Adaptive Genetic Algorithm: This algorithm dynamically adjusts the crossover and mutation rates based on the calculated diversity value. This promotes the exploitation of existing diversity through crossover operations when diversity is high, and encourages exploration by introducing and maintaining more diversity in the population when diversity is low.

RustyGenes supports the following selection strategies:
- Tournament Selection
- Roulette Wheel Selection
- Boltzmann Selection
- Rank Selection
- Linear Selection
- Elitist Selection

It also provides a variety of crossover strategies:
- Generalized Partition Crossover 2 (GPX2)
- Sequential Constructive Crossover (SCX)
- Edge Recombination Crossover (ERX)
- Partially Mapped Crossover (PMX)
- Order Crossover (OX)
- Cycle Crossover (CX)
- Single-Point Crossover (SPX)

The mutation strategies supported by RustyGenes include:
- Swap Mutation
- Scramble Mutation
- Inversion Mutation
- Insertion Mutation
- Duplicate Mutation

## Usage

Here is an example of how to use the library to solve the OneMax problem:

```rust
use rusty_genes::*;
use rand::{distributions::Standard, prelude::*};

/// Represents an individual in the OneMax problem.
///
/// The `genome` is a binary string represented as a vector of booleans. Each boolean
/// represents a bit, and the fitness of an individual is the ratio of 'true' bits to
/// the total number of bits in this binary string.
#[derive(Debug, Clone, PartialEq)]
pub struct OneMax {
    genome: Vec<bool>,
    fitness: f64,
}

impl OneMax {
    // Returns the binary string of the individual.
    pub fn genome(&self) -> &[bool] {
        &self.genome
    }
}

/// Conversion from a `Vec<bool>` to a `OneMax` individual.
///
/// This allows a binary string to be directly converted into an individual.
impl From<Vec<bool>> for OneMax {
    fn from(genome: Vec<bool>) -> Self {
        Self {
            genome,
            fitness: 0.0,
        }
    }
}

/// Implements the method required by the [`Individual`] trait for the OneMax problem.
impl Individual for OneMax {
    /// Returns the fitness of the individual.
    fn fitness(&self) -> f64 {
        self.fitness
    }
}

/// Allows the creation of a new random individual for the OneMax problem.
impl RandomIndividual<usize> for OneMax {
    /// Generates a new random individual for the OneMax problem.
    ///
    /// The `length` parameter determines the length of the binary string (i.e., the genome),
    /// and the `rng` parameter is a random number generator used to generate the binary string.
    fn new_random<R: Rng>(length: &usize, rng: &mut R) -> Self {
        let bits = rng.sample_iter(Standard);
        let genome = Vec::from_iter(bits.take(*length));
        Self::from(genome)
    }
}

/// The evolutionary model for the OneMax problem.
///
/// This struct provides methods for crossover, mutation, and evaluation operations specific
/// to the OneMax problem.
#[derive(Debug, Default)]
pub struct OneMaxEvolutionModel;

/// Implements the methods required by the [`EvolutionModel`] trait for the OneMax problem.
impl EvolutionModel for OneMaxEvolutionModel {
    type Individual = OneMax;
    type Rng = SmallRng;

    /// Creates a new random number generator from entropy.
    fn rng() -> Self::Rng {
        SmallRng::from_entropy()
    }

    /// Performs a single point crossover on the parents' genomes to create a new individual.
    fn crossover(
        &self,
        parent1: &Self::Individual,
        parent2: &Self::Individual,
        rng: &mut Self::Rng,
    ) -> Self::Individual {
        CrossoverStrategy::SinglePoint
            .crossover(&parent1.genome, &parent2.genome, rng)
            .into()
    }

    /// Mutates the individual's genome by using the duplicate mutation strategy.
    fn mutate(&self, individual: &mut Self::Individual, rng: &mut Self::Rng) {
        MutationStrategy::Duplicate.mutate(&mut individual.genome, rng);
    }

    /// Evaluates the individual's fitness by calculating the ratio of `true` bits to the total
    /// number of bits in its genome.
    fn evaluate(&self, individual: &mut Self::Individual) {
        let one_bits = individual.genome.iter().filter(|&&one| one).count();
        let total_bits = individual.genome.len();
        individual.fitness = one_bits as f64 / total_bits as f64;
    }
}

// Create an instance of the evolutionary model.
let one_max = OneMaxEvolutionModel;

// Use the Genetic Algorithm to solve the problem.
let ga = GeneticAlgorithm::build_evolution(one_max);

// Use default parameters of the Genetic Algorithm.
let ga_params = GeneticAlgorithmParameters::default();

// Run the algorithm until the fitness of the fittest individual is close to 1 or
// the number of generations exceeds 100.
let solution = ga.run(&ONE_MAX_LENGTH, &ga_params, |population| {
    population.fittest().fitness() < 0.999999 && population.generation() <= 100
});

// Verify that the solution is correct.
assert!(solution.genome().iter().all(|&one| one == true));
```

## Acknowledgement

This project was a collaborative effort. OpenAI's AI, [ChatGPT-4](https://openai.com/product/gpt-4), assisted with both the documentation and the implementation.

## License

RustyGenes is licensed under the [MIT License](LICENSE).

This project is still in its early stages and is open to contributions. Please feel free to raise issues or submit pull requests.

## Disclaimer

This is the initial release and may be subject to change as the project evolves. Please use it at your own risk.
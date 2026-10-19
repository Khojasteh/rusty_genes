use rusty_genes::*;
use std::hash::{Hash, Hasher};

/// A solution is a genome; its cached evaluation is not part of its identity.
#[derive(Debug, Clone)]
pub struct BitGenome {
    pub genes: Vec<bool>,
    score: f64,
}

impl From<Vec<bool>> for BitGenome {
    fn from(genes: Vec<bool>) -> Self {
        Self {
            genes,
            score: f64::NAN,
        }
    }
}

impl PartialEq for BitGenome {
    fn eq(&self, other: &Self) -> bool {
        self.genes == other.genes
    }
}

impl Hash for BitGenome {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.genes.hash(state);
    }
}

impl Individual for BitGenome {
    fn score(&self) -> f64 {
        self.score
    }
}

impl RandomIndividual<Vec<bool>> for BitGenome {
    fn new_random(genes: &Vec<bool>, rng: &mut Rng) -> Self {
        let mut genes = genes.clone();
        rng.shuffle(&mut genes);
        genes.into()
    }
}

/// A weighted binary optimization problem used across the engine's contract tests.
pub struct WeightedBits {
    pub weights: Vec<f64>,
    pub objective: Objective,
    pub goal: Option<f64>,
}

impl WeightedBits {
    pub fn new(weights: Vec<f64>, objective: Objective) -> Self {
        Self {
            weights,
            objective,
            goal: None,
        }
    }

    pub fn individual(&self, genes: &[bool]) -> BitGenome {
        let mut individual = BitGenome::from(genes.to_vec());
        self.evaluate(&mut individual);
        individual
    }
}

/// Measures how far the genomes lie from their average point, taking each bit as a coordinate.
impl DiversityModel for WeightedBits {
    fn diversity(&self, population: &Population<BitGenome>) -> f64 {
        let length = population[0].genes.len();
        let bits = population
            .iter()
            .map(|individual| individual.genes.iter().map(|&bit| f64::from(u8::from(bit))));
        distance_to_average_point(bits, (length as f64).sqrt())
    }
}

impl EvolutionModel for WeightedBits {
    type Individual = BitGenome;
    fn objective(&self) -> Objective {
        self.objective
    }
    fn goal_score(&self, _: &Population<BitGenome>) -> Option<f64> {
        self.goal
    }
    fn crossover(&self, first: &BitGenome, second: &BitGenome, rng: &mut Rng) -> BitGenome {
        CrossoverStrategy::SinglePoint
            .crossover(&first.genes, &second.genes, rng)
            .into()
    }
    fn mutate(&self, individual: &mut BitGenome, rng: &mut Rng) {
        if !individual.genes.is_empty() {
            let index = rng.random_range(0..individual.genes.len());
            individual.genes[index] = !individual.genes[index];
            individual.score = f64::NAN;
        }
    }
    fn evaluate(&self, individual: &mut BitGenome) {
        individual.score =
            if individual.genes.len() == self.weights.len() && !individual.genes.is_empty() {
                individual
                    .genes
                    .iter()
                    .zip(&self.weights)
                    .filter_map(|(&on, &weight)| on.then_some(weight))
                    .sum()
            } else {
                f64::NAN
            };
    }
}

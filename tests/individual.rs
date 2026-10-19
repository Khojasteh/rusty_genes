use rusty_genes::*;
use std::hash::{BuildHasher, Hash, Hasher, RandomState};

#[test]
fn floating_point_hashes_agree_with_equality() {
    let hasher = RandomState::new();
    let hash = |value| hasher.hash_one(HashedF64(value));
    assert_eq!(hash(0.0), hash(-0.0));
    assert_eq!(hash(1.5), hash(1.5));

    let hash_slices = |values: &[&[f64]]| {
        let mut state = hasher.build_hasher();
        for values in values {
            hash_f64_slice(values, &mut state);
        }
        state.finish()
    };
    assert_eq!(hash_slices(&[&[0.0, 1.0]]), hash_slices(&[&[-0.0, 1.0]]));
    assert_ne!(
        hash_slices(&[&[1.0, 2.0], &[3.0]]),
        hash_slices(&[&[1.0], &[2.0, 3.0]])
    );
}

struct HashedF64(f64);

impl Hash for HashedF64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_f64(self.0, state);
    }
}

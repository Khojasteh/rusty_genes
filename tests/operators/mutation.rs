use rusty_genes::*;

/// Returns the position a gene was removed from and the position it was inserted at, if
/// `mutated` is `original` with one gene moved.
fn moved_gene(original: &[usize], mutated: &[usize]) -> Option<(usize, usize)> {
    let start = (0..original.len()).position(|i| original[i] != mutated[i])?;
    let end = (0..original.len()).rposition(|i| original[i] != mutated[i])?;
    let mut leftwards = original[start..=end].to_vec();
    leftwards.rotate_right(1);
    let mut rightwards = original[start..=end].to_vec();
    rightwards.rotate_left(1);
    if mutated[start..=end] == leftwards[..] {
        Some((end, start))
    } else if mutated[start..=end] == rightwards[..] {
        Some((start, end))
    } else {
        None
    }
}

#[test]
fn insertion_moves_any_gene_in_either_direction() {
    let original = (0..5).collect::<Vec<_>>();
    let mut rng = Rng::from_seed(0);
    let mut moves = Vec::new();
    for _ in 0..1000 {
        let mut genome = original.clone();
        insertion_mutation(&mut genome, &mut rng);
        let moved = moved_gene(&original, &genome).expect("one gene is moved");
        if !moves.contains(&moved) {
            moves.push(moved);
        }
    }
    // Every gene can move to every other position. Moving a gene one position to the right gives
    // the same genome as moving its neighbor one position to the left, which is how it is found.
    moves.sort_unstable();
    let mut expected = (0..5)
        .flat_map(|from| (0..5).map(move |to| (from, to)))
        .filter(|&(from, to)| from > to || to > from + 1)
        .collect::<Vec<_>>();
    expected.sort_unstable();
    assert_eq!(moves, expected);
}

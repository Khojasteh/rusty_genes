use rusty_genes::*;
use std::{
    hash::{Hash, Hasher},
    ops::ControlFlow,
    time::Duration,
};

#[derive(Debug, Clone, PartialEq)]
struct Mock {
    id: u32,
    score: f64,
}

impl Hash for Mock {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Individual for Mock {
    fn score(&self) -> f64 {
        self.score
    }
}

fn mock(id: u32, score: f64) -> Mock {
    Mock { id, score }
}

fn state(individuals: Vec<Mock>, objective: Objective) -> RunState<Mock> {
    RunState::new(Population::with_objective(individuals, objective).unwrap())
}

fn advance(state: &mut RunState<Mock>, individuals: Vec<Mock>) {
    state
        .advance(|population| population.replace(&mut individuals.clone()))
        .unwrap();
}

#[test]
fn a_new_state_starts_at_generation_zero_with_its_fittest_individual() {
    let state = state(vec![mock(1, 2.0), mock(2, 1.0)], Objective::Maximize);
    assert_eq!((state.generation(), state.stagnation()), (0, 0));
    assert_eq!(state.fittest_so_far().id, 1);
}

#[test]
fn advance_tracks_generation_and_stagnation() {
    let mut state = state(vec![mock(1, 1.0)], Objective::Maximize);

    advance(&mut state, vec![mock(2, 1.0)]);
    assert_eq!((state.generation(), state.stagnation()), (1, 1));

    advance(&mut state, vec![mock(3, 2.0)]);
    assert_eq!((state.generation(), state.stagnation()), (2, 0));
}

#[test]
fn advance_counts_stagnation_from_the_best_score_ever_reached() {
    let mut state = state(vec![mock(1, 2.0)], Objective::Maximize);

    // The fittest individual is lost, and then an equally fit one is found again.
    advance(&mut state, vec![mock(2, 1.0)]);
    advance(&mut state, vec![mock(3, 2.0)]);
    assert_eq!(state.stagnation(), 2);

    advance(&mut state, vec![mock(4, 3.0)]);
    assert_eq!(state.stagnation(), 0);
}

#[test]
fn fittest_so_far_is_the_fittest_individual_of_any_generation() {
    let mut state = state(vec![mock(1, 2.0), mock(2, 1.0)], Objective::Maximize);

    advance(&mut state, vec![mock(3, 1.5)]);
    assert_eq!(state.population().fittest().id, 3);
    assert_eq!(state.fittest_so_far().id, 1);

    // An equally fit individual does not replace the one found first.
    advance(&mut state, vec![mock(4, 2.0)]);
    assert_eq!(state.fittest_so_far().id, 1);

    advance(&mut state, vec![mock(5, 2.5)]);
    assert_eq!(state.into_fittest_so_far().id, 5);
}

#[test]
fn improvement_follows_the_objective_of_the_population() {
    let mut state = state(vec![mock(1, 10.0)], Objective::Minimize);

    advance(&mut state, vec![mock(2, 15.0)]);
    assert_eq!((state.stagnation(), state.fittest_so_far().id), (1, 1));

    advance(&mut state, vec![mock(3, -5.0)]);
    assert_eq!((state.stagnation(), state.fittest_so_far().id), (0, 3));
}

#[test]
fn a_failed_advance_leaves_the_state_unchanged() {
    let mut state = state(vec![mock(1, 1.0)], Objective::Maximize);
    let elapsed = state.elapsed();

    let result = state.advance(|population| population.replace(&mut vec![mock(2, f64::NAN)]));

    assert_eq!(result, Err(EvolutionError::EmptyPopulation));
    assert_eq!((state.generation(), state.stagnation()), (0, 0));
    assert_eq!(state.elapsed(), elapsed);
    assert_eq!(state.population().fittest().id, 1);
}

#[test]
fn elapsed_time_is_that_of_the_current_generation() {
    let mut state = state(vec![mock(1, 1.0)], Objective::Maximize);
    let first = state.elapsed();
    std::thread::sleep(Duration::from_millis(5));
    assert_eq!(state.elapsed(), first);

    advance(&mut state, vec![mock(2, 1.0)]);
    assert!(state.elapsed() >= first + Duration::from_millis(5));
}

#[test]
fn limits_stop_once_their_counts_are_reached() {
    let mut state = state(vec![mock(1, 1.0)], Objective::Maximize);
    let generations = Limits::new().generations(2);
    let stagnation = Limits::new().stagnation(2);

    advance(&mut state, vec![mock(2, 1.0)]);
    assert_eq!(generations.check(&state), ControlFlow::Continue(()));
    assert_eq!(stagnation.check(&state), ControlFlow::Continue(()));

    advance(&mut state, vec![mock(3, 1.0)]);
    assert_eq!(
        generations.check(&state),
        ControlFlow::Break(StopReason::GenerationLimit)
    );
    assert_eq!(
        stagnation.check(&state),
        ControlFlow::Break(StopReason::StagnationLimit)
    );
}

#[test]
fn limits_report_the_generation_limit_before_the_stagnation_and_time_limits() {
    let mut state = state(vec![mock(1, 1.0)], Objective::Maximize);
    advance(&mut state, vec![mock(2, 1.0)]);
    let all = Limits::new()
        .generations(1)
        .stagnation(1)
        .time(Duration::ZERO);
    assert_eq!(
        all.check(&state),
        ControlFlow::Break(StopReason::GenerationLimit)
    );
    assert_eq!(
        all.generations(2).check(&state),
        ControlFlow::Break(StopReason::StagnationLimit)
    );
    assert_eq!(
        all.generations(2).stagnation(2).check(&state),
        ControlFlow::Break(StopReason::TimeLimit)
    );
}

#[test]
fn unset_limits_never_stop_a_run() {
    let state = state(vec![mock(1, 1.0)], Objective::Maximize);
    assert_eq!(Limits::new().check(&state), ControlFlow::Continue(()));
    assert_eq!(
        Limits::new().time(Duration::from_secs(3600)).check(&state),
        ControlFlow::Continue(())
    );
}

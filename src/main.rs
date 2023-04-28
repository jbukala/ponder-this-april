extern crate ndarray;
use ndarray::*;

extern crate ndarray_stats;
use rand_distr::{Geometric, Distribution};

// Flip a switch and all switches in the same row and column
fn flip_switch(board: Array2<bool>, row: usize, col: usize) -> Array2<bool> {
    let mut board = board.clone();
    board.row_mut(row).mapv_inplace(|x| !x);
    board.column_mut(col).mapv_inplace(|x| !x);
    board[[row, col]] = !board[[row, col]];
    board
}

// Calculate the (binary) entropy of the board
fn board_entropy(board: &Array2<bool>) -> f64 {

    // Count the amount of true's in the board
    let num_true: u8 = board.iter().map(|x| match *x {
        true => 1,
        false => 0,
    } as u8).sum::<u8>();

    let p = (num_true as f64 / board.len() as f64) as f64;
    let entropy = -p * p.log2() - (1.0 - p) * ((1.0 - p) as f64).log2();
    entropy
}

fn find_all_move_entropies(board: &Array2<bool>) -> Vec<(usize, usize, f64)> {
    let mut entropies = Vec::new();
    for row in 0..board.shape()[0] {
        for col in 0..board.shape()[1] {
            let new_board = flip_switch(board.clone(), row, col);
            let entropy = board_entropy(&new_board);
            entropies.push((row, col, entropy));
        }
    }
    entropies
}

fn sample_low_entropy_move(board: &Array2<bool>) -> (usize, usize, f64) {
    let mut entropies = find_all_move_entropies(board);
    entropies.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    // Now make a Geometric distribution and sample from it to find the top-n lowest entropy move
    let geometric = Geometric::new(0.7).unwrap();
    let mut rng = rand::thread_rng();

    // Sample until small enough to be a viable option
    let mut n = geometric.sample(&mut rng) as usize;
    while n > (board.len()-1) {
        n = geometric.sample(&mut rng) as usize;
    }

    let (row, col, ent) = entropies.get(n).unwrap();
    (*row, *col, *ent)
}

fn minimize_board_entropy(mut board: Array2<bool>, max_steps: usize) -> (Array2<bool>, Vec<(usize, usize)>) {
    // Keep stepping through entropy minimization steps until we reach max_steps
    let mut min_ent = 1.0;
    let mut min_ent_board = board.clone();
    let mut min_ent_moves: Vec<(usize, usize)> = Vec::new();

    let mut moves: Vec<(usize, usize)> = Vec::new();
    for _i in 0..max_steps {
        let (row, col, ent) = sample_low_entropy_move(&board);
        board = flip_switch(board, row, col);
        moves.push((row, col));

        // Save the board with the lowest entropy
        if ent < min_ent {
            min_ent = ent;
            min_ent_board = board.clone();
            min_ent_moves = moves.clone();
        }
        //println!("Step: {}, Entropy: {}", i, board_entropy(&board));
    }
    (min_ent_board, min_ent_moves)
}

fn main() {

    let board = array![
    [false, false, true, true], 
    [true, true, false, true],
    [false, true, true, false], 
    [false, false, false, true]
    ];

    println!("Initial board: {:?}", board);
    println!("Board entropy: {}", board_entropy(&board));

    let (best_board, _best_moves) = minimize_board_entropy(board.clone(), 100000);

    println!("Best board: {:?}", best_board);
    println!("Board entropy: {}", board_entropy(&best_board));
    //println!("Best moves: {:?}", best_moves);
}
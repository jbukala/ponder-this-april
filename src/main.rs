extern crate ndarray;
use ndarray::*;

extern crate ndarray_stats;
use rand_distr::{Geometric, Distribution};

use std::env;
use std::process;

// Read a csv file and return a 2D array
fn read_csv(filename: &str, puzzle_size: usize) -> Array2<bool> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(filename).unwrap();
    
    let mut board = Array2::<bool>::default((puzzle_size, puzzle_size));
    for (i, result) in reader.records().enumerate() {
        let record = result.unwrap();
        for (j, value) in record.iter().enumerate() {
            let val = match value {
                "0" => false,
                "1" => true,
                _ => panic!("Invalid value in csv file"),
            };
            board[[i, j]] = val;
        }
    }
    board
}

fn print_board(board: &Array2<bool>) {
    for row in board.rows() {
        for val in row {
            if *val {
                print!("X");
            } else {
                print!("0");
            }
        }
        println!();
    }
}

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
    let num_true = board.iter().filter(|x| **x).count();
    let p = (num_true as f64 / board.len() as f64) as f64;
    let entropy = -p * p.log2() - (1.0 - p) * ((1.0 - p) as f64).log2();
    entropy
}

// Try all different possible moves and return the entropy of the board after each one
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

fn parse_config(args: &[String]) -> Result<(&str, usize, usize), &str> {
    let num_args = args.len() - 1;
    if num_args == 0 {
        return Ok(("./data/small_puzzle.csv", 4, 1000));
    }
    else if num_args > 0 && num_args < 3 {
        return Err("Not enough arguments");
    }
    else if num_args == 3 {
        let puzzle_file_path = &args[1];
        let puzzle_size = args[2].parse::<usize>().unwrap();
        let max_steps = args[3].parse::<usize>().unwrap();

        return Ok((puzzle_file_path, puzzle_size, max_steps));
    }
    else {
        return Err("Too many arguments");
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let (puzzle_file_path, puzzle_size, max_steps) = parse_config(&args).unwrap_or_else(|err| {
        println!("Problem parsing arguments: {err}");
        process::exit(1);
    });
    let board = read_csv(puzzle_file_path, puzzle_size);

    println!("Initial board:");
    print_board(&board);
    println!("Board entropy: {}", board_entropy(&board));

    let (best_board, _best_moves) = minimize_board_entropy(board.clone(), max_steps);

    println!("Best board:");
    print_board(&best_board);
    println!("Board entropy: {}", board_entropy(&best_board));
    //println!("Best moves: {:?}", best_moves);
}
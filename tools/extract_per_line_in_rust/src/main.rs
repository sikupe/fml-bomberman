use std::fs::{File, canonicalize};
use std::io::{self, BufRead};
use std::path::Path;
use indicatif::ProgressBar;
use ndarray::prelude::*;
use ndarray_npy::write_npy;
use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// path to list file
    #[clap(short, long)]
    input: String,

    /// path to destination (arr.npy)
    #[clap(short, long)]
    destination: String,
}

fn main() {
    let args = Args::parse();
    let input = canonicalize(args.input).expect("Could not canonicalize");
    let destination = args.destination;
    if !Path::new(&input).exists() {
        eprintln!("Input path does not exist");
        return
    }
    if Path::new(&destination).exists() {
        eprintln!("destination already exists");
        return
    }

    // File hosts must exist in current path before this produces output
    if let Ok(lines) = read_lines(input) {
        // Consumes the iterator, returns an (Optional) String

        let bar = ProgressBar::new(10000);
        let mut arr = Array::<i64, Ix1>::zeros(10000);
        for (i, line) in lines.enumerate() {
            bar.inc(1);
            if let Ok(content) = line {
                let split = content.split(",");
                let mut count = 0;
                for item in split {
                    count += item.parse::<i64>().expect("Failed to parse");
                    arr[i] = count;
                }
            }
        }
        write_npy(destination, &arr).expect("Failed to write");
        bar.finish();
    }
}

// The output is wrapped in a Result to allow matching on errors
// Returns an Iterator to the Reader of the lines of the file.
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

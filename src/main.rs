mod cli;
use log::LevelFilter;

fn main() {
    let args = cli::parse();

    let log_level = match args.verbosity {
        0 => LevelFilter::Info,
        1 => LevelFilter::Debug,
        _ => LevelFilter::Trace,
    };

    env_logger::Builder::new().filter_level(log_level).init();

    if let Err(e) = CRB::run(args.config) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

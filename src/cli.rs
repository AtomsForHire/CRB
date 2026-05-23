use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct Args {
    /// Path to the TOML config file
    pub config: PathBuf,

    #[clap(short, long, action = clap::ArgAction::Count)]
    // #[clap(global = true)]
    pub(crate) verbosity: u8,
}

pub fn parse() -> Args {
    Args::parse()
}

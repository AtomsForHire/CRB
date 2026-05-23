use ndarray::{Array1, Array2};
use std::fmt::Display;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

pub trait WriteToFile {
    fn write_to_file(&self, path: &Path) -> io::Result<()>;
}

impl<T: Display> WriteToFile for Array1<T> {
    fn write_to_file(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        for val in self.iter() {
            writeln!(writer, "{}", val)?;
        }
        Ok(())
    }
}

impl<T: Display> WriteToFile for Array2<T> {
    fn write_to_file(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        for row in self.rows() {
            let line = row
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(writer, "{}", line)?;
        }
        Ok(())
    }
}

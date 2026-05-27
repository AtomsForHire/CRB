use ndarray::{Array1, Array2};
use std::fmt::Display;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use kuva::plot::{Heatmap, heatmap, surface3d};
use kuva::prelude::*;
use kuva::render::layout::Layout;
use kuva::render::plots::Plot;
use kuva::render::render::render_multiple;

pub trait WriteToFile {
    fn write_to_file(&self, path: &Path) -> io::Result<()>;
}

pub trait SaveFigure {
    fn save_to_heatmap(&self, title: String, path: &Path) -> io::Result<()>;
    fn save_to_surface(&self, title: String, path: &Path) -> io::Result<()>;
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

impl<T: Display + Copy + Clone + Into<f64>> SaveFigure for Array2<T> {
    fn save_to_surface(&self, title: String, path: &Path) -> io::Result<()> {
        let n_rows = self.nrows();
        let n_cols = self.ncols();
        // Convert to Vec<Vec<T>> for Kuva
        let nested_data: Vec<Vec<f64>> = self
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|&val| val.into()).collect::<Vec<f64>>())
            .collect();

        // I need nested_data to be f64 so satisfy the new() function
        let surface = Surface3DPlot::new(nested_data)
            .with_z_colormap(ColorMap::Viridis)
            .with_no_wireframe();
        let plots = vec![Plot::Surface3D(surface)];
        let layout = Layout::auto_from_plots(&plots)
            .with_title(title)
            .with_ticks(8);
        let scene = render_to_png(plots, layout, 2.0).expect("Couldn't plot figure");

        std::fs::write(path, scene)?;
        Ok(())
    }

    fn save_to_heatmap(&self, title: String, path: &Path) -> io::Result<()> {
        let n_rows = self.nrows();
        let n_cols = self.ncols();
        // Convert to Vec<Vec<T>> for Kuva
        let nested_data: Vec<Vec<T>> = self.rows().into_iter().map(|row| row.to_vec()).collect();

        // let surface = surface3d::Surface3DPlot::new();
        let heatmap = Heatmap::new()
            .with_data(nested_data)
            .with_cell_size(1.0)
            .with_x_range(0, n_cols as f64)
            .with_y_range(0, n_rows as f64);

        let plots = vec![Plot::Heatmap(heatmap)];
        let layout = Layout::auto_from_plots(&plots)
            .with_title(title)
            .with_ticks(10);

        let scene = render_to_png(plots, layout, 2.0).expect("Couldn't plot figure");

        std::fs::write(path, scene)?;
        Ok(())
    }
}

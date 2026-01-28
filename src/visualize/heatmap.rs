// @module: crate::visualize::heatmap
// @status: stable
// @owner: code_expert
// @feature: visualize
// @depends: [crate::error, crate::visualize::quantum, ndarray, plotters]
// @tests: [unit]

//! Heatmap visualization for 2D data.

use ndarray::Array2;
use plotters::prelude::*;
use std::path::Path;

use crate::error::{CirculantError, Result};

use super::quantum::PlotConfig;

/// Plot a 2D array as a heatmap.
///
/// # Arguments
///
/// * `data` - 2D array of values to visualize
/// * `output_path` - Path to save the plot
/// * `config` - Plot configuration
///
/// # Errors
///
/// Returns `VisualizationError` if the plot cannot be created.
#[cfg(any(feature = "visualize-bitmap", feature = "visualize-svg"))]
pub fn plot_heatmap<P: AsRef<Path>>(
    data: &Array2<f64>,
    output_path: P,
    config: &PlotConfig,
) -> Result<()> {
    let path = output_path.as_ref();

    #[cfg(feature = "visualize-bitmap")]
    {
        if path.extension().is_some_and(|e| e == "png") {
            return plot_heatmap_bitmap(data, path, config);
        }
    }

    #[cfg(feature = "visualize-svg")]
    {
        if path.extension().is_some_and(|e| e == "svg") {
            return plot_heatmap_svg(data, path, config);
        }
    }

    Err(CirculantError::VisualizationError(
        "unsupported output format".to_string(),
    ))
}

#[cfg(feature = "visualize-bitmap")]
fn plot_heatmap_bitmap<P: AsRef<Path>>(
    data: &Array2<f64>,
    output_path: P,
    config: &PlotConfig,
) -> Result<()> {
    let (rows, cols) = data.dim();
    let root =
        BitMapBackend::new(output_path.as_ref(), (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max_val - min_val).abs() < 1e-10 {
        1.0
    } else {
        max_val - min_val
    };

    let mut chart = ChartBuilder::on(&root)
        .caption(&config.title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..cols, 0..rows)
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc(&config.x_label)
        .y_desc(&config.y_label)
        .draw()
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    // Draw heatmap cells
    for i in 0..rows {
        for j in 0..cols {
            let val = data[[i, j]];
            let normalized = (val - min_val) / range;
            // Blue to red colormap
            let r = (normalized * 255.0) as u8;
            let b = ((1.0 - normalized) * 255.0) as u8;
            let color = RGBColor(r, 0, b);

            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(j, rows - 1 - i), (j + 1, rows - i)],
                    color.filled(),
                )))
                .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;
        }
    }

    root.present()
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    Ok(())
}

#[cfg(feature = "visualize-svg")]
fn plot_heatmap_svg<P: AsRef<Path>>(
    data: &Array2<f64>,
    output_path: P,
    config: &PlotConfig,
) -> Result<()> {
    let (rows, cols) = data.dim();
    let root =
        SVGBackend::new(output_path.as_ref(), (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max_val - min_val).abs() < 1e-10 {
        1.0
    } else {
        max_val - min_val
    };

    let mut chart = ChartBuilder::on(&root)
        .caption(&config.title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..cols, 0..rows)
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc(&config.x_label)
        .y_desc(&config.y_label)
        .draw()
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    for i in 0..rows {
        for j in 0..cols {
            let val = data[[i, j]];
            let normalized = (val - min_val) / range;
            let r = (normalized * 255.0) as u8;
            let b = ((1.0 - normalized) * 255.0) as u8;
            let color = RGBColor(r, 0, b);

            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(j, rows - 1 - i), (j + 1, rows - i)],
                    color.filled(),
                )))
                .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;
        }
    }

    root.present()
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    Ok(())
}

/// Stub for plot_heatmap when no backend is available
#[cfg(not(any(feature = "visualize-bitmap", feature = "visualize-svg")))]
pub fn plot_heatmap<P: AsRef<Path>>(
    _data: &Array2<f64>,
    _output_path: P,
    _config: &PlotConfig,
) -> Result<()> {
    Err(CirculantError::VisualizationError(
        "no visualization backend enabled - enable visualize-bitmap or visualize-svg feature"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_config_for_heatmap() {
        let config = PlotConfig::with_title("Heatmap").labels("X", "Y");
        assert_eq!(config.title, "Heatmap");
        assert_eq!(config.x_label, "X");
        assert_eq!(config.y_label, "Y");
    }
}

// @module: crate::visualize::quantum
// @status: stable
// @owner: code_expert
// @feature: visualize
// @depends: [crate::error, plotters]
// @tests: [unit]

//! Quantum state visualization utilities.

use plotters::prelude::*;
use std::path::Path;
use std::sync::Once;

use crate::error::{CirculantError, Result};

/// Initialize fonts for plotting (only needs to be done once).
#[cfg(any(feature = "visualize-bitmap", feature = "visualize-svg"))]
fn ensure_font_registered() {
    static FONT_INIT: Once = Once::new();
    FONT_INIT.call_once(|| {
        // Try to load a common system font
        let font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ];

        for path in &font_paths {
            if let Ok(font_data) = std::fs::read(path) {
                // Leak the font data to give it 'static lifetime
                // This is safe since we only do it once per program execution
                let font_data: &'static [u8] = Box::leak(font_data.into_boxed_slice());
                let _ = plotters::style::register_font(
                    "sans-serif",
                    plotters::style::FontStyle::Normal,
                    font_data,
                );
                return;
            }
        }
    });
}

/// Configuration for plot generation.
#[derive(Clone, Debug)]
pub struct PlotConfig {
    /// Plot title.
    pub title: String,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// X-axis label.
    pub x_label: String,
    /// Y-axis label.
    pub y_label: String,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            title: String::new(),
            width: 800,
            height: 600,
            x_label: "Position".to_string(),
            y_label: "Probability".to_string(),
        }
    }
}

impl PlotConfig {
    /// Create a new plot configuration with title.
    pub fn with_title(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Default::default()
        }
    }

    /// Set the plot dimensions.
    pub fn dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set axis labels.
    pub fn labels(mut self, x_label: impl Into<String>, y_label: impl Into<String>) -> Self {
        self.x_label = x_label.into();
        self.y_label = y_label.into();
        self
    }
}

/// Plot a 1D probability distribution to a file.
///
/// # Arguments
///
/// * `probs` - Probability values for each position
/// * `output_path` - Path to save the plot
/// * `config` - Plot configuration
///
/// # Errors
///
/// Returns `VisualizationError` if the plot cannot be created.
#[cfg(any(feature = "visualize-bitmap", feature = "visualize-svg"))]
pub fn plot_probabilities<P: AsRef<Path>>(
    probs: &[f64],
    output_path: P,
    config: &PlotConfig,
) -> Result<()> {
    let path = output_path.as_ref();

    #[cfg(feature = "visualize-bitmap")]
    {
        if path.extension().is_some_and(|e| e == "png") {
            return plot_probabilities_bitmap(probs, path, config);
        }
    }

    #[cfg(feature = "visualize-svg")]
    {
        if path.extension().is_some_and(|e| e == "svg") {
            return plot_probabilities_svg(probs, path, config);
        }
    }

    Err(CirculantError::VisualizationError(
        "unsupported output format".to_string(),
    ))
}

#[cfg(feature = "visualize-bitmap")]
fn plot_probabilities_bitmap<P: AsRef<Path>>(
    probs: &[f64],
    output_path: P,
    config: &PlotConfig,
) -> Result<()> {
    ensure_font_registered();
    let root =
        BitMapBackend::new(output_path.as_ref(), (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    let max_prob = probs.iter().cloned().fold(0.0_f64, f64::max);
    let y_max = if max_prob > 0.0 { max_prob * 1.1 } else { 1.0 };

    let mut chart = ChartBuilder::on(&root)
        .caption(&config.title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..probs.len(), 0.0..y_max)
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc(&config.x_label)
        .y_desc(&config.y_label)
        .draw()
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    chart
        .draw_series(
            probs
                .iter()
                .enumerate()
                .map(|(x, y)| Rectangle::new([(x, 0.0), (x + 1, *y)], BLUE.filled())),
        )
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    root.present()
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    Ok(())
}

#[cfg(feature = "visualize-svg")]
fn plot_probabilities_svg<P: AsRef<Path>>(
    probs: &[f64],
    output_path: P,
    config: &PlotConfig,
) -> Result<()> {
    let root =
        SVGBackend::new(output_path.as_ref(), (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    let max_prob = probs.iter().cloned().fold(0.0_f64, f64::max);
    let y_max = if max_prob > 0.0 { max_prob * 1.1 } else { 1.0 };

    let mut chart = ChartBuilder::on(&root)
        .caption(&config.title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..probs.len(), 0.0..y_max)
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc(&config.x_label)
        .y_desc(&config.y_label)
        .draw()
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    chart
        .draw_series(
            probs
                .iter()
                .enumerate()
                .map(|(x, y)| Rectangle::new([(x, 0.0), (x + 1, *y)], BLUE.filled())),
        )
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    root.present()
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    Ok(())
}

/// Stub for plot_probabilities when no backend is available
#[cfg(not(any(feature = "visualize-bitmap", feature = "visualize-svg")))]
pub fn plot_probabilities<P: AsRef<Path>>(
    _probs: &[f64],
    _output_path: P,
    _config: &PlotConfig,
) -> Result<()> {
    Err(CirculantError::VisualizationError(
        "no visualization backend enabled - enable visualize-bitmap or visualize-svg feature"
            .to_string(),
    ))
}

/// Plot quantum walk evolution over multiple time steps.
///
/// # Arguments
///
/// * `probabilities` - Vector of probability distributions, one per time step
/// * `output_path` - Path to save the plot
/// * `config` - Plot configuration
///
/// # Errors
///
/// Returns `VisualizationError` if the plot cannot be created.
#[cfg(any(feature = "visualize-bitmap", feature = "visualize-svg"))]
pub fn plot_walk_evolution<P: AsRef<Path>>(
    probabilities: &[Vec<f64>],
    output_path: P,
    config: &PlotConfig,
) -> Result<()> {
    if probabilities.is_empty() {
        return Err(CirculantError::VisualizationError(
            "no probability data provided".to_string(),
        ));
    }

    let path = output_path.as_ref();

    #[cfg(feature = "visualize-bitmap")]
    {
        if path.extension().is_some_and(|e| e == "png") {
            return plot_walk_evolution_bitmap(probabilities, path, config);
        }
    }

    #[cfg(feature = "visualize-svg")]
    {
        if path.extension().is_some_and(|e| e == "svg") {
            return plot_walk_evolution_svg(probabilities, path, config);
        }
    }

    Err(CirculantError::VisualizationError(
        "unsupported output format".to_string(),
    ))
}

#[cfg(feature = "visualize-bitmap")]
fn plot_walk_evolution_bitmap<P: AsRef<Path>>(
    probabilities: &[Vec<f64>],
    output_path: P,
    config: &PlotConfig,
) -> Result<()> {
    ensure_font_registered();
    let root =
        BitMapBackend::new(output_path.as_ref(), (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    let num_steps = probabilities.len();
    let num_positions = probabilities[0].len();
    let max_prob = probabilities
        .iter()
        .flat_map(|p| p.iter())
        .cloned()
        .fold(0.0_f64, f64::max);
    let y_max = if max_prob > 0.0 { max_prob * 1.1 } else { 1.0 };

    let mut chart = ChartBuilder::on(&root)
        .caption(&config.title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..num_positions, 0.0..y_max)
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc(&config.x_label)
        .y_desc(&config.y_label)
        .draw()
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    // Use different colors for different time steps
    let colors = [RED, GREEN, BLUE, CYAN, MAGENTA];
    for (i, probs) in probabilities.iter().enumerate() {
        let color = colors[i % colors.len()];
        chart
            .draw_series(LineSeries::new(
                probs.iter().enumerate().map(|(x, y)| (x, *y)),
                color,
            ))
            .map_err(|e| CirculantError::VisualizationError(e.to_string()))?
            .label(format!(
                "t={}",
                i * (num_steps / probabilities.len().max(1))
            ))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    root.present()
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    Ok(())
}

#[cfg(feature = "visualize-svg")]
fn plot_walk_evolution_svg<P: AsRef<Path>>(
    probabilities: &[Vec<f64>],
    output_path: P,
    config: &PlotConfig,
) -> Result<()> {
    let root =
        SVGBackend::new(output_path.as_ref(), (config.width, config.height)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    let num_steps = probabilities.len();
    let num_positions = probabilities[0].len();
    let max_prob = probabilities
        .iter()
        .flat_map(|p| p.iter())
        .cloned()
        .fold(0.0_f64, f64::max);
    let y_max = if max_prob > 0.0 { max_prob * 1.1 } else { 1.0 };

    let mut chart = ChartBuilder::on(&root)
        .caption(&config.title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..num_positions, 0.0..y_max)
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    chart
        .configure_mesh()
        .x_desc(&config.x_label)
        .y_desc(&config.y_label)
        .draw()
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    let colors = [RED, GREEN, BLUE, CYAN, MAGENTA];
    for (i, probs) in probabilities.iter().enumerate() {
        let color = colors[i % colors.len()];
        chart
            .draw_series(LineSeries::new(
                probs.iter().enumerate().map(|(x, y)| (x, *y)),
                color,
            ))
            .map_err(|e| CirculantError::VisualizationError(e.to_string()))?
            .label(format!(
                "t={}",
                i * (num_steps / probabilities.len().max(1))
            ))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .draw()
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    root.present()
        .map_err(|e| CirculantError::VisualizationError(e.to_string()))?;

    Ok(())
}

/// Stub for plot_walk_evolution when no backend is available
#[cfg(not(any(feature = "visualize-bitmap", feature = "visualize-svg")))]
pub fn plot_walk_evolution<P: AsRef<Path>>(
    _probabilities: &[Vec<f64>],
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
    fn test_plot_config_default() {
        let config = PlotConfig::default();
        assert_eq!(config.width, 800);
        assert_eq!(config.height, 600);
    }

    #[test]
    fn test_plot_config_builder() {
        let config = PlotConfig::with_title("Test")
            .dimensions(1024, 768)
            .labels("X", "Y");
        assert_eq!(config.title, "Test");
        assert_eq!(config.width, 1024);
        assert_eq!(config.height, 768);
        assert_eq!(config.x_label, "X");
        assert_eq!(config.y_label, "Y");
    }
}

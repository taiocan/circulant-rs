// @module: crate::visualize
// @status: stable
// @owner: code_expert
// @feature: visualize
// @depends: [crate::visualize::heatmap, crate::visualize::quantum]
// @tests: [none]

//! Visualization utilities for quantum states and matrices.
//!
//! This module provides plotting functionality for probability distributions,
//! quantum state evolution, and heatmaps.
//!
//! # Features
//!
//! Enable one of the visualization features:
//! ```toml
//! [dependencies]
//! # For bitmap output (PNG, etc.)
//! circulant-rs = { version = "0.2", features = ["visualize-bitmap"] }
//! # For SVG output
//! circulant-rs = { version = "0.2", features = ["visualize-svg"] }
//! ```

mod heatmap;
mod quantum;

pub use heatmap::plot_heatmap;
pub use quantum::{plot_probabilities, plot_walk_evolution, PlotConfig};

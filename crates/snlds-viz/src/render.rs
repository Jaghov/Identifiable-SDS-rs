//! Re-export of `snlds_data::render`.
//!
//! Rendering moved to `snlds-data` so the simulator can produce image observations
//! without a viz dependency. Existing `snlds_viz::render::draw_sequence` callers
//! continue to compile unchanged.

pub use snlds_data::render::*;

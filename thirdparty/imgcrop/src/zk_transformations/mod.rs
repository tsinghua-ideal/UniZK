use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub enum Transformation {
    Crop {
        orig_w: u32,
        orig_h: u32,
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    },
}

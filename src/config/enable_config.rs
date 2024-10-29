pub struct EnableConfig {
    pub fft: bool,
    pub tree: bool,
    pub transpose: bool,
    pub poly: bool,
    pub hash: bool,
}

pub static mut ENABLE_CONFIG: EnableConfig = EnableConfig {
    fft: true,
    tree: true,
    transpose: true,
    poly: true,
    hash: true,
};

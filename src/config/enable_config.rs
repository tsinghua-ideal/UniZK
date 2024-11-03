pub struct EnableConfig {
    pub fft: bool,
    pub hash: bool,
    pub other: bool,
}

pub static mut ENABLE_CONFIG: EnableConfig = EnableConfig {
    fft: true,
    hash: true,
    other: true,
};

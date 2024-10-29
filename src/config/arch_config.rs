use crate::util::SIZE_F;

#[derive(Debug, Clone, Copy)]
pub struct ArchConfig {
    pub rdbuf_sz_kb: usize,
    pub wrbuf_sz_kb: usize,
    pub active_buf_frac: f64,
    pub mvl: usize, // max vector length for PE
    pub num_tiles: usize,
    pub array_length: usize,
}

pub static mut ARCH_CONFIG: ArchConfig = ArchConfig {
    rdbuf_sz_kb: 4096,
    wrbuf_sz_kb: 4096,
    active_buf_frac: 0.5,
    mvl: 8,
    num_tiles: 32,
    array_length: 12,
};

impl ArchConfig {
    pub fn active_buf_size(&self) -> usize {
        (self.active_buf_frac * self.rdbuf_sz_kb as f64) as usize
    }

    pub fn num_elems(&self) -> usize {
        self.active_buf_size() * 1024 / SIZE_F
    }

    pub fn num_pes(&self) -> usize {
        self.num_tiles * self.array_length * self.array_length
    }
}

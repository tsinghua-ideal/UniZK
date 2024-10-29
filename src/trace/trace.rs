use std::ops::Index;

use crate::{kernel::memory_copy::MemCpy, system::system::find_consecutive_segments, util::SIZE_F};

#[derive(Debug, Clone, Copy)]
pub enum FetchType {
    Read,
    Write,
}

/// The first dimension of the vector is a read block fit in the buffer,
/// the second dimension is the number of read length.
/// The third dimension is a vector with length = number of banks of on-chip buffer.
/// The elements in the third dimension vector will send into different banks.
#[derive(Debug, Clone)]
pub struct Fetch {
    pub fetch_type: FetchType,
    pub addr: Vec<Vec<(u64, u64)>>,
    pub systolic: bool,
    pub word_sz: usize,
    pub mergable: bool,
    pub delay: Vec<usize>,
    pub interval: f32, // cycles interval between two request lines
}

impl Fetch {
    pub fn new(t: FetchType) -> Fetch {
        Fetch {
            fetch_type: t,
            addr: Vec::new(),
            systolic: false,
            word_sz: SIZE_F,
            mergable: false,
            delay: Vec::new(),
            interval: 1.0,
        }
    }

    pub fn push(&mut self, fetch: Vec<Vec<usize>>) {
        self.addr.push(find_consecutive_segments(
            &fetch.into_iter().flatten().collect(),
        ));
    }

    pub fn extend(&mut self, fetch: Fetch) {
        self.addr.extend(fetch.addr);
        self.delay.extend(fetch.delay);
    }

    pub fn len(&self) -> usize {
        self.addr.len()
    }

    pub fn num_fetch_lines(&self) -> usize {
        self.addr.iter().map(|x| x.len()).sum()
    }

    pub fn addr_trans(&mut self, memcpy: &MemCpy) {
        self.addr.iter_mut().for_each(|x| {
            let mut x_new = x.clone();
            x.clear();
            while !x_new.is_empty() {
                let y = x_new.pop().unwrap();
                let y = memcpy.addr_trans(&y);
                x.push(y[0]);
                y.iter().skip(1).for_each(|z| x_new.push(*z));
            }
        })
    }

    pub fn addr_trans_vec(&mut self, memcpys: &Vec<MemCpy>) {
        self.addr.iter_mut().for_each(|x| {
            let mut x_new = x.clone();
            x.clear();
            while !x_new.is_empty() {
                let y = x_new.pop().unwrap();
                let mut y_new = vec![];
                for mc in memcpys.iter() {
                    y_new = mc.addr_trans(&y);
                    if y_new[0].0 != y.0 {
                        break;
                    }
                }
                x.push(y_new[0]);
                y_new.iter().skip(1).for_each(|z| x_new.push(*z));
            }
        })
    }

    pub fn clear(&mut self) {
        self.addr.clear();
        self.delay.clear();
        self.interval = 0.0;
    }
}

impl Index<usize> for Fetch {
    type Output = Vec<(u64, u64)>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.addr[index]
    }
}

/// The elements in the second dimension vector will send into different PEs.
#[derive(Debug, Clone)]
pub struct Request {
    pub _addr: Vec<Vec<(u64, u64)>>,
    pub word_sz: usize,
    pub num_lines: Vec<usize>,
}

impl Request {
    pub fn new() -> Request {
        Request {
            _addr: Vec::new(),
            word_sz: SIZE_F,
            num_lines: vec![],
        }
    }

    pub fn len(&self) -> usize {
        self.num_lines.len()
    }

    pub fn push(&mut self, request: Vec<Vec<usize>>) {
        self.num_lines.push(request.len());
    }

    pub fn extend(&mut self, request: Request) {
        self.num_lines.extend(request.num_lines);
    }

    pub fn num_request_lines(&self) -> usize {
        return self.num_lines.iter().sum();
    }

    pub fn clear(&mut self) {
        self.num_lines.clear();
    }
}

use std::cmp::min;
use std::vec;

use log::info;

use crate::trace::trace::{Fetch, FetchType, Request};
use crate::config::arch_config::ARCH_CONFIG;
use crate::config::enable_config::ENABLE_CONFIG;
use crate::kernel::kernel::Kernel;
use crate::util::{ceil_div_usize, log2, HASH_COMPUTATION, SINGLE_HASH_DELAY, SIZE_F};

#[derive(Debug, Clone)]
pub struct TreeConfig {
    pub leaf_length: usize, // length of the leaf
    pub cap_height: usize,
    pub num_leaves: usize,
    pub addr_leaves: usize,
    pub addr_transposed_leaves: usize,
    pub addr_digest_buf: usize,
    pub addr_cap_buf: usize,
    pub transposed_leaves: bool,
}

pub fn get_buf_sz_elems() -> usize {
    unsafe { ARCH_CONFIG.active_buf_size() * 1024 / SIZE_F }
}

pub struct Tree {
    pub config: TreeConfig,
    pub prefetch: Fetch,
    pub read_request: Request,
    pub write_request: Request,
    pub drain: Fetch,
}

impl Kernel for Tree {
    fn create_prefetch(&mut self) {
        let log2_num_leaves = log2(self.config.num_leaves);
        let num_elems = get_buf_sz_elems();
        let residual_level = log2_num_leaves + 1 - self.config.cap_height;
        let mut level = 0;

        loop {
            let leaf_length = if level == 0 {
                self.config.leaf_length
            } else {
                12
            };
            let log2_num_leaves_current_level = log2_num_leaves - level;
            let num_leaves_current_level = 1 << log2_num_leaves_current_level;
            let log2_num_leaves_reading = log2(num_elems / leaf_length);
            let log2_num_leaves_reading =
                log2_num_leaves_reading.min(log2_num_leaves_current_level - self.config.cap_height);
            let num_leaves_reading = 1 << log2_num_leaves_reading;
            let num_levels = min(residual_level - level, log2_num_leaves_reading + 1);
            // println!(
            //     "residual_level - level: {}, log2_num_leaves_reading + 1: {}",
            //     residual_level - level,
            //     log2_num_leaves_reading + 1
            // );
            // println!(
            //     "prefetch\nlevel: {}, num_leaves_current_level: {}, num_leaves_reading: {}, num_levels: {}",
            //     level, num_leaves_current_level, num_leaves_reading, num_levels
            // );

            for leaf_idx in (0..num_leaves_current_level).step_by(num_leaves_reading) {
                let mut fetch = Vec::new();
                for i in leaf_idx..(leaf_idx + num_leaves_reading) {
                    if level == 0 {
                        for j in (0..leaf_length).step_by(Tree::SPONGE_RATE) {
                            let idx = (j..min(j + Tree::SPONGE_RATE, leaf_length))
                                .map(|x| i * self.config.leaf_length + x)
                                .collect();
                            fetch.push(self.idx_to_addr_leaves(&idx));
                        }
                    } else {
                        let mut idx_0: Vec<usize> = (0..Tree::DIGEST_LENGTH)
                            .map(|x| {
                                self.tree_idx_transform(level - 1, i << 1) * Tree::DIGEST_LENGTH + x
                            })
                            .collect::<Vec<_>>();
                        let idx_1 = (0..Tree::DIGEST_LENGTH)
                            .map(|x| {
                                self.tree_idx_transform(level - 1, (i << 1) + 1)
                                    * Tree::DIGEST_LENGTH
                                    + x
                            })
                            .collect::<Vec<_>>();
                        idx_0.extend(idx_1);
                        fetch.push(self.idx_to_addr_digest(&idx_0));
                    }
                }
                self.prefetch.push(fetch);
            }
            level += num_levels;
            if level >= residual_level {
                break;
            }
        }
        self.prefetch.delay = vec![0; self.prefetch.len()];
        self.prefetch.interval = 16.0 + 1.0 + 6.0;
        self.prefetch.systolic = true;
    }

    fn create_read_request(&mut self) {
        let log2_num_leaves = log2(self.config.num_leaves);
        let num_elems = get_buf_sz_elems();
        let residual_level = log2_num_leaves + 1 - self.config.cap_height;

        let mut level = 0;
        loop {
            let leaf_length = if level == 0 {
                self.config.leaf_length
            } else {
                12
            };
            let log2_num_leaves_current_level = log2_num_leaves - level;
            let num_leaves_current_level = 1 << log2_num_leaves_current_level;
            let log2_num_leaves_reading = log2(num_elems / leaf_length);
            let log2_num_leaves_reading =
                log2_num_leaves_reading.min(log2_num_leaves_current_level - self.config.cap_height);
            let num_leaves_reading = 1 << log2_num_leaves_reading;
            let num_levels = min(residual_level - level, log2_num_leaves_reading + 1);

            for leaf_idx in (0..num_leaves_current_level).step_by(num_leaves_reading) {
                let mut request = Vec::new();
                for level_idx in 0..num_levels {
                    let leaf_idx_left = leaf_idx >> level_idx;
                    let leaf_idx_right = (leaf_idx + num_leaves_reading) >> level_idx;
                    for i in leaf_idx_left..leaf_idx_right {
                        if level == 0 && level_idx == 0 {
                            for j in (0..leaf_length).step_by(Tree::SPONGE_RATE) {
                                let req = (j..min(j + Tree::SPONGE_RATE, leaf_length))
                                    .map(|x| i * self.config.leaf_length + x)
                                    .collect();
                                request.push(self.idx_to_addr_leaves(&req));
                            }
                        } else {
                            let mut req = (0..Tree::DIGEST_LENGTH)
                                .map(|x| {
                                    self.tree_idx_transform(level + level_idx - 1, i << 1)
                                        * Tree::DIGEST_LENGTH
                                        + x
                                })
                                .collect::<Vec<_>>();
                            let idx_1 = (0..Tree::DIGEST_LENGTH)
                                .map(|x| {
                                    self.tree_idx_transform(level + level_idx - 1, (i << 1) + 1)
                                        * Tree::DIGEST_LENGTH
                                        + x
                                })
                                .collect::<Vec<_>>();
                            req.extend(idx_1);
                            request.push(self.idx_to_addr_digest(&req));
                        }
                    }
                }
                self.read_request.push(request);
            }

            level += num_levels;
            if level >= residual_level {
                break;
            }
        }

        // check correctness
        let mut length = 0;

        let mut num_leaves = self.config.num_leaves;
        while num_leaves >= (1 << self.config.cap_height) {
            if num_leaves == self.config.num_leaves {
                length += num_leaves * ceil_div_usize(self.config.leaf_length, Tree::SPONGE_RATE);
            } else {
                length += num_leaves;
            }
            num_leaves >>= 1;
        }
        assert_eq!(self.read_request.num_request_lines(), length);
    }

    fn create_write_request(&mut self) {
        let log2_num_leaves = log2(self.config.num_leaves);
        let num_elems = get_buf_sz_elems();
        let residual_level = log2_num_leaves + 1 - self.config.cap_height;

        let mut level = 0;
        loop {
            let leaf_length = if level == 0 {
                self.config.leaf_length
            } else {
                12
            };
            let log2_num_leaves_current_level = log2_num_leaves - level;
            let num_leaves_current_level = 1 << log2_num_leaves_current_level;
            let log2_num_leaves_reading = log2(num_elems / leaf_length);
            let log2_num_leaves_reading =
                log2_num_leaves_reading.min(log2_num_leaves_current_level - self.config.cap_height);
            let num_leaves_reading = 1 << log2_num_leaves_reading;
            let num_levels = min(residual_level - level, log2_num_leaves_reading + 1);

            for leaf_idx in (0..num_leaves_current_level).step_by(num_leaves_reading) {
                let mut request = Vec::new();
                for level_idx in 0..num_levels {
                    let leaf_idx_left = leaf_idx >> level_idx;
                    let leaf_idx_right = (leaf_idx + num_leaves_reading) >> level_idx;
                    for i in leaf_idx_left..leaf_idx_right {
                        if level + level_idx == residual_level - 1 {
                            let req = (0..Tree::DIGEST_LENGTH)
                                .map(|x| i * Tree::DIGEST_LENGTH + x)
                                .collect::<Vec<_>>();
                            request.push(self.idx_to_addr_cap(&req));
                        } else {
                            let req = (0..Tree::DIGEST_LENGTH)
                                .map(|x| {
                                    self.tree_idx_transform(level + level_idx, i)
                                        * Tree::DIGEST_LENGTH
                                        + x
                                })
                                .collect::<Vec<_>>();
                            request.push(self.idx_to_addr_digest(&req));
                        }
                    }
                }
                self.write_request.push(request);
            }

            level += num_levels;
            if level >= residual_level {
                break;
            }
        }

        // check correctness
        let mut length = 0;

        let mut num_leaves = self.config.num_leaves;
        while num_leaves >= (1 << self.config.cap_height) {
            length += num_leaves;
            num_leaves >>= 1;
        }
        assert_eq!(self.write_request.num_request_lines(), length);
    }

    fn create_drain(&mut self) {
        let num_digests = 2 * (self.config.num_leaves - (1 << self.config.cap_height));
        let log2_num_leaves = log2(self.config.num_leaves);
        let num_elems = get_buf_sz_elems();
        let residual_level = log2_num_leaves + 1 - self.config.cap_height;

        let mut level = 0;
        // let mut idx_set = HashSet::new();
        loop {
            let leaf_length = if level == 0 {
                self.config.leaf_length
            } else {
                12
            };
            let log2_num_leaves_current_level = log2_num_leaves - level;
            let num_leaves_current_level = 1 << log2_num_leaves_current_level;
            let log2_num_leaves_reading = log2(num_elems / leaf_length);
            let log2_num_leaves_reading =
                log2_num_leaves_reading.min(log2_num_leaves_current_level - self.config.cap_height);
            let num_leaves_reading = 1 << log2_num_leaves_reading;
            let num_levels = min(residual_level - level, log2_num_leaves_reading + 1);
            // println!(
            //     "drain\nlevel: {}, num_leaves_current_level: {}, num_leaves_reading: {}, num_levels: {}",
            //     level, num_leaves_current_level, num_leaves_reading, num_levels
            // );

            for leaf_idx in (0..num_leaves_current_level).step_by(num_leaves_reading) {
                let mut fetch = Vec::new();

                if level == 0 && self.config.transposed_leaves {
                    assert_ne!(self.config.addr_transposed_leaves, 0);
                    let leaf_idx_left = leaf_idx;
                    let leaf_idx_right = leaf_idx + num_leaves_reading;
                    for leaf_idx in leaf_idx_left..leaf_idx_right {
                        let idx = (0..self.config.leaf_length)
                            .map(|x| leaf_idx * self.config.leaf_length + x)
                            .collect::<Vec<_>>();
                        fetch.push(self.idx_to_addr_transposed_leaves(&idx));
                    }
                }

                if level < residual_level - 1 {
                    let leaf_idx_left = leaf_idx;
                    let leaf_idx_right = leaf_idx + num_leaves_reading;
                    let leaf_idx_left = self.tree_idx_transform(level, leaf_idx_left);
                    let leaf_idx_right = self.tree_idx_transform(level, leaf_idx_right - 1) + 1;

                    for leaf_idx in leaf_idx_left..leaf_idx_right {
                        // idx_set.insert(leaf_idx);
                        assert!(leaf_idx < num_digests);

                        let idx = (0..Tree::DIGEST_LENGTH)
                            .map(|x| leaf_idx * Tree::DIGEST_LENGTH + x)
                            .collect::<Vec<_>>();
                        fetch.push(self.idx_to_addr_digest(&idx));
                    }
                }
                for level_idx in 0..num_levels {
                    debug_assert!(level_idx <= residual_level);
                    if level_idx < residual_level - 1 {
                        let leaf_idx_left = leaf_idx >> level_idx;
                        let leaf_idx_right = (leaf_idx + num_leaves_reading) >> level_idx;
                        for i in leaf_idx_left..leaf_idx_right {
                            let idx = (0..Tree::DIGEST_LENGTH)
                                .map(|x| {
                                    self.tree_idx_transform(level + level_idx, i)
                                        * Tree::DIGEST_LENGTH
                                        + x
                                })
                                .collect::<Vec<_>>();
                            fetch.push(self.idx_to_addr_digest(&idx));
                        }
                    }
                }

                if level + num_levels >= residual_level {
                    let cap_index = leaf_idx / num_leaves_reading;
                    // idx_set.insert(cap_index);
                    let idx = (0..Tree::DIGEST_LENGTH)
                        .map(|x| cap_index * Tree::DIGEST_LENGTH + x)
                        .collect::<Vec<_>>();
                    fetch.push(self.idx_to_addr_cap(&idx));
                }

                self.drain.push(fetch);
            }

            level += num_levels;
            if level >= residual_level {
                break;
            }
        }
        self.drain.delay = vec![SINGLE_HASH_DELAY; self.drain.len()];
        self.drain.interval = 0.;
        self.drain.systolic = true;
    }

    fn get_prefetch(&self) -> Fetch {
        self.prefetch.clone()
    }
    fn get_drain(&self) -> Fetch {
        self.drain.clone()
    }
    fn get_read_request(&self) -> Request {
        self.read_request.clone()
    }
    fn get_write_request(&self) -> Request {
        self.write_request.clone()
    }
    fn log(&self) {
        info!("Tree Kernel config: {:?}", self.config);
    }
    fn get_kernel_type(&self) -> String {
        String::from("Tree")
    }
    fn get_computation(&self) -> usize {
        let mut res = ceil_div_usize(self.config.leaf_length, 8) * self.config.num_leaves;
        let mut current_num_leaves = self.config.num_leaves;
        while current_num_leaves > (1 << self.config.cap_height) {
            current_num_leaves /= 2;
            res += current_num_leaves;
        }
        res * HASH_COMPUTATION
    }
}

impl Tree {
    const SPONGE_RATE: usize = 8;
    pub const DIGEST_LENGTH: usize = 4;

    pub fn new(config: TreeConfig) -> Tree {
        let mut k = Tree {
            config,
            prefetch: Fetch::new(FetchType::Read),
            read_request: Request::new(),
            write_request: Request::new(),
            drain: Fetch::new(FetchType::Write),
        };

        k.log();
        if unsafe { ENABLE_CONFIG.tree } {
            k.init();
        }
        k
    }

    fn tree_idx_transform(&self, level: usize, idx: usize) -> usize {
        let height = log2(self.config.num_leaves) + 1;
        let subtree_height = height - self.config.cap_height;

        let number_of_nodes = 1 << (height - level - 1);
        let number_of_nodes_in_subtree = number_of_nodes >> self.config.cap_height;
        let subtree_idx = idx / number_of_nodes_in_subtree;

        let num_digests = Self::num_digests(self.config.num_leaves, self.config.cap_height);
        let digest_buf_offset = (num_digests >> self.config.cap_height) * subtree_idx;

        let mut subtree_offset = 0usize;
        for i in 0..level {
            subtree_offset |= 1 << (subtree_height - i - 1);
        }

        digest_buf_offset + subtree_offset + idx % (1 << (subtree_height - level - 1))
    }

    /// transform the index of the tree from (level, idx) to the idx of the new layout
    fn _tree_idx_transform_plonk(&self, level: usize, idx: usize) -> usize {
        let height = log2(self.config.num_leaves) + 1;
        let subtree_height = height - self.config.cap_height;
        assert!(level < height - self.config.cap_height - 1);

        let number_of_nodes = 1 << (height - level - 1);
        let number_of_nodes_in_subtree = number_of_nodes >> self.config.cap_height;
        let subtree_idx = idx / number_of_nodes_in_subtree;

        let num_digests = Self::num_digests(self.config.num_leaves, self.config.cap_height);
        let digest_buf_offset = (num_digests >> self.config.cap_height) * subtree_idx;

        let number_of_left_nodes = if idx & 1 == 1 {
            (1 << (level + 1)) - 1
        } else {
            (1 << (level + 1)) - 2
        };
        let mask: usize = (1 << (subtree_height - level - 1)) - 2;

        number_of_left_nodes + digest_buf_offset + (((idx & mask) << level) << 1)
    }

    fn idx_to_addr_leaves(&self, index: &Vec<usize>) -> Vec<usize> {
        index
            .iter()
            .map(|x| {
                let y = if self.config.transposed_leaves {
                    x / self.config.leaf_length
                        + x % self.config.leaf_length * self.config.num_leaves
                } else {
                    *x
                };
                self.config.addr_leaves + y * SIZE_F
            })
            .collect()
    }

    fn idx_to_addr_transposed_leaves(&self, index: &Vec<usize>) -> Vec<usize> {
        index
            .iter()
            .map(|x| self.config.addr_transposed_leaves + x * SIZE_F)
            .collect()
    }

    fn idx_to_addr_digest(&self, index: &Vec<usize>) -> Vec<usize> {
        index
            .iter()
            .map(|x| self.config.addr_digest_buf + x * SIZE_F)
            .collect()
    }

    fn idx_to_addr_cap(&self, index: &Vec<usize>) -> Vec<usize> {
        index
            .iter()
            .map(|x| self.config.addr_cap_buf + x * SIZE_F)
            .collect()
    }

    pub fn num_digests(num_leaves: usize, cap_height: usize) -> usize {
        2 * (num_leaves - (1 << cap_height))
    }

    pub fn num_caps(cap_height: usize) -> usize {
        1 << cap_height
    }

    pub fn get_padding_length(num_leaves: usize, leaf_length: usize, cap_height: usize) -> usize {
        let log2_num_leaves = log2(num_leaves);
        let num_elems = get_buf_sz_elems();

        let leaf_length = leaf_length;
        let log2_num_leaves_reading = log2(num_elems / leaf_length);
        let log2_num_leaves_reading = log2_num_leaves_reading.min(log2_num_leaves - cap_height);
        1 << log2_num_leaves_reading
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_idx_transform() {
        let config = TreeConfig {
            leaf_length: 8,
            cap_height: 2,
            num_leaves: 16,
            addr_leaves: 0,
            addr_transposed_leaves: 0,
            addr_digest_buf: 0,
            addr_cap_buf: 0,
            transposed_leaves: false,
        };
        let tree = Tree {
            config,
            prefetch: Fetch::new(FetchType::Read),
            read_request: Request::new(),
            write_request: Request::new(),
            drain: Fetch::new(FetchType::Write),
        };

        assert_eq!(tree.tree_idx_transform(0, 0), 0);
        assert_eq!(tree.tree_idx_transform(0, 1), 1);
        assert_eq!(tree.tree_idx_transform(0, 2), 2);
        assert_eq!(tree.tree_idx_transform(0, 3), 3);
        assert_eq!(tree.tree_idx_transform(0, 4), 6);
        assert_eq!(tree.tree_idx_transform(0, 5), 7);
        assert_eq!(tree.tree_idx_transform(0, 6), 8);
        assert_eq!(tree.tree_idx_transform(0, 7), 9);
        assert_eq!(tree.tree_idx_transform(0, 8), 12);
        assert_eq!(tree.tree_idx_transform(0, 9), 13);
        assert_eq!(tree.tree_idx_transform(0, 10), 14);
        assert_eq!(tree.tree_idx_transform(0, 11), 15);
        assert_eq!(tree.tree_idx_transform(0, 12), 18);
        assert_eq!(tree.tree_idx_transform(0, 13), 19);
        assert_eq!(tree.tree_idx_transform(0, 14), 20);
        assert_eq!(tree.tree_idx_transform(0, 15), 21);

        assert_eq!(tree.tree_idx_transform(1, 0), 4);
        assert_eq!(tree.tree_idx_transform(1, 1), 5);
        assert_eq!(tree.tree_idx_transform(1, 2), 10);
        assert_eq!(tree.tree_idx_transform(1, 3), 11);
        assert_eq!(tree.tree_idx_transform(1, 4), 16);
        assert_eq!(tree.tree_idx_transform(1, 5), 17);
        assert_eq!(tree.tree_idx_transform(1, 6), 22);
        assert_eq!(tree.tree_idx_transform(1, 7), 23);
    }
}

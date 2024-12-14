use std::collections::HashMap;

use plonky2::util::ceil_div_usize;

use crate::trace::trace::{Fetch, FetchType, Request};
use crate::config::arch_config::ARCH_CONFIG;
use crate::config::ram_config::OpRecord;
use crate::config::RamConfig;
use crate::kernel::filter_drain::merge;
use crate::kernel::kernel::Kernel;
use crate::memory::memory_allocator::MemAlloc;
pub struct System {
    pub mem: MemAlloc,
    pub ramsim: RamConfig,
    pub last_prefetch_block: Vec<(u64, u64)>,
    pub last_drain_block: Vec<(u64, u64)>,

    pub computation: HashMap<String, usize>,
}

impl System {
    pub fn new(mem: MemAlloc, ramsim: RamConfig) -> System {
        let last_prefetch_block = vec![];
        let last_drain_block = vec![];
        System {
            mem,
            ramsim,

            last_prefetch_block,
            last_drain_block,
            computation: HashMap::new(),
        }
    }

    pub fn reset(&mut self) {
        self.computation.clear();
        self.last_prefetch_block.clear();
        self.last_drain_block.clear();
        self.ramsim.reset();
    }

    fn run_trace(
        &mut self,
        mut prefetch: Fetch,
        read_request: Request,
        write_request: Request,
        mut drain: Fetch,
    ) {
        assert!(
            prefetch.addr.len() == prefetch.delay.len()
                && drain.addr.len() == drain.delay.len()
                && prefetch.len() == read_request.len()
                && drain.len() == write_request.len()
                && (prefetch.len() == drain.len() || prefetch.len() == 0 || drain.len() == 0),
            "Error, prefetch.addr.len() = {}, prefetch.delay.len() = {}, read_request.len() = {}, write_request.len() = {}, drain.addr.len() = {}, drain.delay.len() = {}",
            prefetch.addr.len(),
            prefetch.delay.len(),
            read_request.len(),
            write_request.len(),
            drain.addr.len(),
            drain.delay.len()
        );

        prefetch.addr.insert(0, self.last_prefetch_block.clone());
        drain.addr.insert(0, self.last_drain_block.clone());
        merge(&mut prefetch, &mut drain);
        prefetch.addr.remove(0);
        drain.addr.remove(0);
        self.last_prefetch_block = prefetch.addr.last().unwrap_or(&vec![]).clone();
        self.last_drain_block = drain.addr.last().unwrap_or(&vec![]).clone();

        assert_eq!(prefetch.systolic, drain.systolic);
        let parallel_level = unsafe {
            if prefetch.systolic {
                ARCH_CONFIG.num_tiles
            } else {
                ARCH_CONFIG.num_tiles * ARCH_CONFIG.array_length
            }
        };

        let fetch_len = (prefetch.addr.len()).max(drain.addr.len());

        for fetch_idx in 0..fetch_len {
            let read_sg = prefetch.addr.get(fetch_idx).unwrap_or(&vec![]).clone();
            let write_sg = drain.addr.get(fetch_idx).unwrap_or(&vec![]).clone();

            let read_op = read_sg
                .iter()
                .enumerate()
                .map(|(id, &addr)| OpRecord {
                    id: id as u64,
                    fetch_type: prefetch.fetch_type,
                    delay: 0,
                    addr: addr.0,
                    dependencies: Vec::new(),
                    size: (addr.1 - addr.0 + 1) as u32,
                })
                .collect::<Vec<_>>();
            let mut write_op = write_sg
                .iter()
                .enumerate()
                .map(|(id, &addr)| OpRecord {
                    id: (id + read_op.len()) as u64,
                    fetch_type: drain.fetch_type,
                    delay: 0,
                    addr: addr.0,
                    dependencies: Vec::new(),
                    size: (addr.1 - addr.0 + 1) as u32,
                })
                .collect::<Vec<_>>();
            for op in write_op.iter_mut() {
                op.dependencies.extend(read_op.iter().map(|x| x.id));
            }
            if let Some(op) = write_op.first_mut() {
                let delay_usize = prefetch.delay.get(fetch_idx).unwrap_or(&0)
                    + drain.delay.get(fetch_idx).unwrap_or(&0)
                    + prefetch.interval as usize
                        * ceil_div_usize(
                            *read_request.num_lines.get(fetch_idx).unwrap_or(&0),
                            parallel_level,
                        )
                    + drain.interval as usize
                        * ceil_div_usize(
                            *write_request.num_lines.get(fetch_idx).unwrap_or(&0),
                            parallel_level,
                        );
                op.delay = delay_usize as u32;
            }
            let first_write_id = write_op.first().unwrap_or(&OpRecord::default()).id;
            write_op
                .iter_mut()
                .skip(1)
                .for_each(|x| x.dependencies.push(first_write_id));

            if self.ramsim.add_trace(read_op, write_op).is_err() {
                panic!("Error adding trace");
            }
        }
    }

    pub fn run_once<K: Kernel>(&mut self, kernel: &K) {
        //add computation to self.computation
        let comp = self
            .computation
            .entry(kernel.get_kernel_type())
            .or_insert(0);
        *comp += kernel.get_computation();

        let prefetch = kernel.get_prefetch();
        let read_request = kernel.get_read_request();
        let write_request = kernel.get_write_request();
        let drain = kernel.get_drain();
        self.run_trace(prefetch, read_request, write_request, drain)
    }

    pub fn run_vec<T: Kernel>(&mut self, kernels: Vec<T>) {
        let prefetches = kernels.iter().map(|x| x.get_prefetch()).collect::<Vec<_>>();
        let read_requests = kernels
            .iter()
            .map(|x| x.get_read_request())
            .collect::<Vec<_>>();
        let write_requests = kernels
            .iter()
            .map(|x| x.get_write_request())
            .collect::<Vec<_>>();
        let drains = kernels.iter().map(|x| x.get_drain()).collect::<Vec<_>>();

        let mut prefetch = Fetch::new(FetchType::Read);
        let mut read_request = Request::new();
        let mut write_request = Request::new();
        let mut drain = Fetch::new(FetchType::Write);
        prefetch.addr.push(vec![]);
        prefetch.delay = vec![0; prefetch.len()];
        drain.addr.push(vec![]);
        drain.delay = vec![0; drain.len()];
        read_request.push(vec![]);
        write_request.push(vec![]);

        for i in 0..prefetches.len() {
            for j in 0..prefetches[i].addr.len() {
                prefetch.addr[0].extend(prefetches[i].addr[j].clone());
                read_request.num_lines[0] += read_requests[i].num_lines[j];
            }
            for j in 0..drains[i].addr.len() {
                drain.addr[0].extend(drains[i].addr[j].clone());
                write_request.num_lines[0] += write_requests[i].num_lines[j];
            }
        }
        self.run_trace(prefetch, read_request, write_request, drain);
    }

    pub fn get_computation(&mut self) -> HashMap<String, usize> {
        self.computation.clone()
    }
}

pub fn find_consecutive_segments(nums: &Vec<usize>) -> Vec<(u64, u64)> {
    let mut sorted_numbers = nums.clone();
    sorted_numbers.sort();

    let mut ranges = Vec::new();
    if !sorted_numbers.is_empty() && sorted_numbers[0] == 0 {
        sorted_numbers.remove(0);
    }
    if sorted_numbers.is_empty() {
        return ranges;
    }
    let mut start = sorted_numbers[0];
    let mut end = sorted_numbers[0];

    for &num in sorted_numbers.iter().skip(1) {
        if num - end <= OpRecord::SIZE as usize {
            end = num;
        } else {
            ranges.push((start as u64, end as u64));
            start = num;
            end = num;
        }
    }

    ranges.push((start as u64, end as u64));

    ranges
}

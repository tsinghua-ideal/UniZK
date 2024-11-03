use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};

use crate::trace::trace::{Fetch, FetchType, Request};
use crate::config::arch_config::ARCH_CONFIG;
use crate::config::enable_config::ENABLE_CONFIG;
use crate::kernel::vector_operation::VecOpConfig;
use crate::memory::memory_allocator::MemAlloc;
use crate::util::SIZE_F;

use super::kernel::Kernel;
use super::vector_operation::VecOpSrc;

const NUM_VEC_REGS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputType {
    Not,
    NotFinal,
    Final,
}

//todo check array_length and num requests lines for correct throughput and delay
// todo reorder fetch
#[derive(Debug, Clone)]
pub struct Convoy {
    vec_ops: Vec<VecOpConfig>,
    num_ld_ops: usize,
    reg_file_state: Vec<(usize, usize, OutputType)>, // start, size, is_output
}

impl Convoy {
    pub fn new() -> Convoy {
        Convoy {
            vec_ops: Vec::new(),
            num_ld_ops: 0,
            reg_file_state: vec![(0, 0, OutputType::Not); NUM_VEC_REGS],
        }
    }

    pub fn num_vec_mul(&self) -> usize {
        self.vec_ops.iter().filter(|op| op.is_mul()).count()
    }

    pub fn num_vec_add_or_sub(&self) -> usize {
        self.vec_ops.iter().filter(|op| op.is_add_or_sub()).count()
    }

    pub fn need_ld(&self, vec_op: VecOpConfig) -> usize {
        let mut res_0 = true;
        let mut res_1 = match vec_op.op_src {
            VecOpSrc::VV => true,
            _ => false,
        };
        for i in 0..NUM_VEC_REGS {
            if cover(
                (self.reg_file_state[i].0, self.reg_file_state[i].1),
                (vec_op.addr_input_0, vec_op.vector_length),
            ) {
                res_0 = false;
            }
            if res_1
                && cover(
                    (self.reg_file_state[i].0, self.reg_file_state[i].1),
                    (vec_op.addr_input_1, vec_op.vector_length),
                )
            {
                res_1 = false;
            }
        }
        if vec_op.addr_input_0 == 0 {
            res_0 = false;
        }
        if vec_op.addr_input_1 == 0 {
            res_1 = false;
        }

        return res_0 as usize + res_1 as usize;
    }

    pub fn used(&self, reg: (usize, usize)) -> bool {
        for op in self.vec_ops.iter() {
            if cover(reg, (op.addr_output, op.vector_length))
                || cover(reg, (op.addr_input_0, op.vector_length))
                || cover(reg, (op.addr_input_1, op.vector_length))
            {
                return true;
            }
        }
        false
    }

    pub fn overlap(&self, reg: (usize, usize)) -> bool {
        for op in self.vec_ops.iter() {
            if overlap(reg, (op.addr_output, op.vector_length))
                || overlap(reg, (op.addr_input_0, op.vector_length))
            {
                return true;
            }
        }
        false
    }

    pub fn add_ref_file_state(&mut self, addr: usize, length: usize, is_output: OutputType) {
        if addr == 0 {
            return;
        }
        for i in 0..NUM_VEC_REGS {
            if cover(
                (self.reg_file_state[i].0, self.reg_file_state[i].1),
                (addr, length),
            ) {
                return;
            }
        }

        for i in 0..NUM_VEC_REGS {
            if !self.used((self.reg_file_state[i].0, self.reg_file_state[i].1)) {
                self.reg_file_state.remove(i);
                self.reg_file_state.push((addr, length, is_output));
                return;
            }
        }
    }

    pub fn add_vec_op(&mut self, vec_op: VecOpConfig) -> bool {
        if self.vec_ops.len() >= 3
            || self.num_vec_mul() >= 1 && vec_op.is_mul()
            || self.num_vec_add_or_sub() >= 2 && vec_op.is_add_or_sub()
        {
            return false;
        }

        if self.num_ld_ops + self.need_ld(vec_op) > 2 {
            return false;
        }

        self.vec_ops.push(vec_op);
        self.num_ld_ops += self.need_ld(vec_op);

        self.add_ref_file_state(vec_op.addr_input_0, vec_op.vector_length, OutputType::Not);
        if let VecOpSrc::VV = vec_op.op_src {
            self.add_ref_file_state(vec_op.addr_input_1, vec_op.vector_length, OutputType::Not);
        }
        let ot = if vec_op.is_final_output {
            OutputType::Final
        } else {
            OutputType::NotFinal
        };
        self.add_ref_file_state(vec_op.addr_output, vec_op.vector_length, ot);

        true
    }

    pub fn update_reg_file_state(&mut self, state: &Vec<(usize, usize, OutputType)>) -> bool {
        let old_state = self.reg_file_state.clone();

        self.reg_file_state = state.clone();

        let mut new_ld_ops = 0;
        for op in self.vec_ops.clone().iter() {
            new_ld_ops += self.need_ld(*op);

            if new_ld_ops > 2 {
                self.reg_file_state = old_state;
                return false;
            }

            self.add_ref_file_state(op.addr_input_0, op.vector_length, OutputType::Not);
            if let VecOpSrc::VV = op.op_src {
                self.add_ref_file_state(op.addr_input_1, op.vector_length, OutputType::Not);
            }
            let ot = if op.is_final_output {
                OutputType::Final
            } else {
                OutputType::NotFinal
            };
            self.add_ref_file_state(op.addr_output, op.vector_length, ot);
        }

        true
    }

    pub fn split_with_new_state(&mut self, state: &Vec<(usize, usize, OutputType)>) -> Vec<Convoy> {
        let mut res: Vec<Convoy> = Vec::new();
        let mut new_convoy = Convoy::new();
        new_convoy.reg_file_state = state.clone();

        let mut new_ld_ops = 0;
        for op in self.vec_ops.clone().iter() {
            new_ld_ops += new_convoy.need_ld(*op);

            if new_ld_ops > 2 {
                res.push(new_convoy);
                new_convoy = Convoy::new();
                new_convoy.reg_file_state = res.last().unwrap().reg_file_state.clone();
                new_ld_ops = 0;
            }

            new_convoy.add_vec_op(*op);
        }

        res.push(new_convoy);
        res
    }

    /// RAW, WAR, WAW
    pub fn hazard(&self, vec_op: VecOpConfig) -> bool {
        for op in self.vec_ops.iter() {
            // RAW
            if overlap(
                (vec_op.addr_input_0, vec_op.vector_length),
                (op.addr_output, op.vector_length),
            ) {
                return true;
            }
            if let VecOpSrc::VV = vec_op.op_src {
                if overlap(
                    (vec_op.addr_input_1, vec_op.vector_length),
                    (op.addr_output, op.vector_length),
                ) {
                    return true;
                }
            }
            // WAR
            if overlap(
                (vec_op.addr_output, vec_op.vector_length),
                (op.addr_input_0, op.vector_length),
            ) {
                return true;
            }
            if let VecOpSrc::VV = op.op_src {
                if overlap(
                    (vec_op.addr_output, vec_op.vector_length),
                    (op.addr_input_1, op.vector_length),
                ) {
                    return true;
                }
            }
            // WAW
            if overlap(
                (vec_op.addr_output, vec_op.vector_length),
                (op.addr_output, op.vector_length),
            ) {
                return true;
            }
        }
        false
    }

    pub fn get_prefetch_segments(
        &self,
        state: &Vec<(usize, usize, OutputType)>,
    ) -> Vec<(usize, usize)> {
        let mut res: Vec<(usize, usize)> = Vec::new();
        for op in self.vec_ops.iter() {
            if !state_cover(state, (op.addr_input_0, op.vector_length)) && op.addr_input_0 != 0 {
                res.push((op.addr_input_0, op.vector_length));
            }
            if let VecOpSrc::VV = op.op_src {
                if !state_cover(state, (op.addr_input_1, op.vector_length)) && op.addr_input_1 != 0
                {
                    res.push((op.addr_input_1, op.vector_length));
                }
            } else {
                if op.addr_input_1 != 0 {
                    res.push((op.addr_input_1, 1));
                }
            }
        }
        res
    }

    pub fn get_drain_segments(
        &self,
        state: &Vec<(usize, usize, OutputType)>,
    ) -> Vec<(usize, usize, OutputType)> {
        let mut res = Vec::new();
        for (start, size, is_output) in state.iter() {
            if *is_output == OutputType::Not || *start == 0 {
                continue;
            }
            if !state_cover(&self.reg_file_state, (*start, *size)) {
                res.push((*start, *size, *is_output));
            }
        }
        res
    }
}

#[derive(Clone)]
pub struct VectorChain {
    vec_ops: Vec<VecOpConfig>,
    vec_op_segs: Vec<Vec<usize>>,
    pub prefetch: Fetch,
    pub read_request: Request,
    pub write_request: Request,
    pub drain: Fetch,

    pub num_preload_elems: usize,
}

impl Kernel for VectorChain {
    fn create_prefetch(&mut self) {
        // println!("num_elems: {:?}", unsafe { ARCH_CONFIG.num_elems() });
        // println!("num_preload_elems: {:?}", self.num_preload_elems);
        let elems_capacity = unsafe { ARCH_CONFIG.num_elems() * 2 - self.num_preload_elems };

        let mut elems_set: HashMap<usize, (bool, bool)> = HashMap::new();
        let get_elems_to_insert =
            |addr: usize, size: usize, d: bool, elems_set: &mut HashMap<usize, (bool, bool)>| {
                let mut res = HashSet::new();
                if addr == 0 {
                    return res;
                }
                for i in 0..size {
                    let addr_ = addr + i * SIZE_F;
                    let set_contains = elems_set.contains_key(&addr_);
                    if !set_contains {
                        res.insert((addr_, !d, d));
                    }
                    if d && set_contains {
                        if let Some(tag) = elems_set.get_mut(&addr_) {
                            tag.1 = true;
                        }
                    }
                }
                res
            };
        let mut vec_op_seg = Vec::new();

        let mut update_self = |elems_set: &mut HashMap<usize, (bool, bool)>,
                               vec_op_seg: &mut Vec<usize>| {
            let elems_p = elems_set
                .iter()
                .filter(|(_, tag)| tag.0)
                .map(|(addr, _)| *addr)
                .collect();
            let elems_d = elems_set
                .iter()
                .filter(|(_, tag)| tag.1)
                .map(|(addr, _)| *addr)
                .collect();
            self.prefetch.push(vec![elems_p]);
            self.drain.push(vec![elems_d]);
            elems_set.clear();

            self.vec_op_segs.push(vec_op_seg.clone());
            vec_op_seg.clear();
        };

        for (i, vo) in self.vec_ops.iter().enumerate() {
            let in0 = get_elems_to_insert(vo.addr_input_0, vo.vector_length, false, &mut elems_set);
            let in1 = match vo.op_src {
                VecOpSrc::VV => {
                    get_elems_to_insert(vo.addr_input_1, vo.vector_length, false, &mut elems_set)
                }
                _ => get_elems_to_insert(vo.addr_input_1, 1, false, &mut elems_set),
            };
            let out = get_elems_to_insert(vo.addr_output, vo.vector_length, true, &mut elems_set);
            if elems_set.len() + in0.len() + in1.len() + out.len() > elems_capacity {
                update_self(&mut elems_set, &mut vec_op_seg);
            }
            for e in in0.iter().chain(in1.iter()).chain(out.iter()) {
                elems_set.insert(e.0, (e.1, e.2));
            }
            vec_op_seg.push(i);
        }
        if !elems_set.is_empty() {
            assert!(vec_op_seg.len() > 0);
            update_self(&mut elems_set, &mut vec_op_seg);
        }

        self.prefetch.mergable = true;
        self.prefetch.delay = vec![0; self.prefetch.len()];
        self.prefetch.interval = 0.0;
        self.drain.mergable = true;
        self.drain.delay = self
            .vec_op_segs
            .iter()
            .map(|x| x.iter().map(|y| self.vec_ops[*y].delay()).sum())
            .collect();
        self.drain.interval = 1.0;
    }

    fn create_drain(&mut self) {
        // merged in create_prefetch
    }

    fn create_read_request(&mut self) {}

    fn create_write_request(&mut self) {}

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
        // debug!("VectorChain: {:?}", self.vec_ops);
    }
    fn get_kernel_type(&self) -> String {
        String::from("Vector")
    }

    fn get_computation(&self) -> usize {
        let mut res = 0;
        for c in self.vec_ops.iter() {
            res += c.vector_length;
        }
        res
    }
}

impl VectorChain {
    pub fn new(vec_ops: Vec<VecOpConfig>, mem: &MemAlloc) -> VectorChain {
        let vec_ops_addr = vec_ops
            .iter()
            .map(|x| {
                let mut y = x.clone();
                if mem.preloaded(y.addr_input_0) {
                    y.addr_input_0 = 0;
                }
                if mem.preloaded(y.addr_input_1) {
                    y.addr_input_1 = 0;
                }
                if mem.preloaded(y.addr_output) {
                    y.addr_output = 0;
                }
                y
            })
            .collect::<Vec<_>>();
        let mut k = VectorChain {
            vec_ops: vec_ops_addr,
            vec_op_segs: Vec::new(),
            prefetch: Fetch::new(FetchType::Read),
            read_request: Request::new(),
            write_request: Request::new(),
            drain: Fetch::new(FetchType::Write),
            num_preload_elems: mem.num_preload_elems,
        };

        if unsafe { !ENABLE_CONFIG.other } {
            return k;
        }

        k.create_prefetch();
        k.create_drain();
        k.vec_ops = vec_ops;

        for vec_op_seg_idx in k.vec_op_segs.iter() {
            let vec_op_seg = vec_op_seg_idx
                .iter()
                .map(|x| k.vec_ops[*x].clone())
                .collect();
            let convoys = vector_chaining(&vec_op_seg);
            let read_segments = get_prefetch_segments(&convoys);
            let write_segments = get_drain_segments(&convoys);
            let read_lines = get_request_lines(read_segments);
            let write_lines = get_request_lines(write_segments);

            k.read_request.push(read_lines);
            k.write_request.push(write_lines);
        }

        k
    }
}

/// A concoy in the vector chain could contain at most one vector multiplication and two vector additions/subtractions.
/// A convoy could load two vectors from memory.
/// A convoy could store two vector to memory.
fn vector_chaining(vec_ops: &Vec<VecOpConfig>) -> Vec<Convoy> {
    // NUM_VEC_REGS vectors with 16 elements per each reg in register file
    let mut convoys: Vec<Convoy> = Vec::new();

    for op in vec_ops.iter() {
        let inserted = convoys.last_mut().map_or(false, |c| c.add_vec_op(*op));
        if !inserted {
            let mut new_convoy = Convoy::new();
            new_convoy.reg_file_state = convoys
                .last_mut()
                .map_or(vec![(0, 0, OutputType::Not); NUM_VEC_REGS], |c| {
                    c.reg_file_state.clone()
                });
            new_convoy.add_vec_op(*op);
            convoys.push(new_convoy);
        }
    }

    convoys
}

fn cover(a: (usize, usize), b: (usize, usize)) -> bool {
    if b.1 == 0 {
        return true;
    }
    a.0 <= b.0 && a.0 + a.1 * SIZE_F >= b.0 + b.1 * SIZE_F
}

fn state_cover(state: &Vec<(usize, usize, OutputType)>, reg: (usize, usize)) -> bool {
    for s in state.iter() {
        if cover((s.0, s.1), reg) {
            return true;
        }
    }
    false
}

fn overlap(a: (usize, usize), b: (usize, usize)) -> bool {
    max(a.0, b.0) < min(a.0 + a.1 * SIZE_F, b.0 + b.1 * SIZE_F)
}

fn get_prefetch_segments(convoys: &Vec<Convoy>) -> Vec<Vec<(usize, usize)>> {
    let mut res = Vec::new();
    let mut state = vec![(0, 0, OutputType::Not); NUM_VEC_REGS];
    for c in convoys.iter() {
        res.push(c.get_prefetch_segments(&state));
        state = c.reg_file_state.clone();
    }
    res
}

fn get_drain_segments(convoys: &Vec<Convoy>) -> Vec<Vec<(usize, usize)>> {
    let mut res = Vec::new();
    let mut state_ptr = convoys.iter().skip(1);
    for (i, c) in convoys.iter().enumerate() {
        let state = state_ptr
            .next()
            .unwrap_or(&Convoy::new())
            .reg_file_state
            .clone();
        let seg_old = c.get_drain_segments(&state);
        let mut seg = Vec::new();
        for (start, size, is_output) in seg_old.iter() {
            if *is_output == OutputType::Final {
                seg.push((*start, *size));
                continue;
            }
            for cp in convoys.iter().skip(i + 1) {
                if cp.overlap((*start, *size)) {
                    seg.push((*start, *size));
                    break;
                }
            }
        }
        res.push(seg);
    }
    res
}

fn get_request_lines(segments: Vec<Vec<(usize, usize)>>) -> Vec<Vec<usize>> {
    let al = unsafe { ARCH_CONFIG.array_length };
    segments
        .iter()
        .map(|segs| {
            segs.iter()
                .map(|(addr, size)| {
                    (0..*size)
                        .map(|x| addr + x * SIZE_F)
                        .collect::<Vec<_>>()
                        .chunks(al)
                        .map(|chunk| chunk.to_vec())
                        .collect::<Vec<Vec<usize>>>()
                })
                .flatten()
                .collect::<Vec<Vec<usize>>>()
        })
        .flatten()
        .collect::<Vec<Vec<usize>>>()
}

use crate::util::SIZE_F;

#[derive(Debug)]
pub struct MemBlock {
    pub id: String,
    pub start: usize,
    pub end: usize,
    pub size: usize,
    pub free: bool,
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct MemAlloc {
    pub size: usize,  // memory size in bytes
    pub align: usize, // align size in bytes
    pub blocks: Vec<MemBlock>,

    pub preload_vecs: Vec<(usize, usize)>, // preload vecs with start addr
    pub num_preload_elems: usize,
}

impl MemAlloc {
    pub fn new(size_gb: usize, align: usize) -> MemAlloc {
        let size = size_gb * 1024 * 1024 * 1024;
        let mut blocks = Vec::new();
        blocks.push(MemBlock {
            id: String::from("0"),
            start: 0,
            end: size,
            size: size,
            free: true,
            data: Vec::new(),
        });
        let mut mem = MemAlloc {
            size: size,
            align: align,
            blocks: blocks,
            preload_vecs: Vec::new(),
            num_preload_elems: 0,
        };
        mem.alloc("occupy", 8);
        mem
    }

    pub fn alloc(&mut self, id: &str, mut size: usize) -> Option<usize> {
        let id = String::from(id);
        for block in self.blocks.iter() {
            if block.id == id {
                panic!("same name block {:?} already allocated", id);
            }
        }

        if size % self.align != 0 {
            size = (size / self.align + 1) * self.align;
        }
        let mut pos = None;
        for (idx, block) in self.blocks.iter().enumerate() {
            if block.free && block.size >= size {
                pos = Some(idx);
                break;
            }
        }

        if pos.is_none() {
            panic!(
                "failed to allocate block {:?} with {:?} bytes,\ncurrent blocks {:?}",
                id, size, self.blocks
            );
        }
        let idx = pos.unwrap();
        let remain_size = self.blocks[idx].size - size;

        let new_block = MemBlock {
            id: id,
            start: self.blocks[idx].start,
            end: self.blocks[idx].start + size,
            size: size,
            free: false,
            data: Vec::with_capacity(size as usize),
        };

        if remain_size > 0 {
            let remain_block = MemBlock {
                id: String::from("0"),
                start: self.blocks[idx].start + size,
                end: self.blocks[idx].end,
                size: remain_size,
                free: true,
                data: Vec::new(),
            };
            self.blocks.insert(idx + 1, remain_block);
        }

        self.blocks[idx] = new_block;

        return Some(self.blocks[idx].start);
    }

    fn merge(&mut self, mut idx: usize) {
        if idx >= self.blocks.len() {
            panic!("index out of range");
        }

        if idx > 0 {
            if self.blocks[idx].free && self.blocks[idx - 1].free {
                self.blocks[idx - 1].end = self.blocks[idx].end;
                self.blocks[idx - 1].size += self.blocks[idx].size;
                self.blocks.remove(idx);
                idx = idx - 1;
            }
        }

        if idx < self.blocks.len() - 1 {
            if self.blocks[idx].free && self.blocks[idx + 1].free {
                self.blocks[idx].end = self.blocks[idx + 1].end;
                self.blocks[idx].size += self.blocks[idx + 1].size;
                self.blocks.remove(idx + 1);
            }
        }
    }

    pub fn free(&mut self, id: &str) {
        let id = String::from(id);
        let mut pos = None;
        for (idx, block) in self.blocks.iter().enumerate() {
            if block.id == id {
                pos = Some(idx);
                break;
            }
        }

        if pos.is_none() {
            panic!("block {:?} not found", id);
        }

        let idx = pos.unwrap();
        self.blocks[idx].id = String::from("0");
        self.blocks[idx].free = true;
        self.blocks[idx].data.clear();

        self.merge(idx);
    }

    pub fn get_addr(&self, id: &str) -> Option<usize> {
        let id = String::from(id);
        for block in self.blocks.iter() {
            if block.id == id {
                return Some(block.start);
            }
        }
        return None;
    }

    pub fn get_size(&self, id: &str) -> Option<usize> {
        let id = String::from(id);
        for block in self.blocks.iter() {
            if block.id == id {
                return Some(block.size);
            }
        }
        return None;
    }

    pub fn read(&self, addr: usize, mut size: usize) -> Vec<u8> {
        let start = addr;
        let end = addr + size;

        let mut data = Vec::new();

        for block in self.blocks.iter() {
            if block.free {
                continue;
            }
            let max_start = std::cmp::max(start, block.start);
            let min_end = std::cmp::min(end, block.end);
            data.extend(&block.data[(max_start - block.start)..(min_end - block.start)]);
            size -= min_end - max_start;
            if size == 0 {
                break;
            }
        }

        if size > 0 {
            panic!("read from unallocated memory");
        }
        return data;
    }

    pub fn write(&mut self, addr: usize, data: Vec<u8>) {
        let start = addr;
        let end = addr + data.len();
        let mut size = data.len();

        for block in self.blocks.iter_mut() {
            if block.free {
                continue;
            }
            let max_start = std::cmp::max(start, block.start);
            let min_end = std::cmp::min(end, block.end);
            block.data[(max_start - block.start)..(min_end - block.start)]
                .copy_from_slice(&data[(max_start - start)..(min_end - start)]);
            size -= min_end - max_start;
        }

        if size > 0 {
            panic!("write to unallocated memory");
        }
    }

    pub fn get_name(&self, addr: usize) -> String {
        for block in self.blocks.iter() {
            if block.start <= addr && addr < block.end {
                return block.id.clone();
            }
        }
        return String::from("0");
    }

    // size in number of elements
    pub fn preload(&mut self, addr: usize, size: usize) {
        // binary insert
        let mut left = 0;
        let mut right = self.preload_vecs.len();
        while left < right {
            let mid = left + (right - left) / 2;
            if self.preload_vecs[mid].0 < addr {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        self.preload_vecs.insert(left, (addr, size * SIZE_F));
        self.num_preload_elems += size;
    }
    pub fn preloaded(&self, addr: usize) -> bool {
        if self.preload_vecs.len() == 0 {
            return false;
        }
        // binary search
        let mut left = 0;
        let mut right = self.preload_vecs.len();
        while left < right {
            let mid = left + (right - left) / 2;
            if self.preload_vecs[mid].0 < addr {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if left < self.preload_vecs.len() {
            if addr == self.preload_vecs[left].0 {
                return true;
            }
        }
        if left == 0 {
            return false;
        }
        left -= 1;
        if addr >= self.preload_vecs[left].0
            && addr < self.preload_vecs[left].0 + self.preload_vecs[left].1
        {
            return true;
        }

        return false;
    }
    pub fn unpreload(&mut self, addr: usize) {
        let mut left = 0;
        let mut right = self.preload_vecs.len();
        while left < right {
            let mid = left + (right - left) / 2;
            if self.preload_vecs[mid].0 < addr {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if left >= self.preload_vecs.len() {
            return;
        }
        if addr == self.preload_vecs[left].0 {
            self.num_preload_elems -= self.preload_vecs[left].1 / SIZE_F;
            self.preload_vecs.remove(left);
            return;
        }
    }
    pub fn clear_preload(&mut self) {
        self.preload_vecs.clear();
        self.num_preload_elems = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mem_alloc() {
        let mut ma = MemAlloc::new(1024, 64);
        let addr = ma.alloc("1", 128);
        let addr_2 = ma.alloc("2", 128);
        let addr_3 = ma.alloc("3", 128);
        let addr_4 = ma.alloc("4", 1024);
        println!("mem state is {:?}", ma);
        ma.free("2");
        println!("mem state is {:?}", ma);
        ma.free("1");
        println!("mem state is {:?}", ma);
        ma.free("3");
        println!("mem state is {:?}", ma);
        let addr_5 = ma.alloc("5", 1024);
        assert_eq!(addr, Some(0));
        assert_eq!(addr_2, Some(128));
        assert_eq!(addr_3, Some(256));
        assert_eq!(addr_4, None);
        assert_eq!(addr_5, Some(0));
    }
}

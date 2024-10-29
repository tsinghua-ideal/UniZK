use crate::trace::trace::FetchType;
use crate::util::BUFSIZE;
use anyhow::{Ok, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufWriter;
use std::io::SeekFrom;

pub struct OpRecord {
    pub id: u64,
    pub addr: u64,
    pub fetch_type: FetchType,
    pub delay: u32,
    pub dependencies: Vec<u64>,
    pub size: u32,
}
impl OpRecord {
    pub fn default() -> Self {
        Self {
            id: 0,
            fetch_type: FetchType::Read,
            delay: 0,
            addr: 0,
            dependencies: Vec::new(),
            size: 0,
        }
    }
}

impl OpRecord {
    pub const SIZE: u32 = 64; // 64Bytes
}

pub struct RamConfig {
    pub file_name: String,
    pub excutable_path: String,
    pub config_path: String,
    pub burst_length: usize,

    num_current_ops: u64,
    penultimate_read_id: i64,
    penultimate_write_id: i64,
    last_read_id: i64,
    last_write_id: i64,
    pub op_cnt: usize,

    bin_file: BufWriter<File>,
    txt_file: File,
    log_file: File,

    pub txt_output: bool,

    pub length_static: HashMap<usize, usize>,
}

impl RamConfig {
    pub fn new(file_name: &str) -> Self {
        let name = format!("./traces/{}.bin", file_name);
        let file = File::create(name).unwrap();
        let mut writer = std::io::BufWriter::new(file);

        let magic_word = String::from("BINFILE");
        let _ = writer.write_all(&to_buf(magic_word.as_bytes()));

        let name = format!("./traces/{}.txt", file_name);
        let file = File::create(name).unwrap();

        let name = format!("{}.log", file_name);
        let log_file = File::create(name).unwrap();

        Self {
            file_name: file_name.to_string(),
            excutable_path: "./thirdparty/ramulator2/build/ramulator2".to_string(),
            config_path: "./configs/".to_string() + file_name + ".yaml",
            burst_length: 64, //bytes
            num_current_ops: 0,
            penultimate_read_id: -2,
            penultimate_write_id: -2,
            last_read_id: -1,
            last_write_id: -1,
            op_cnt: 0,

            bin_file: writer,
            txt_file: file,
            log_file: log_file,
            txt_output: false,
            length_static: HashMap::new(),
        }
    }

    pub fn reset(&mut self) {
        self.num_current_ops = 0;
        self.penultimate_read_id = -2;
        self.penultimate_write_id = -2;
        self.last_read_id = -1;
        self.last_write_id = -1;
        self.op_cnt = 0;
        self.length_static.clear();

        self.bin_file.flush().unwrap();
        self.bin_file
            .seek(SeekFrom::Start(0))
            .expect("reset bin file failed");
        let magic_word = String::from("BINFILE");
        self.bin_file
            .write_all(&to_buf(magic_word.as_bytes()))
            .unwrap();
        // let name = format!("./traces/{}.bin", self.file_name);
        // let file = File::create(name).unwrap();
        // let mut writer = std::io::BufWriter::new(file);
        // let magic_word = String::from("BINFILE");
        // let _ = writer.write_all(&to_buf(magic_word.as_bytes()));
        // self.bin_file = writer;

        self.txt_file.set_len(0).expect("reset txt file failed");
    }

    pub fn add_trace(
        &mut self,
        mut read_op: Vec<OpRecord>,
        mut write_op: Vec<OpRecord>,
    ) -> Result<()> {
        read_op.iter_mut().for_each(|op| {
            op.id += self.num_current_ops;
            for dep in op.dependencies.iter_mut() {
                *dep += self.num_current_ops;
            }
        });
        write_op.iter_mut().for_each(|op| {
            op.id += self.num_current_ops;
            for dep in op.dependencies.iter_mut() {
                *dep += self.num_current_ops;
            }
        });
        self.num_current_ops += read_op.len() as u64 + write_op.len() as u64;

        for op in read_op.iter_mut() {
            if self.penultimate_write_id >= 0 {
                op.dependencies.push(self.penultimate_write_id as u64);
            }
            // if self.last_read_id >= 0 {
            //     op.dependencies.push(self.last_read_id as u64);
            // }
        }

        if let Some(op) = write_op.last() {
            self.penultimate_write_id = self.last_write_id;
            self.last_write_id = op.id as i64;
        }

        if let Some(op) = read_op.last() {
            self.penultimate_read_id = self.last_read_id;
            self.last_read_id = op.id as i64;
        }

        // for op in read_op.iter_mut().chain(write_op.iter_mut()) {
        //     if op.size <= 1024 {
        //         op.dependencies.clear();
        //     }
        // }

        self.write_trace(read_op, write_op)?;

        Ok(())
    }

    pub fn write_trace(&mut self, read_op: Vec<OpRecord>, write_op: Vec<OpRecord>) -> Result<()> {
        for op in read_op.iter().chain(write_op.iter()) {
            assert_eq!(op.id as usize, self.op_cnt);
            self.bin_file.write_all(&to_buf(&op.id.to_le_bytes()))?;
            self.bin_file.write_all(&to_buf(&op.addr.to_le_bytes()))?;
            self.bin_file
                .write_all(&to_buf(&(op.fetch_type as u32).to_le_bytes()))?;
            self.bin_file.write_all(&to_buf(&op.delay.to_le_bytes()))?;
            // self.bin_file.write_all(&to_buf(&0u64.to_le_bytes()))?;
            self.bin_file.write_all(&to_buf(&op.size.to_le_bytes()))?;
            self.bin_file
                .write_all(&to_buf(&(op.dependencies.len() as u64).to_le_bytes()))?;
            for dep in op.dependencies.iter() {
                self.bin_file.write_all(&to_buf(&dep.to_le_bytes()))?;
            }
            // self.bin_file
            //     .write_all(&to_buf(&(0 as u64).to_le_bytes()))?;
            self.op_cnt += 1;
        }

        if self.txt_output {
            for op in read_op.iter().chain(write_op.iter()) {
                self.txt_file
                    .write_all(format!("id: {}\n", op.id).as_bytes())?;
                self.txt_file
                    .write_all(format!("addr: {}\n", op.addr).as_bytes())?;
                self.txt_file
                    .write_all(format!("fetch_type: {:?}\n", op.fetch_type).as_bytes())?;
                self.txt_file
                    .write_all(format!("delay: {}\n", op.delay).as_bytes())?;
                self.txt_file
                    .write_all(format!("size: {}\n", op.size).as_bytes())?;
                self.txt_file
                    .write_all(format!("dependencies: ").as_bytes())?;
                for dep in op.dependencies.iter() {
                    self.txt_file.write_all(format!("{}, ", dep).as_bytes())?;
                }
                self.txt_file.write_all(format!("\n").as_bytes())?;
            }
        }

        for op in read_op.iter().chain(write_op.iter()) {
            let size = op.size as usize;
            let lg_size = size.next_power_of_two().trailing_zeros() as usize;
            *self.length_static.entry(lg_size).or_insert(0) += 1;
        }
        Ok(())
    }

    pub fn static_(&self) {
        let mut total = 0;
        for (lg_size, cnt) in self.length_static.iter() {
            total += cnt * (1 << lg_size);
        }
        println!("total: {}", total);
        // for (lg_size, cnt) in self.length_static.iter() {
        //     println!("{}: {}, {}%", lg_size, cnt, cnt * (1 << lg_size) * 100 / total);
        // }
        let max_lg_size = self
            .length_static
            .iter()
            .max()
            .unwrap_or((&0, &0))
            .0
            .clone();
        let mut total_percent = 0;
        for i in 0..=max_lg_size {
            if i == 7 {
                println!("----------------64B-----------------")
            }
            if let Some(cnt) = self.length_static.get(&i) {
                let perc = cnt * (1 << i) * 100 / total;
                total_percent += perc;
                println!("{}: {}, {}%, in {}%", i, cnt, perc, total_percent);
            }
        }
    }

    pub fn run(&mut self) {
        self.bin_file.flush().unwrap();
        let output = std::process::Command::new(&self.excutable_path)
            .arg("-f")
            .arg(&self.config_path)
            .output();
        let output = output.unwrap();
        self.log_file.write_all(&output.stdout).unwrap();
    }
}

fn to_buf(a: &[u8]) -> [u8; BUFSIZE] {
    let mut buf = [0; 8];
    for i in 0..a.len() {
        buf[i] = a[i];
    }
    buf
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    #[test]
    fn test_ram_config() -> Result<()> {
        const RAMDOM: bool = true;
        const CONSECUTIVE_LENGTH: usize = 1 << 2;
        const NUM_REQUESTS: usize = (1 << 22) / CONSECUTIVE_LENGTH;
        let mut rng = rand::thread_rng();
        let addrs_ramdom = (0..NUM_REQUESTS)
            .map(|_| rng.gen_range(0..1 << 63))
            .collect::<Vec<_>>();
        let addrs_stream = {
            let base_addr = rng.gen_range(0..1 << 63);
            (0..NUM_REQUESTS)
                .map(|i| base_addr + i as u64 * OpRecord::SIZE as u64 * CONSECUTIVE_LENGTH as u64)
                .collect::<Vec<_>>()
        };
        let mut ram = RamConfig::new("test_random_4");
        // ram.txt_output = true;
        let mut cnt = 0;
        for addr in if RAMDOM { addrs_ramdom } else { addrs_stream } {
            let op = OpRecord {
                id: cnt,
                addr: addr,
                fetch_type: if rng.gen_range(0..4) == 0 {
                    FetchType::Write
                } else {
                    FetchType::Read
                },
                delay: 0,
                dependencies: Vec::new(),
                size: CONSECUTIVE_LENGTH as u32 * OpRecord::SIZE,
            };
            cnt += 1;
            ram.write_trace(vec![op], vec![])?;
        }
        ram.static_();
        // ram.run();
        Ok(())
    }

    #[test]
    fn test_stride() -> Result<()> {
        const WIDTH: usize = (1 << 20);
        const HEIGHT: usize = 400;
        const STRIDE: usize = 1 << 12;
        let mut rng = rand::thread_rng();
        let base_addr = 0; //rng.gen_range(0..1 << 63);
        let mut ram = RamConfig::new("test_stride");
        // ram.txt_output = true;
        let mut cnt = 0;
        for col in (0..WIDTH).step_by(STRIDE) {
            for row in 0..HEIGHT {
                let addr = base_addr + (row * WIDTH + col) as u64;
                let op = OpRecord {
                    id: cnt,
                    addr: addr,
                    fetch_type: FetchType::Read,
                    delay: 0,
                    dependencies: Vec::new(),
                    size: STRIDE as u32,
                };
                cnt += 1;
                ram.write_trace(vec![op], vec![])?;
            }
        }
        // for col in (0..WIDTH).step_by(STRIDE) {
        //     for row in 0..HEIGHT {
        //         let row_addr = (col + row * STRIDE) % WIDTH;
        //         let addr = (row * WIDTH + row_addr) as u64;
        //         let op = OpRecord {
        //             id: cnt,
        //             addr: addr,
        //             fetch_type: FetchType::Read,
        //             delay: 0,
        //             dependencies: Vec::new(),
        //             size: STRIDE as u32,
        //         };
        //         cnt += 1;
        //         ram.write_trace(vec![op], vec![])?;
        //     }
        // }
        ram.static_();
        // ram.run();
        Ok(())
    }
}

use crate::trace::trace::{Fetch, FetchType, Request};
use crate::util::SIZE_F;
use plonky2::field::goldilocks_field::GoldilocksField as F;
use plonky2::field::types::{Field, PrimeField64};

pub trait Kernel {
    fn init(&mut self) {
        self.create_prefetch();
        self.create_drain();
        self.create_read_request();
        self.create_write_request();
    }
    fn create_drain(&mut self);
    fn create_prefetch(&mut self);
    fn create_read_request(&mut self);
    fn create_write_request(&mut self);
    fn to_word(&self, data: &Vec<u8>) -> F {
        let mut word = 0u64;
        for i in 0..data.len() {
            word |= (data[i] as u64) << (i * 8);
        }

        F::from_canonical_u64(word)
    }
    fn to_bytes(&self, data: &F) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(SIZE_F);
        let word = F::to_canonical_u64(data);
        for i in 0..SIZE_F {
            bytes.push(((word >> (i * 8)) & 0xff) as u8);
        }
        bytes
    }

    fn get_prefetch(&self) -> Fetch {
        Fetch::new(FetchType::Read)
    }
    fn get_drain(&self) -> Fetch {
        Fetch::new(FetchType::Write)
    }
    fn get_read_request(&self) -> Request {
        Request::new()
    }
    fn get_write_request(&self) -> Request {
        Request::new()
    }
    fn get_computation(&self) -> usize;
    fn get_kernel_type(&self) -> String;
    fn log(&self);
}

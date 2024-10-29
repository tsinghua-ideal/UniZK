const RCON: [u32; 10] = [
    0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000, 0x80000000,
    0x1b000000, 0x36000000,
];

// 获取最高位
fn get_highest_position(mut number: u16) -> i32 {
    let mut i = 0;
    while number != 0 {
        i += 1;
        number >>= 1;
    }
    i
}

// GF(2^8)的多项式除法
fn division(num_l: u16, num_r: u16) -> u8 {
    let mut r0 = num_l;
    let mut q = 0u8;
    let mut bit_count = get_highest_position(r0) - get_highest_position(num_r);

    while bit_count >= 0 {
        q |= 1 << bit_count;
        r0 ^= num_r << bit_count;
        bit_count = get_highest_position(r0) - get_highest_position(num_r);
    }

    q
}

// GF(2^8)多项式乘法
fn multiplication(num_l: u8, num_r: u8) -> u16 {
    let mut result = 0u16;

    for i in 0..8 {
        result ^= u16::from((num_l >> i) & 0x01) * (u16::from(num_r) << i);
    }

    result
}

// 伽罗瓦域的扩展欧几里得算法
fn eea_v2(mut r0: i32, mut r1: i32) -> i32 {
    let mod_value = r0;
    let mut t0 = 0;
    let mut t1 = 1;
    let mut t = t1;

    if r1 == 0 {
        return 0;
    }

    while r1 != 1 {
        let q = division(r0 as u16, r1 as u16);

        let r = r0 ^ (multiplication(q, r1 as u8) as i32);
        t = t0 ^ (multiplication(q, t1 as u8) as i32);

        r0 = r1;
        r1 = r;
        t0 = t1;
        t1 = t;
    }

    if t < 0 {
        t ^= mod_value;
    }

    t
}

pub struct AESwcs {
    key: Vec<u32>,
    round: usize,
}

impl AESwcs {
    pub fn new(key: &[u8; 16]) -> Self {
        let mut key32 = Vec::with_capacity(4);
        for i in 0..4 {
            key32.push(u32::from_be_bytes([
                key[4 * i],
                key[4 * i + 1],
                key[4 * i + 2],
                key[4 * i + 3],
            ]));
        }

        AESwcs {
            key: key32,
            round: 0,
        }
    }

    fn key_update(&mut self) {
        let mut temp = self.key[3];
        temp = Self::rot_word(temp);
        temp = Self::sub_word(temp);
        temp ^= RCON[self.round];
        temp ^= self.key[0];
        self.key[0] = temp;
        self.key[1] ^= self.key[0];
        self.key[2] ^= self.key[1];
        self.key[3] ^= self.key[2];
        self.round += 1;
    }

    fn sub_word(word: u32) -> u32 {
        let b0 = Self::compute_sbox((word >> 24) as u8) as u32;
        let b1 = Self::compute_sbox(((word >> 16) & 0xff) as u8) as u32;
        let b2 = Self::compute_sbox(((word >> 8) & 0xff) as u8) as u32;
        let b3 = Self::compute_sbox((word & 0xff) as u8) as u32;
        (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
    }

    fn rot_word(word: u32) -> u32 {
        (word << 8) | (word >> 24)
    }

    fn sub_bytes(state: &mut [u8; 16]) {
        for i in 0..16 {
            state[i] = Self::compute_sbox(state[i] as u8);
        }
    }

    pub fn compute_sbox(a: u8) -> u8 {
        let r0 = 0x11b;
        let b = (eea_v2(r0, a as i32) & 0xFF) as u8;

        let res = (0..8)
            .map(|i| {
                (((b >> i) & 1)
                    ^ ((b >> ((i + 4) % 8)) & 1)
                    ^ ((b >> ((i + 5) % 8)) & 1)
                    ^ ((b >> ((i + 6) % 8)) & 1)
                    ^ ((b >> ((i + 7) % 8)) & 1))
                    << i
            })
            .fold(0x63, |acc, x| acc ^ x);
        res
    }

    fn shift_rows(state: &mut [u8; 16]) {
        let temp = state[1];
        state[1] = state[5];
        state[5] = state[9];
        state[9] = state[13];
        state[13] = temp;

        let temp = state[2];
        state[2] = state[10];
        state[10] = temp;
        let temp = state[6];
        state[6] = state[14];
        state[14] = temp;

        let temp = state[3];
        state[3] = state[15];
        state[15] = state[11];
        state[11] = state[7];
        state[7] = temp;
    }

    fn mix_columns(state: &mut [u8; 16]) {
        for i in 0..4 {
            let s0 = state[4 * i];
            let s1 = state[4 * i + 1];
            let s2 = state[4 * i + 2];
            let s3 = state[4 * i + 3];

            state[4 * i] = Self::gmul(2, s0) ^ Self::gmul(3, s1) ^ s2 ^ s3;
            state[4 * i + 1] = s0 ^ Self::gmul(2, s1) ^ Self::gmul(3, s2) ^ s3;
            state[4 * i + 2] = s0 ^ s1 ^ Self::gmul(2, s2) ^ Self::gmul(3, s3);
            state[4 * i + 3] = Self::gmul(3, s0) ^ s1 ^ s2 ^ Self::gmul(2, s3);
        }
    }

    fn gmul(a: u8, b: u8) -> u8 {
        let mut p = 0;
        let mut hi_bit_set;
        let mut counter = 0;
        let mut temp_a = a;
        let mut temp_b = b;
        while (temp_a != 0 || temp_b != 0) && counter < 8 {
            if temp_b & 1 == 1 {
                p ^= temp_a;
            }
            hi_bit_set = (temp_a & 0x80) != 0;
            temp_a <<= 1;
            if hi_bit_set {
                temp_a ^= 0x1b;
            }
            temp_b >>= 1;
            counter += 1;
        }
        p
    }

    fn add_round_key(&mut self, state: &mut [u8; 16]) {
        for i in 0..4 {
            let rk = self.key[i].to_be_bytes();
            for j in 0..4 {
                state[4 * i + j] ^= rk[j];
            }
        }

        if self.round < 10 {
            self.key_update();
        }
    }

    pub fn encrypt(&mut self, plaintext: &[u8; 16]) -> [u8; 16] {
        let mut state = *plaintext;

        self.add_round_key(&mut state);

        for _ in 1..10 {
            Self::sub_bytes(&mut state);
            Self::shift_rows(&mut state);
            Self::mix_columns(&mut state);
            self.add_round_key(&mut state);
        }

        Self::sub_bytes(&mut state);
        Self::shift_rows(&mut state);
        self.add_round_key(&mut state);

        state
    }
}

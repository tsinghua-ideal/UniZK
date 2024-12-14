use std::collections::{HashMap, LinkedList};

use crate::util::SIZE_F;

type K = usize;
type V = usize;

pub struct LRUcache {
    capacity: usize,
    map: HashMap<K, V>,
    recent_keys: LinkedList<K>,
    used: usize,
}

impl LRUcache {
    pub fn new(capacity: usize) -> Self {
        LRUcache {
            capacity,
            map: HashMap::new(),
            recent_keys: LinkedList::new(),
            used: 0,
        }
    }

    pub fn get_capacity(&self) -> usize {
        self.capacity
    }

    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.map.contains_key(key) {
            self.mark_as_recent(key);
            self.map.get(key)
        } else {
            None
        }
    }

    pub fn put(&mut self, key: K, value: V, wo_load: bool) -> (Vec<(K, V)>, Vec<(K, V)>) {
        if self.map.contains_key(&key) {
            let old_value = *self.map.get(&key).unwrap();
            if old_value >= value {
                self.mark_as_recent(&key);
                (Vec::new(), Vec::new())
            } else {
                self.mark_as_recent(&key);
                (
                    vec![(key + old_value * SIZE_F, value - old_value)],
                    Vec::new(),
                )
            }
        } else {
            assert!(value <= self.capacity);
            let mut evicted = Vec::new();
            while self.used + value > self.capacity {
                let tmp = self.recent_keys.pop_back();
                if tmp.is_none() {
                    println!("capacity: {}", self.capacity);
                    println!("used: {}", self.used);
                    println!("value: {}", value);
                    println!("recent_keys: {:?}", self.recent_keys);
                }
                let oldest_key = tmp.unwrap(); //self.recent_keys.pop_back().unwrap();
                self.used -= self.map.get(&oldest_key).unwrap();
                let evicted_value = self.map.remove(&oldest_key).unwrap();
                evicted.push((oldest_key, evicted_value));
            }
            self.map.insert(key.clone(), value);
            self.recent_keys.push_front(key);
            self.used += value;

            let load = if wo_load { vec![] } else { vec![(key, value)] };
            (load, evicted)
        }
    }

    fn mark_as_recent(&mut self, key: &K) {
        let position = self.recent_keys.iter().position(|x| x == key).unwrap();
        self.recent_keys.remove(position);
        self.recent_keys.push_front(key.clone());
    }

    pub fn drain(&mut self) -> Vec<(K, V)> {
        let mut res = Vec::new();
        while let Some(key) = self.recent_keys.pop_back() {
            let value = self.map.remove(&key).unwrap();
            res.push((key, value));
        }
        res
    }

    fn print(&self) {
        for key in self.recent_keys.iter() {
            println!("{}: {}", key, self.map.get(key).unwrap());
        }
        println!("Used: {}", self.used);
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru() {
        let mut cache = LRUcache::new(30);
        cache.put(0, 10, false);
        cache.put(10, 10, false);
        cache.put(20, 10, false);
        cache.get(&0);
        cache.get(&10);
        cache.put(30, 10, false);

        cache.print();
    }
}

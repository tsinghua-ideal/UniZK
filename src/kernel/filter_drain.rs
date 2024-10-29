use crate::trace::trace::Fetch;

fn filter_lines(
    exist: &Vec<(u64, u64)>,
    next_prefetch: &mut Vec<(u64, u64)>,
    load: &mut Vec<(u64, u64)>,
) {
    let mut line_idx = 0;
    loop {
        if line_idx >= next_prefetch.len() {
            break;
        }
        for (start, end) in exist {
            let line = next_prefetch[line_idx];
            let overlap_start = line.0.max(*start);
            let overlap_end = (line.1).min(*end);
            if overlap_start <= overlap_end {
                if overlap_start == line.0 && overlap_end == line.1 {
                    next_prefetch.remove(line_idx);
                    break;
                } else if overlap_start == line.0 && overlap_end < line.1 {
                    next_prefetch[line_idx] = (overlap_end + 1, line.1);
                } else if overlap_start > line.0 && overlap_end == line.1 {
                    next_prefetch[line_idx] = (line.0, overlap_start - 1);
                } else {
                    next_prefetch[line_idx] = (line.0, overlap_start - 1);
                    next_prefetch.push((overlap_end + 1, line.1));
                }
                load.push((overlap_start, overlap_end));
            }
        }
        line_idx += 1;
    }
}

pub fn merge(prefetch: &mut Fetch, drain: &mut Fetch) {
    assert_eq!(drain.mergable, prefetch.mergable);
    if !drain.mergable {
        return;
    }

    let mut preload_lines = Vec::new();

    for j in 0..drain.len().max(prefetch.len()) {
        if j + 1 >= prefetch.len() {
            return;
        };

        let mut load_lines = Vec::new();

        filter_lines(
            &prefetch.addr[j].clone(),
            &mut prefetch.addr[j + 1],
            &mut load_lines,
        );
        if j < drain.len() {
            filter_lines(&drain.addr[j], &mut prefetch.addr[j + 1], &mut load_lines);
            filter_lines(&load_lines, &mut drain.addr[j], &mut Vec::new());
        }
        filter_lines(&preload_lines, &mut prefetch.addr[j + 1], &mut load_lines);
        preload_lines = load_lines.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::trace::FetchType;

    #[test]
    fn test_merge() {
        let mut prefetch0 = Fetch::new(FetchType::Read);
        let mut drain0 = Fetch::new(FetchType::Read);
        prefetch0.mergable = true;
        drain0.mergable = true;

        for i in 0..3 {
            let mut line = Vec::new();
            for j in 0..4 {
                line.push((0..12).map(|k| j + i + k * 4).collect());
            }
            prefetch0.push(line.clone());
            let mut line = Vec::new();
            for j in 0..4 {
                line.push((0..12).map(|k| j + i + 1 + k * 4).collect());
            }
            drain0.push(line.clone());
        }

        println!("prefetch0 {:?}", prefetch0);
        println!("drain0 {:?}", drain0);
        merge(&mut prefetch0, &mut drain0);
        println!("prefetch0 {:?}", prefetch0);
        println!("drain0 {:?}", drain0);
    }
}

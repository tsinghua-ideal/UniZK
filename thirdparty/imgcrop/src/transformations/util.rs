pub(crate) fn pixel_number_to_coords(num: usize, width: u32) -> (u32, u32) {
    (num as u32 % width, num as u32 / width)
}

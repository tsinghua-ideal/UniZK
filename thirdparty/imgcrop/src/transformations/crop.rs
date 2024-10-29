use plonky2::iop::target::Target;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2_field::goldilocks_field::GoldilocksField;

use super::util::pixel_number_to_coords;

pub(crate) fn build_crop_circuit<const L: usize>(
    builder: &mut CircuitBuilder<GoldilocksField, 2>,
    original_chunk: &[Target],
    edited_chunk: &[Target],
    chunk_number: usize,
    orig_w: u32,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) {
    let lx_bound = x;
    let rx_bound = x + w;
    let uy_bound = y;
    let dy_bound = y + h;
    for (i, (orig_pixel, edit_pixel)) in original_chunk.iter().zip(edited_chunk).enumerate() {
        let (x, y) = pixel_number_to_coords(chunk_number * L + i, orig_w);
        if x >= lx_bound && x < rx_bound && y >= uy_bound && y < dy_bound {
            builder.is_equal(*orig_pixel, *edit_pixel);
        } else {
            builder.assert_zero(*edit_pixel);
        }
    }
}

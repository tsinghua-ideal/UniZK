use env_logger::Env;
use image::imageops::crop;
use log::info;
use unizk::util::set_config;
use zkedit_zkp::builder::TransformationCircuitBuilder;
use zkedit_zkp::zk_transformations::Transformation;

use image::io::Reader as ImageReader;
use image::ImageBuffer;
use image::{buffer::Pixels, Rgba};
use plonky2::iop::generator::generate_partial_witness;
use unizk::config::arch_config::ARCH_CONFIG;
use unizk::config::enable_config::ENABLE_CONFIG;
use unizk::config::RamConfig;
use unizk::memory::memory_allocator::MemAlloc;
use unizk::plonk::prover::prove_with_partition_witness;
use unizk::system::system::System;

use clap::{value_parser, Arg, Command};

fn pixels_to_bytes(pixels: Pixels<Rgba<u8>>) -> Vec<u8> {
    let mut pixel_bytes = vec![];
    for pixel in pixels {
        pixel_bytes.extend_from_slice(&pixel.0)
    }
    pixel_bytes
}

fn align_crop_edit(
    crop_img: ImageBuffer<Rgba<u8>, Vec<u8>>,
    orig_width: u32,
    orig_height: u32,
    crop_x: u32,
    crop_y: u32,
    crop_width: u32,
    crop_height: u32,
) -> Vec<u8> {
    let lx_bound = crop_x;
    let rx_bound = crop_x + crop_width;
    let uy_bound = crop_y;
    let dy_bound = crop_y + crop_height;

    let mut aligned_image = ImageBuffer::new(orig_width, orig_height);
    for (x, y, pixel) in aligned_image.enumerate_pixels_mut() {
        if x >= lx_bound && x < rx_bound && y >= uy_bound && y < dy_bound {
            *pixel = crop_img.get_pixel(x - crop_x, y - crop_y).clone();
        }
    }

    pixels_to_bytes(aligned_image.pixels())
}

const L: usize = 12 * 85 * 256 * 8;

fn main() {
    set_config();

    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let ramsim = RamConfig::new(&format!("{}", "crop"));
    // ramsim.txt_output = true;
    let mem = MemAlloc::new(256, 4096);
    let mut sys = System::new(mem, ramsim);

    info!("Starting simulator");
    let mut img = ImageReader::open("./random_color.png")
        .unwrap()
        .decode()
        .unwrap()
        .into_rgba8();
    let pixels = img.pixels();
    let width = img.width();
    let height = img.height();
    println!("Image width: {}, height: {}", width, height);

    let (x, y, w, h) = (55, 30, 512, 512);

    let pixel_bytes = pixels_to_bytes(pixels);
    let crop_img = crop(&mut img, x, y, w, h).to_image();
    let mut crop_bytes = pixels_to_bytes(crop_img.pixels());
    crop_bytes.extend(std::iter::repeat(0).take(pixel_bytes.len() - crop_bytes.len()));

    let crop_transformation = Transformation::Crop {
        orig_w: width,
        orig_h: height,
        x: x,
        y: y,
        w: w,
        h: h,
    };
    let alidned_crop_bytes = align_crop_edit(crop_img, width, height, x, y, w, h);

    let builder = TransformationCircuitBuilder::<L>::new(
        pixel_bytes.len(),
        Box::new(crop_transformation.clone()),
    );
    let mut circuit = builder.build_curcuit();

    let proofs = circuit.get_circuit(&pixel_bytes, &alidned_crop_bytes);

    assert!(proofs.len() == 1, "Only one proof is expected");

    for (data, pw) in proofs {
        let partition_witness = generate_partial_witness(pw, &data.prover_only, &data.common);
        prove_with_partition_witness(&mut sys, &data.prover_only, &data.common, partition_witness);
        // sys.mem.clean();
    }

    info!("Total number of memreq: {}", sys.ramsim.op_cnt);
    info!("Total number of operations: {:?}", sys.get_computation());

    info!("Simulator finished");
    sys.ramsim.static_();

    info!("Start Ramsim");
    sys.ramsim.run();
}

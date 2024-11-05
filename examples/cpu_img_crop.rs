use anyhow::{Context, Result};
use std::fs;
use std::time::Instant;

use image::imageops::crop;
use image::ImageBuffer;
use log::info;
use log::LevelFilter;
use zkedit_zkp::img_crop_lib::cli::{parse_options, Zkedit};
use zkedit_zkp::img_crop_lib::metadata::ProofMetadata;

use image::io::Reader as ImageReader;
use image::{buffer::Pixels, Rgba};

use zkedit_zkp::builder::TransformationCircuitBuilder;
use zkedit_zkp::util::calculate_poseidon;
use zkedit_zkp::zk_transformations::Transformation;

use maybe_rayon::rayon;

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

fn prove_crop(
    orig_img_path: String,
    crop_x: u32,
    crop_y: u32,
    crop_w: u32,
    crop_h: u32,
) -> Result<()> {
    let mut img = ImageReader::open(orig_img_path)?.decode()?.into_rgba8();
    let pixels = img.pixels();
    let width = img.width();
    let height = img.height();
    let bytes_length = width * height * 4;

    println!(
        "Read image {}x{} pixels. {}B, {}kB, {}mB",
        width,
        height,
        bytes_length,
        bytes_length / 1024,
        bytes_length / (1024 * 1024)
    );

    let pixel_bytes = pixels_to_bytes(pixels);
    let pixels_hash = calculate_poseidon(&pixel_bytes);

    let crop_img = crop(&mut img, crop_x, crop_y, crop_w, crop_h).to_image();

    println!(
        "Crop image {}x{} pixels",
        crop_img.width(),
        crop_img.height()
    );

    let mut crop_bytes = pixels_to_bytes(crop_img.pixels());
    crop_bytes.extend(std::iter::repeat(0).take(pixel_bytes.len() - crop_bytes.len()));

    crop_img.save("img_crop.png");

    let alidned_crop_bytes =
        align_crop_edit(crop_img, width, height, crop_x, crop_y, crop_w, crop_h);

    let crop_transformation = Transformation::Crop {
        orig_w: width,
        orig_h: height,
        x: crop_x,
        y: crop_y,
        w: crop_w,
        h: crop_h,
    };

    println!("Building curcuit");
    let start = Instant::now();
    let builder = TransformationCircuitBuilder::<L>::new(
        pixel_bytes.len(),
        Box::new(crop_transformation.clone()),
    );
    let mut circuit = builder.build_curcuit();
    let duration = start.elapsed();
    println!("Built curcuit in {:?}s", duration);

    circuit.prove(&pixel_bytes, &alidned_crop_bytes);

    Ok(())
}

fn verify(edited_image_path: String, metadata_path: String) -> Result<()> {
    let metadata: ProofMetadata = rmp_serde::from_slice(&fs::read(metadata_path)?)?;
    println!(
        "Original length: {}, edited length: {}",
        metadata.original_length, metadata.edited_length
    );
    let original_hash = metadata.proof.original_hash();
    let edited_hash = metadata.proof.edited_hash();

    let mut img = ImageReader::open(edited_image_path)?.decode()?.into_rgba8();

    match metadata.transformation {
        Transformation::Crop {
            orig_w,
            x,
            y,
            w,
            h,
            orig_h,
        } => {
            let pixel_bytes = align_crop_edit(img, orig_w, orig_h, x, y, w, h);
            let pixels_hash = calculate_poseidon(&pixel_bytes);
            assert_eq!(edited_hash, pixels_hash);
        }
    }

    println!("Building curcuit");
    let start = Instant::now();
    let builder = TransformationCircuitBuilder::<L>::new(
        metadata.original_length,
        Box::new(metadata.transformation),
    );
    let circuit = builder.build_curcuit();
    println!("Built curcuit in {:?}s", start.elapsed());

    let verify_start = Instant::now();
    match metadata.proof.verify(circuit.circuit) {
        Ok(_) => println!("Proof is valid!"),
        Err(_) => println!("Proof is invalid!"),
    }
    println!("Verified in {:?}s", verify_start.elapsed());
    Ok(())
}

fn main() -> Result<()> {
    // let options = parse_options()?;

    let mut builder = env_logger::Builder::from_default_env();
    builder.format_timestamp(None);
    builder.filter_level(LevelFilter::Info);
    builder.try_init()?;

    // Run the benchmark
    let num_cpus = num_cpus::get();
    let threads = num_cpus;
    println!("Number of CPUs: {}", num_cpus);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("Failed to build thread pool.")?
        .install(|| {
            info!(
                "Using {} compute threads on {} cores",
                rayon::current_num_threads(),
                num_cpus
            );
            // Run the benchmark. `options.lookup_type` determines which benchmark to run.

            let im_path = "./random_color.png".to_string();
            prove_crop(im_path, 55, 30, 512, 512)
        })?;
    Ok(())
}

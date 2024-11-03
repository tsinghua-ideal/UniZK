use structopt::StructOpt;

#[derive(Clone, StructOpt, Debug)]
#[structopt(
    name = "zkedit_options",
    about = "the prover and the verifier for image transformations"
)]
pub enum Zkedit {
    Prove {
        /// A path to original img file
        #[structopt(short = "i")]
        orig_img_path: String,

        /// X coordinate of crop upper left pixel
        #[structopt(short = "x", default_value = "55")]
        crop_x: u32,

        /// Y coordinate of crop upper left pixel
        #[structopt(short = "y", default_value = "30")]
        crop_y: u32,

        /// Width of the crop
        #[structopt(short = "w", default_value = "512")]
        crop_w: u32,

        /// Height of the crop
        #[structopt(short = "h", default_value = "512")]
        crop_h: u32,
    },

    Verify {
        /// A path to the original img file
        #[structopt(short = "e")]
        edited_image_path: String,

        /// A path to the metadata file
        #[structopt(short = "m")]
        metadata_path: String,
    },
}

pub fn parse_options() -> Result<Zkedit, structopt::clap::Error> {
    Zkedit::from_args_safe()
}

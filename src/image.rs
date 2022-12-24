use std::{
    fs::File,
    io::{BufWriter, Write},
};

pub struct Image {
    data: Vec<u32>,
    height: usize,
    width: usize,
}

pub static INFERNO_COLOUR_MAP: [[u8; 3]; 64] = [
    [0, 0, 4],
    [2, 1, 10],
    [4, 3, 19],
    [7, 5, 27],
    [11, 7, 36],
    [16, 9, 46],
    [21, 11, 56],
    [27, 12, 65],
    [34, 12, 75],
    [40, 11, 84],
    [48, 10, 91],
    [55, 9, 97],
    [62, 9, 102],
    [69, 10, 105],
    [75, 12, 107],
    [82, 14, 109],
    [88, 16, 110],
    [95, 19, 110],
    [101, 21, 110],
    [108, 24, 110],
    [114, 26, 110],
    [120, 28, 109],
    [127, 30, 108],
    [133, 33, 107],
    [140, 35, 105],
    [146, 37, 104],
    [153, 40, 101],
    [159, 42, 99],
    [165, 45, 96],
    [172, 47, 93],
    [178, 50, 90],
    [184, 53, 87],
    [190, 56, 83],
    [196, 60, 79],
    [201, 64, 75],
    [207, 68, 70],
    [212, 72, 66],
    [217, 77, 61],
    [221, 82, 57],
    [226, 87, 52],
    [230, 93, 47],
    [234, 99, 42],
    [237, 105, 37],
    [240, 111, 32],
    [243, 118, 27],
    [245, 125, 21],
    [247, 132, 16],
    [249, 139, 11],
    [250, 146, 7],
    [251, 154, 6],
    [252, 161, 8],
    [252, 169, 14],
    [252, 177, 21],
    [251, 185, 30],
    [250, 193, 39],
    [249, 201, 49],
    [247, 209, 60],
    [246, 217, 72],
    [244, 224, 85],
    [242, 232, 100],
    [241, 239, 116],
    [243, 245, 133],
    [246, 250, 150],
    [252, 255, 164],
];

pub fn inferno_colour_map(value: u8) -> u32 {
    // move the value into the range 0..64
    let value = (value as usize * 64) / 256;
    let v = INFERNO_COLOUR_MAP[value];
    u32::from(v[2]) | (u32::from(v[1]) << 8) | (u32::from(v[0]) << 16)
}

impl Image {
    pub fn zeroed(width: usize, height: usize) -> Self {
        Self { data: vec![0; width * height], height, width }
    }

    pub fn rows(&self) -> impl Iterator<Item = &[u32]> {
        self.data.chunks(self.width)
    }

    #[allow(dead_code)]
    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [u32]> {
        self.data.chunks_mut(self.width)
    }

    #[allow(dead_code)]
    pub fn get(&self, x: usize, y: usize) -> u32 {
        self.data[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, value: u32) {
        self.data[y * self.width + x] = value;
    }

    pub const fn width(&self) -> usize {
        self.width
    }

    pub const fn height(&self) -> usize {
        self.height
    }

    // Write the image to a TGA file with the given name.
    // Format specification: http://www.gamers.org/dEngine/quake3/TGA.txt
    pub fn save_as_tga<P>(&self, filename: P)
    where
        P: AsRef<std::path::Path>,
    {
        #![allow(clippy::cast_possible_truncation)]
        let file = File::create(&filename).unwrap();
        // use a buffered writer
        let mut writer = BufWriter::new(file);

        // Write the TGA header.
        #[rustfmt::skip]
        let header: [u8; 18] = [
            0, // no image ID
            0, // no colour map
            2, // uncompressed 24-bit image
            0, 0, 0, 0, 0, // empty colour map specification
            0, 0, // X origin
            0, 0, // Y origin
            self.width() as u8, (self.width() >> 8) as u8, // width
            self.height() as u8, (self.height() >> 8) as u8, // height
            24, // bits per pixel
            0, // image descriptor
        ];

        writer.write_all(&header).unwrap();

        for row in self.rows() {
            for &loc in row.iter() {
                let pixel: [u8; 3] =
                    [(loc & 0xFF) as u8, (loc >> 8 & 0xFF) as u8, (loc >> 16 & 0xFF) as u8];
                writer.write_all(&pixel).unwrap();
            }
        }

        writer.flush().unwrap();

        println!("Wrote {}", filename.as_ref().display());
    }
}

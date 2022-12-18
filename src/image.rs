use std::{
    fs::File,
    io::{BufWriter, Write},
};

pub struct Image {
    data: Vec<u32>,
    height: usize,
    width: usize,
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

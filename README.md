# Small S3 Texture Compression Library for Odin
(The original implementation is a port of stb_dxt.h)

Supports **BC1-5** and **DXT1-5**

> S3TC is a technique for compressing images for use as textures. Standard image compression techniques like JPEG and PNG can achieve greater compression ratios than S3TC. However, S3TC is designed to be implemented in high-performance hardware. JPEG and PNG decompress images all-at-once, while S3TC allows specific sections of the image to be decompressed independently.

> S3TC is a block-based format. The image is broken up into 4x4 blocks. For non-power-of-two images that aren't a multiple of 4 in size, the other colors of the 4x4 block are taken to be black. Each 4x4 block is independent of any other, so it can be decompressed independently.
> (from [opengl wiki](https://www.khronos.org/opengl/wiki/S3_Texture_Compression))

### Supported Formats
Name       | Description               | Premul Alpha  | Compression      | Texture type
-----------|---------------------------|---------------|------------------|---------------
BC1/DXT1   | 1-bit alpha / opaque      | Yes           | 6:1 (24bit src)  | Simple non-alpha
BC2/DXT2   | Explicit alpha            | Yes           | 4:1              | Sharp alpha
BC2/DXT3   | Explicit alpha            | No            | 4:1              | Sharp alpha
BC3/DXT4   | Interpolated alpha        | Yes           | 4:1              | Gradient alpha
BC3/DXT5   | Interpolated alpha        | No            | 4:1              | Gradient alpha
BC4        | Interpolated greyscale    | --            | 2:1              | Gradient
BC5        | Interpolated two-channel  | --            | 2:1              | Gradient

## Usage

```odin
import dxt "odxt"

main :: proc() {
    image = load_image("foo.png")
    
    for bx := 0; bx < image.width; bx += 4 {
        for by := 0; by < image.height; by += 4 {
            block: [16][4]u8
            for sx in 0..<4 {
                for sy in 0..<4 {
                    x = bx + sx
                    y = by + sy
                    if x >= image.width || y > image.width {
                        continue
                    }
                    
                    index = x + y * image.width
                    block[sx + sy * 4] = {
                        image.data[index + 0],
                        image.data[index + 1],
                        image.data[index + 2],
                        image.data[index + 3],
                    }
                }
            }
            
            compressed := dxt.compress_bc1_block(block)
            
            // ...
        }
    }
}
```

## Contributing
Bug fixes and other contributions are welcome, please submit a PR!

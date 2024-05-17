package odxt_test

import dxt ".."
import "core:fmt"
import stbi "vendor:stb/image"

foreign import "stb_dxt.lib"
foreign stb_dxt {
    stb_compress_dxt_block :: proc(dest: [^]u8, src_rgba_four_bytes_per_pixel: [^]u8, alpha: i32, mode: i32) ---
}

main :: proc() {
    paths := [?]string {
        "BricksReclaimedWhitewashedOffset001_Sphere.png",
        "BricksReclaimedWhitewashedOffset001_NRM_2K_METALNESS.png",
        "BricksReclaimedWhitewashedOffset001_DISP16_2K_METALNESS.png",
    }

    high_qual := false

    for path in paths {
        fmt.println(path)
        width, height, channels: i32
        data := stbi.load(fmt.ctprintf("{}", path), &width, &height, &channels, 4)

        if data == nil {
            fmt.println("Error: failed to load image")
            fmt.println(stbi.failure_reason())
            continue
        }

        for bx := 0; bx < int(width); bx += 4 {
            for by := 0; by < int(height); by += 4 {
                block: [16][4]u8
                for sx in 0 ..< 4 {
                    for sy in 0 ..< 4 {
                        x := bx + sx
                        y := by + sy
                        if x >= int(width) || y >= int(width) {
                            continue
                        }

                        offset := 4 * (x + y * int(width))
                        block[sx + 4 * sy] = {data[offset + 0], data[offset + 1], data[offset + 2], data[offset + 3]}
                    }
                }

                // fmt.println("Block", bx, by)

                res0, res1: [2][8]u8
                res0 = dxt.compress_bc3_block(src = block, high_quality = high_qual)

                stb_compress_dxt_block(cast([^]u8)&res1, cast([^]u8)&block, alpha = 1, mode = high_qual ? 2 : 0)

                // fmt.println(bx, by, res0, res1)

                if res0[0] != res1[0] {
                    fmt.println("  Incorrect 0", bx, by, res0[0], res1[0])
                    // assert(false)
                }

                if res0[1] != res1[1] {
                    fmt.println("  Incorrect 1", bx, by, res0[1], res1[1])
                    // assert(false)
                }
            }
        }
    }
}

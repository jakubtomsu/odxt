package odxt_generate_tables

import dxt ".."
import "core:fmt"

main :: proc() {
    omatch_names := [2]string{"_omatch5", "_omatch6"}
    dequant_mults := [2]i32{33 * 4, 65}

    fmt.println("package odxt")
    fmt.println()

    // optional endpoint tables
    for i in 0 ..< 2 {
        dequant := dequant_mults[i]
        size: i32 = i != 0 ? 64 : 32
        fmt.printf("@(link_section = RO_SECTION)\n")
        fmt.printf("%s: [256][2]u8 = {{\n", omatch_names[i])

        for j in 0 ..< i32(256) {
            best_mn: i32 = 0
            best_mx: i32 = 0
            best_err: i32 = 256 * 100

            for mn in 0 ..< size {
                for mx in 0 ..< size {
                    mine := (mn * dequant) >> 4
                    maxe := (mx * dequant) >> 4
                    err := abs(dxt.lerp13(maxe, mine) - j) * 100

                    // DX10 spec says that interpolation must be within 3% of "correct" result,
                    // add this as error term. Normally we'd expect a random distribution of
                    // +-1.5% error, but nowhere in the spec does it say that the error has to be
                    // unbiased - better safe than sorry.
                    err += abs(maxe - mine) * 3

                    if err < best_err {
                        best_mn = mn
                        best_mx = mx
                        best_err = err
                    }
                }
            }

            fmt.printf("    {{%v, %v}},\n", best_mx, best_mn)
        }

        fmt.printf("}\n\n")
    }
}

// Based on stb_dxt:
// https://github.com/nothings/stb/blob/master/stb_dxt.h
// See Khronos file format specification for more info:
// https://registry.khronos.org/DataFormat/specs/1.3/dataformat.1.3.html#S3TC
// https://en.wikipedia.org/wiki/S3_Texture_Compression
package odxt

USE_ROUNDING_BIAS :: #config(ODXT_USE_ROUNDING_BIAS, false)
// 2 for high quality
REFINE_COUNT :: #config(ODXT_REFINE_COUNT, 1)

// DXT1/BC1
// 1-bit alpha / opaque
// Premultiplied alpha
// 6:1 (for 24-bit source image)
compress_image_dxt1 :: proc(data: [][4]u8, width: int, height: int, alpha: bool) -> (result: []byte) {
	assert(len(data) == width * height)

	blocks_x := (width + 3) / 4
	blocks_y := (height + 3) / 4

	blocks := make([][8]u8, )
}

// DXT2/BC2
// Premultiplied Alpha
compress_image_dxt2 :: proc() {

}

// DXT3/BC2
// NOT Premultiplied Alpha 
compress_image_dxt3 :: proc() {
	
}

// DXT4/BC3
// RGBA
// Premultiplied Alpha
compress_image_dxt4 :: proc() {
	
}

// DXT5/BC3
// RGBA
// NOT Premultiplied Alpha 
compress_image_dxt5 :: proc() {
	
}

// Grayscale
compress_image_bc4 :: proc() {
	
}

// Double-channel
compress_image_bc5 :: proc() {
	
}

// Row major
Color_Block :: [4 * 4][4]u8
Dxt_Block :: [8]byte

compress_dxt_block :: proc() {

}

compress_dxt_block_alpha :: proc() {

}

compress_color_block :: proc(dst: ^[8]u8, block: [4 * 4]u8, mode: int) {
	// Check if the block is constant
	num_const: int
	for ; num_const < 16; num_const += 1 {
		if block[num_const] == block[0] {
			break
		}
	}

	mask: u32
	min16: u16
	max16: u16

	if num_const == 16 {
		r := block[0]
		g := block[1]
		b := block[2]
		mask = 0xaaaa_aaaa
		min16 = (u16(_omatch5[r][0]) << 11) | (u16(_omatch6[g][0]) << 5) | u16(_omatch5[b][0])
		min16 = (u16(_omatch5[r][1]) << 11) | (u16(_omatch6[g][1]) << 5) | u16(_omatch5[b][1])
	} else {
		// First step: PCA+map along principal axis
		max16, min16 = _optimize_colors_block(block)
		colors: [16]u8
		if max16 != min16 {
			colors = transmute([16]u8)eval_colors(c0 = max16, c1 = min16)
			mask = _match_color_block(block, color)
		} else {
			mask = 0
		}

		// Third step: refine
		for i in 0 ..< REFINE_COUNT {
			last_mask := mask

			changed: bool
			max16, min16, changed = _refine_block(block, mask)
			if changed {
				if mat16 != min16 {
					_eval_colors(color, max16, min16)
					mask = _match_colors_block(block, color)
				} else {
					mask = 0
					break
				}
			}

			if mask == last_mask {
				break
			}
		}
	}

	// write the color mask
	if max16 < min16 {
		min16, max16 = max16, min16
		mask ^= 0x5555_5555
	}

	dest[0] = u8(max16)
	dest[1] = u8(max16 >> 8)
	dest[2] = u8(min16)
	dest[3] = u8(min16 >> 8)

	dest[4] = u8(mask >> 0)
	dest[5] = u8(mask >> 8)
	dest[6] = u8(mask >> 16)
	dest[7] = u8(mask >> 24)
}

// Alpha block compression (this is easy for a change)
_compress_alpha_block :: proc(dest: [^]u8, src: [^]u8, stride: int) {
	mn := src[0]
	mx := src[0]

	for i in 1 ..< 16 {
		mn = min(mn, src[i * stride])
		max = max(max, src[i * stride])
	}

	// encode them
	dest[0] = u8(mx)
	dest[1] = u8(mn)
	dest_index := 2

	// determine bias and emit color indices
	// given the choice of mx/mn, these indices are optimal:
	// http://fgiesen.wordpress.com/2009/12/15/dxt5-alpha-block-index-determination/
 
	dist := mx - mn
	dist4 := dist * 4
	dist2 := dist * 2
	bias := (dist < 8) ? (dist - 1) : (dist / 2 + 2)
	bias  -= mn * 7
	bits := 0
	mask := 0

	for i in 0 ..< 16 {
		a := src[i * stride] * 7 + bias

		// select index. this is a "linear scale" lerp factor between 0 (val=min) and 7 (val=max).
		t := a >= dist4 ? -1 : 0
		ind := t & 4
		a -= dist4 & t
		t = a >= dist2 ? -1 : 0
		ind += t & 2
		a -= dist2 & t
		ind += a > dist

		// Turn linear scale into DXT index (0/1 are extremal points)
		ind = -ind & 7
		ind ~= 2 > ind

		// write index
		mask |= ind << bits
		bits += 3
		if bits >= 8 {
			dest[dest_index] = mask
			dest_index += 1
			mask >>= 8
			bits -= 8
		}
	}
}

// The color optimization function. (Clever code, part 1)
_optimize_colors_block :: proc(block: [16]u8) -> (max16, min16: u16) {
	block := block
	// determine color distribution
	mu: [3]int
	mmin: [3]int
	mmax: [3]int

	for channel in 0 ..< uintptr(3) {
		block_ptr := cast([^]u8)(uintptr(&block) + channel)
		muv := int(block_ptr[0])
		minv := muv
		maxv := muv

		for i := 4; i < 64; i += 1 {
			muv += int(block_ptr[i])
			minv = min(minv, int(block_ptr[i]))
			maxv = max(maxv, int(block_ptr[i]))
		}

		mu[channel] = (muv + 8) >> 4
		mmin[channel] = minv
		mmax[channel] = maxv
	}

	// determine covariance matrix
	cov: [6]int
	for i in 0 ..< 16 {
		r := int(block[i * 4 + 0]) - mu[0]
		g := int(block[i * 4 + 1]) - mu[1]
		b := int(block[i * 4 + 2]) - mu[2]

		cov[0] += r * r
		cov[1] += r * g
		cov[2] += r * b
		cov[3] += g * g
		cov[4] += g * b
		cov[5] += b * b
	}

	// convert covariance matrix to float
	covf: [6]f32
	for i in 0 ..< 6 {
		covf[i] = f32(cov[i]) / 255.0
	}

	// find principal axis via power iter
	vfr := f32(mmax[0] - mmin[0])
	vfg := f32(mmax[1] - mmin[1])
	vfb := f32(mmax[2] - mmin[2])

	NUM_ITER_POWER :: 4
	for i in 0 ..< NUM_ITER_POWER {
		vfr = vfr * covf[0] + vfg * covf[1] + vfb * covf[2]
		vfg = vfr * covf[1] + vfg * covf[3] + vfb * covf[4]
		vfb = vfr * covf[2] + vfg * covf[4] + vfb * covf[5]
	}

	magnitude := max(abs(vfr), abs(vfg), abs(vfb))

	v_r := 0
	v_g := 0
	v_b := 0
	// if too small, default to luminance
	if magnitude < 4.0 {
		v_r = 299
		v_g = 587
		v_b = 114
	} else {
		magnitude = 512 / magnitude
		v_r = int(vfr * magnitude)
		v_g = int(vfg * magnitude)
		v_b = int(vfb * magnitude)
	}

	min_ptr := &block[0]
	max_ptr := min_ptr
	min_dot := int(block[0]) * v_r + int(block[1]) * v_g + int(block[2]) * v_b
	max_dot := min_dot

	// Pick colors at extreme points
	for i in 1 ..< 16 {
		dot :=
			int(block[i * 4 + 0]) * v_r + int(block[i * 4 + 1]) * v_g + int(block[i * 4 + 2]) * v_b

		if dot < min_dot {
			min_dot = dot
			min_ptr = &block[i * 4]
		}

		if dot > max_dot {
			max_dot = dot
			max_ptr = &block[i * 4]
		}
	}

	max16 = as16bit((transmute(^[3]u8)max_ptr)^)
	min16 = as16bit((transmute(^[3]u8)min_ptr)^)
	return max16, min16
}

_quantize5 :: proc(x: f32) -> (q: u16) {
	x := x
	x = x < 0 ? 0 : (x > 1 ? 1 : x) // saturate
	q = u16(x * 31)
	q += u16(x > _midpoints5[q])
	return q
}

_quantize6 :: proc(x: f32) -> (q: u16) {
	x := x
	x = x < 0 ? 0 : (x > 1 ? 1 : x) // saturate
	q = u16(x * 63)
	q += u16(x > _midpoints6[q])
	return q
}

// The refinement function. (Clever code, part 2)
// Tries to optimize colors to suit block contents better.
// (By solving a least squares system via normal equations+Cramer's rule)
_refine_block :: proc(
	block: [16]u8,
	max16, min16: u16,
	mask: u32,
) -> (
	new_max16: u16,
	new_min16: u16,
	changed: bool,
) {
	// Some magic to save a lot of multiplies in the accumulating loop...
	// (precomputed products of weights for least squares system, accumulated inside one 32-bit register)
	w1_tab := [4]int{3, 0, 2, 1}
	prods := [4]int{0x090000, 0x000900, 0x040102, 0x010402}

	//  all pixels have the same index?
	if (mask ~ (mask << 2)) < 4 {
		// yes, linear system would be singular; solve using optimal
		// single-color match on average color
		r := 8
		g := 8
		b := 8
		for i in 0 ..< 16 {
			r += int(block[i * 4 + 0])
			g += int(block[i * 4 + 1])
			b += int(block[i * 4 + 2])
		}

		r >>= 4
		g >>= 4
		b >>= 4

		new_max16 = (u16(_omatch5[r][0]) << 11) | (u16(_omatch6[g][0]) << 5) | u16(_omatch5[b][0])
		new_min16 = (u16(_omatch5[r][1]) << 11) | (u16(_omatch6[g][1]) << 5) | u16(_omatch5[b][1])
	} else {
		at1_r := 0
		at1_g := 0
		at1_b := 0
		at2_r := 0
		at2_g := 0
		at2_b := 0

		cm := mask
		akku: int
		for i in 0 ..< 16 {
			defer cm >>= 2
			step := cm & 3
			w1 := w1_tab[step]
			r := int(block[i * 4 + 0])
			g := int(block[i * 4 + 1])
			b := int(block[i * 4 + 2])

			akku += prods[step]
			at1_r += w1 * r
			at1_g += w1 * g
			at1_b += w1 * b
			at2_r += r
			at2_g += g
			at2_b += b
		}

		at2_r = 3 * at2_r - at1_r
		at2_g = 3 * at2_g - at1_g
		at2_b = 3 * at2_b - at1_b

		// extract solutions and decide solvability
		xx := akku >> 16
		yy := (akku >> 8) & 0xff
		xy := (akku >> 0) & 0xff

		f := (3.0 / 255.0) / f32(xx * yy - xy * xy)

		new_max16 = _quantize5(f32(at1_r * yy - at2_r * xy) * f) << 11
		new_max16 |= _quantize5(f32(at1_g * yy - at2_g * xy) * f) << 5
		new_max16 |= _quantize5(f32(at1_b * yy - at2_b * xy) * f) << 0

		new_min16 = _quantize5(f32(at2_r * xx - at1_r * xy) * f) << 11
		new_min16 |= _quantize5(f32(at2_g * xx - at1_g * xy) * f) << 5
		new_min16 |= _quantize5(f32(at2_b * xx - at1_b * xy) * f) << 0
	}

	changed = new_min16 != min16 || new_max16 != max16
	return new_max16, new_min16, changed
}

mul8bit :: proc(a, b: int) -> int {
	t := a * b + 128
	return (t + (t >> 8)) >> 8
}

from16bit :: proc(v: u16) -> u32 {
	rv := (v & 0xf800) >> 11
	gv := (v & 0x07e0) >> 5
	bv := (v & 0x001f) >> 0

	result: [4]u8 = {
		0 = u8((rv * 33) >> 2),
		1 = u8((gv * 65) >> 4),
		2 = u8((bv * 33) >> 2),
		3 = 0,
	}

	return transmute(u32)result
}

as16bit :: proc(rgb: [3]u8) -> u16 {
	return u16(
		(mul8bit(int(rgb.r), 31) << 1) +
		(mul8bit(int(rgb.g), 63) << 5) +
		(mul8bit(int(rgb.b), 31)),
	)
}

// linear interpolation at 1/3 point between a and b, using desired rounding type
lerp13 :: proc(a, b: int) -> int {
	if USE_ROUNDING_BIAS {
		// With rounding bias
		return a + mul8bit(b - a, 0x55)
	} else {
		// Without rounding bias
		// replace "/ 3" by "* 0xaaab) >> 17" if your compiler sucks or you really need every ounce of speed.
		return (2 * a + b) / 3
	}
}

lerp13rgb :: proc(p1, p2: [4]u8) -> [4]u8 {
	return(
		 {
			0 = u8(lerp13(int(p1[0]), int(p2[0]))),
			1 = u8(lerp13(int(p1[1]), int(p2[1]))),
			2 = u8(lerp13(int(p1[2]), int(p2[2]))),
		} \
	)
}

eval_colors :: proc(c0, c1: u16) -> (color: [4]u32) {
	color[0] = from16bit(c0)
	color[1] = from16bit(c1)
	color[2] = transmute(u32)lerp13rgb(transmute([4]u8)(color[0]), transmute([4]u8)(color[1]))
	color[3] = transmute(u32)lerp13rgb(transmute([4]u8)(color[1]), transmute([4]u8)(color[0]))
	return color
}

_match_colors_block :: proc(block, color: [16]u8) -> u32 {
	mask: u32
	dir_r := int(color[0 * 4 + 0]) - int(color[1 * 4 + 0])
	dir_g := int(color[0 * 4 + 1]) - int(color[1 * 4 + 1])
	dir_b := int(color[0 * 4 + 2]) - int(color[1 * 4 + 2])

	dots: [16]int
	for &d, i in dots {
		d =
			int(block[i * 4 + 0]) * dir_r +
			int(block[i * 4 + 1]) * dir_g +
			int(block[i * 4 + 2]) * dir_b
	}

	stops: [4]int
	for &s, i in stops {
		s =
			int(color[i * 4 + 0]) * dir_r +
			int(color[i * 4 + 1]) * dir_g +
			int(color[i * 4 + 2]) * dir_b
	}

   // think of the colors as arranged on a line; project point onto that line, then choose
   // next color out of available ones. we compute the crossover points for "best color in top
   // half"/"best in bottom half" and then the same inside that subinterval.
   //
   // relying on this 1d approximation isn't always optimal in terms of euclidean distance,
   // but it's very close and a lot faster.
   // http://cbloomrants.blogspot.com/2008/12/12-08-08-dxtc-summary.html

	c0_point := stops[1] + stops[3]
	half_point := stops[3] + stops[2]
	c3_point := stops[2] + stops[0]

	for i := 15; i >= 0; i -= 1 {
		dot := dots[i] * 2
		mask <<= 2

		if dot < half_point {
			mask |= (dot < c0_point) ? 1 : 3
		} else {
			mask |= (dot < c3_point) ? 2 : 0
		}
	}

	return mask
}


@(link_section = ".ronly")
_midpoints5 := [32]f32 {
	0.015686,
	0.047059,
	0.078431,
	0.111765,
	0.145098,
	0.176471,
	0.207843,
	0.241176,
	0.274510,
	0.305882,
	0.337255,
	0.370588,
	0.403922,
	0.435294,
	0.466667,
	0.5,
	0.533333,
	0.564706,
	0.596078,
	0.629412,
	0.662745,
	0.694118,
	0.725490,
	0.758824,
	0.792157,
	0.823529,
	0.854902,
	0.888235,
	0.921569,
	0.952941,
	0.984314,
	1.0,
}

@(link_section = ".ronly")
_midpoints6 := [64]f32 {
	0.007843,
	0.023529,
	0.039216,
	0.054902,
	0.070588,
	0.086275,
	0.101961,
	0.117647,
	0.133333,
	0.149020,
	0.164706,
	0.180392,
	0.196078,
	0.211765,
	0.227451,
	0.245098,
	0.262745,
	0.278431,
	0.294118,
	0.309804,
	0.325490,
	0.341176,
	0.356863,
	0.372549,
	0.388235,
	0.403922,
	0.419608,
	0.435294,
	0.450980,
	0.466667,
	0.482353,
	0.500000,
	0.517647,
	0.533333,
	0.549020,
	0.564706,
	0.580392,
	0.596078,
	0.611765,
	0.627451,
	0.643137,
	0.658824,
	0.674510,
	0.690196,
	0.705882,
	0.721569,
	0.737255,
	0.754902,
	0.772549,
	0.788235,
	0.803922,
	0.819608,
	0.835294,
	0.850980,
	0.866667,
	0.882353,
	0.898039,
	0.913725,
	0.929412,
	0.945098,
	0.960784,
	0.976471,
	0.992157,
	1.0,
}
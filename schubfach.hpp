//--------------------------------------------------------------------------------
// This file contains an implementation of the Schubfach algorithm as described in
//
// [1] Raffaello Giulietti, "The Schubfach way to render doubles",
//     https://drive.google.com/open?id=1luHhyQF9zKlM8yJ1nebU0OgVYhfC6CBN
//--------------------------------------------------------------------------------

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#if _MSC_VER
#include <intrin.h>
#endif

#ifndef SF_ASSERT
#define SF_ASSERT(X) assert(X)
#endif

namespace schubfach {

template <typename Dest, typename Source> static inline Dest reinterpret_bits(Source source) {
  static_assert(sizeof(Dest) == sizeof(Source), "size mismatch");

  Dest dest;
  std::memcpy(&dest, &source, sizeof(Source));
  return dest;
}

template <typename Float> struct float_traits {
  static constexpr uint16_t significand_width = std::numeric_limits<Float>::digits;
  static constexpr uint16_t exponent_width = std::bit_width((unsigned int)std::numeric_limits<Float>::max_exponent);
  static constexpr uint16_t sign_width = 1;
  static constexpr bool has_hidden_bit = ((sign_width + exponent_width + significand_width) % 8) != 0;
  static constexpr uint16_t storage_width = sign_width + exponent_width + significand_width + ((has_hidden_bit) ? -1 : 0);
  static constexpr int32_t exponent_bias = std::numeric_limits<Float>::max_exponent + significand_width - 2;
  using uint_t =
      std::conditional_t<storage_width <= 8, uint8_t,
                         std::conditional_t<storage_width <= 16, uint16_t,
                                            std::conditional_t<storage_width <= 32, uint32_t,
                                                               std::conditional_t<storage_width <= 64, uint64_t, __uint128_t>>>>;
  static constexpr uint16_t exponent_shift = storage_width - sign_width - exponent_width;
  static constexpr uint16_t sign_shift = storage_width - sign_width;
  static constexpr uint_t significand_mask = (uint_t{1} << (significand_width + ((has_hidden_bit) ? -1 : 0))) - uint_t{1};
  static constexpr uint_t exponent_mask = ((uint_t{1} << exponent_width) - uint_t{1}) << exponent_shift;
  static constexpr uint_t sign_mask = ((uint_t{1} << sign_width) - uint_t{1}) << sign_shift;
};

template <typename uint_t> struct math {
  using limits = std::numeric_limits<uint_t>;

  using uint_2_t = std::conditional_t<limits::digits <= 32, uint64_t, __uint128_t>;
  using uint_4_t = __uint128_t;

  static inline int32_t floor_log2_pow10(int32_t e) { return (e * 1741647) >> 19; }

  static inline uint_2_t compute_pow10(int32_t k);

  static inline uint_t word0(uint_4_t x) { return static_cast<uint_t>(x >> limits::digits); }

  static inline uint_t word1(uint_4_t x) { return static_cast<uint_t>(x >> (2 * limits::digits)); }

  static inline uint_t round_to_odd(uint_2_t g, uint_t cp) {
    const uint_4_t p = uint_4_t{g} * cp;

    const uint_t y1 = word1(p);
    const uint_t y0 = word0(p);

    return y1 | (y0 > 1);
  }
};

template <> uint64_t math<uint32_t>::compute_pow10(int32_t k) {
  static constexpr int32_t kMin = -31;
  static constexpr int32_t kMax = 45;
  static constexpr uint64_t g[kMax - kMin + 1] = {
      0x81CEB32C4B43FCF5, // -31
      0xA2425FF75E14FC32, // -30
      0xCAD2F7F5359A3B3F, // -29
      0xFD87B5F28300CA0E, // -28
      0x9E74D1B791E07E49, // -27
      0xC612062576589DDB, // -26
      0xF79687AED3EEC552, // -25
      0x9ABE14CD44753B53, // -24
      0xC16D9A0095928A28, // -23
      0xF1C90080BAF72CB2, // -22
      0x971DA05074DA7BEF, // -21
      0xBCE5086492111AEB, // -20
      0xEC1E4A7DB69561A6, // -19
      0x9392EE8E921D5D08, // -18
      0xB877AA3236A4B44A, // -17
      0xE69594BEC44DE15C, // -16
      0x901D7CF73AB0ACDA, // -15
      0xB424DC35095CD810, // -14
      0xE12E13424BB40E14, // -13
      0x8CBCCC096F5088CC, // -12
      0xAFEBFF0BCB24AAFF, // -11
      0xDBE6FECEBDEDD5BF, // -10
      0x89705F4136B4A598, //  -9
      0xABCC77118461CEFD, //  -8
      0xD6BF94D5E57A42BD, //  -7
      0x8637BD05AF6C69B6, //  -6
      0xA7C5AC471B478424, //  -5
      0xD1B71758E219652C, //  -4
      0x83126E978D4FDF3C, //  -3
      0xA3D70A3D70A3D70B, //  -2
      0xCCCCCCCCCCCCCCCD, //  -1
      0x8000000000000000, //   0
      0xA000000000000000, //   1
      0xC800000000000000, //   2
      0xFA00000000000000, //   3
      0x9C40000000000000, //   4
      0xC350000000000000, //   5
      0xF424000000000000, //   6
      0x9896800000000000, //   7
      0xBEBC200000000000, //   8
      0xEE6B280000000000, //   9
      0x9502F90000000000, //  10
      0xBA43B74000000000, //  11
      0xE8D4A51000000000, //  12
      0x9184E72A00000000, //  13
      0xB5E620F480000000, //  14
      0xE35FA931A0000000, //  15
      0x8E1BC9BF04000000, //  16
      0xB1A2BC2EC5000000, //  17
      0xDE0B6B3A76400000, //  18
      0x8AC7230489E80000, //  19
      0xAD78EBC5AC620000, //  20
      0xD8D726B7177A8000, //  21
      0x878678326EAC9000, //  22
      0xA968163F0A57B400, //  23
      0xD3C21BCECCEDA100, //  24
      0x84595161401484A0, //  25
      0xA56FA5B99019A5C8, //  26
      0xCECB8F27F4200F3A, //  27
      0x813F3978F8940985, //  28
      0xA18F07D736B90BE6, //  29
      0xC9F2C9CD04674EDF, //  30
      0xFC6F7C4045812297, //  31
      0x9DC5ADA82B70B59E, //  32
      0xC5371912364CE306, //  33
      0xF684DF56C3E01BC7, //  34
      0x9A130B963A6C115D, //  35
      0xC097CE7BC90715B4, //  36
      0xF0BDC21ABB48DB21, //  37
      0x96769950B50D88F5, //  38
      0xBC143FA4E250EB32, //  39
      0xEB194F8E1AE525FE, //  40
      0x92EFD1B8D0CF37BF, //  41
      0xB7ABC627050305AE, //  42
      0xE596B7B0C643C71A, //  43
      0x8F7E32CE7BEA5C70, //  44
      0xB35DBF821AE4F38C, //  45
  };

  SF_ASSERT(k >= kMin);
  SF_ASSERT(k <= kMax);
  return g[static_cast<uint32_t>(k - kMin)];
}

template <typename Float> struct float_triple {
  using float_traits = float_traits<Float>;
  using math = math<typename float_traits::uint_t>;
  using uint_t = float_traits::uint_t;
  using uint_2_t = math::uint_2_t;

  float_traits::uint_t significand;
  int32_t exponent;
  int8_t sign;

  float_triple(Float value) {
    significand = reinterpret_bits<typename float_traits::uint_t>(value);
    exponent = (significand & float_traits::exponent_mask) >> float_traits::exponent_shift;
    sign = (significand & float_traits::sign_mask) == 0 ? 1 : -1;
    significand = significand & float_traits::significand_mask;

    if (exponent == 0 && significand != 0) {
      exponent = float_traits::significand_width - std::bit_width(significand);
      significand = significand << exponent;
      exponent = 1 - float_traits::exponent_bias - exponent;
    } else if (exponent != 0) {
      if (float_traits::has_hidden_bit)
        significand = significand | (typename float_traits::uint_t{1} << (float_traits::significand_width - 1));
      exponent -= float_traits::exponent_bias;
    }
  }

  float_triple& to_decimal() {
    const bool is_even = (significand % 2 == 0);
    const bool accept_lower = is_even;
    const bool accept_upper = is_even;

    const bool lower_boundary_is_closer = std::popcount(significand) == 1;

    const uint_t cbl = 4 * significand - 2 + lower_boundary_is_closer;
    const uint_t cb = 4 * significand;
    const uint_t cbr = 4 * significand + 2;

    const int32_t k = (exponent * 1262611 - (lower_boundary_is_closer ? 524031 : 0)) >> 22;
    const int32_t h = exponent + math::floor_log2_pow10(-k) + 1;
    exponent = k;

    const uint_2_t pow10 = math::compute_pow10(-exponent);
    const uint_t vbl = math::round_to_odd(pow10, cbl << h);
    const uint_t vb = math::round_to_odd(pow10, cb << h);
    const uint_t vbr = math::round_to_odd(pow10, cbr << h);

    const uint_t lower = vbl + !accept_lower;
    const uint_t upper = vbr - !accept_upper;

    significand = vb / 4;

    if (significand >= 10) {
      const uint_t sp = significand / 10;
      const bool up_inside = lower <= 40 * sp;
      const bool wp_inside = 40 * sp + 40 <= upper;
      if (up_inside != wp_inside) {
        significand = sp + wp_inside;
        exponent++;
        return *this;
      }
    }

    const bool u_inside = lower <= 4 * significand;
    const bool w_inside = 4 * significand + 4 <= upper;
    if (u_inside != w_inside) {
      significand += w_inside;
      return *this;
    }

    const uint_t mid = 4 * significand + 2;
    const bool round_up = vb > mid || (vb == mid && (significand & 1) != 0);

    significand += round_up;
    return *this;
  }
};
} // namespace schubfach

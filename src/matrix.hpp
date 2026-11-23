#ifndef AVX2_MATH
#define AVX2_MATH

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>

#if !defined(OSMIMATH_DISABLE_AVX2) && (defined(__i386__) || defined(__x86_64__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define OSMIMATH_DETAIL_AVX2 1
#define OSMIMATH_DETAIL_TARGET __attribute__((target("avx2")))
#elif !defined(OSMIMATH_DISABLE_AVX2) && defined(__AVX2__)
#include <immintrin.h>
#define OSMIMATH_DETAIL_AVX2 1
#define OSMIMATH_DETAIL_TARGET
#else
#define OSMIMATH_DETAIL_AVX2 0
#define OSMIMATH_DETAIL_TARGET
#endif

class AVX2Math {
public:
  template<std::size_t Rows, std::size_t Cols>
  struct matrix {
    static_assert(
      Rows == 0 || Cols <= std::numeric_limits<std::size_t>::max() / sizeof(float) / Rows,
      "dimensions are too large");

    static constexpr std::size_t rows = Rows;
    static constexpr std::size_t cols = Cols;

    std::array<float, Rows * Cols> elements{};

    [[nodiscard]] static constexpr std::size_t size() noexcept {
      return Rows * Cols;
    }

    [[nodiscard]] constexpr float* data() noexcept {
      return elements.data();
    }

    [[nodiscard]] constexpr const float* data() const noexcept {
      return elements.data();
    }

    constexpr float& operator()(std::size_t row, std::size_t col) noexcept {
      return elements[row * Cols + col];
    }

    constexpr const float& operator()(std::size_t row, std::size_t col) const noexcept {
      return elements[row * Cols + col];
    }

    constexpr float& operator[](std::size_t row, std::size_t col) noexcept {
      return (*this)(row, col);
    }

    constexpr const float& operator[](std::size_t row, std::size_t col) const noexcept {
      return (*this)(row, col);
    }

    [[nodiscard]] static constexpr matrix identity() noexcept
      requires(Rows == Cols)
    {
      matrix result{};
      for (std::size_t i = 0; i < Rows; ++i) {
        result(i, i) = 1.0f;
      }

      return result;
    }

    constexpr bool operator==(const matrix&) const = default;
  };

  template<std::size_t Rows, std::size_t Cols>
  [[nodiscard]] matrix<Rows, Cols> make_matrix(float (&arr)[Rows*Cols], std::size_t len) {
    if (len == Rows * Cols) { throw std::runtime_error(""); }
    matrix<Rows, Cols> result; std::copy(arr, arr + Rows * Cols, result.elements.begin());
    return result;
  }

  template<std::size_t Rows, std::size_t Cols>
  [[nodiscard]] matrix<Rows, Cols> make_matrix(const float (&arr)[Rows][Cols], std::size_t len) {
    if (len == Rows * Cols) { throw std::runtime_error(""); }
    matrix<Rows, Cols> result;
    for (std::size_t i = 0; i < Rows; ++i){
      for (std::size_t j = 0; j < Cols; ++j){
        result.data()[i*Cols + j] = arr[i][j];
      }
    }
    return result;
  }

  template<std::size_t Rows, std::size_t Cols>
  [[nodiscard]] static constexpr matrix<Cols, Rows> AVX2Transpose(
    const matrix<Rows, Cols>& source) {
    matrix<Cols, Rows> result{};
    AVX2Transpose(source.data(), result.data(), Rows, Cols);
    return result;
  }

  template<std::size_t Rows, std::size_t Inner, std::size_t Cols>
  [[nodiscard]] static constexpr matrix<Rows, Cols> AVX2Dot(
    const matrix<Rows, Inner>& a, const matrix<Inner, Cols>& b) {
    matrix<Rows, Cols> result{};
#if OSMIMATH_DETAIL_AVX2
    if !consteval {
      if constexpr (Rows != 0 && Inner != 0 && Cols >= 8) {
        if (UseAVX2()) {
          DotRowsAVX2<Rows, Inner, Cols>(result.data(), a.data(), b.data());
          return result;
        }
      }
    }
#endif
    for (std::size_t row = 0; row < Rows; ++row) {
      for (std::size_t k = 0; k < Inner; ++k) {
        for (std::size_t col = 0; col < Cols; ++col) {
          result(row, col) += a(row, k) * b(k, col);
        }
      }
    }

    return result;
  }

  template<std::size_t Size>
  [[nodiscard]] static constexpr std::optional<matrix<Size, Size>> AVX2Inverse(
    const matrix<Size, Size>& source) {
    auto working = source;
#if OSMIMATH_DETAIL_AVX2
    if !consteval {
      if constexpr (Size >= 8) {
        if (UseAVX2()) {
          matrix<Size, Size> result;
          std::array<std::size_t, Size> permutation;
          if (InverseAVX2<Size>(working.data(), result.data(), permutation.data(), Size)) {
            return result;
          }
          return std::nullopt;
        }
      }
    }
#endif
    auto result = matrix<Size, Size>::identity();
    if (!InverseRows(working.data(), result.data(), Size)) {
      return std::nullopt;
    }

    return result;
  }

  template<std::size_t Size>
  [[nodiscard]] static constexpr float AVX2Det(const matrix<Size, Size>& source) {
    auto working = source;
#if OSMIMATH_DETAIL_AVX2
    if !consteval {
      if constexpr (Size >= 8) {
        if (UseAVX2()) {
          return DeterminantAVX2<Size>(working.data(), Size);
        }
      }
    }
#endif
    return DeterminantRows(working.data(), Size);
  }

  static constexpr void AVX2Transpose(
    const float* source, float* destination, std::size_t M, std::size_t N) {
    if (CheckedCount(M, N) == 0) {
      return;
    }

#if OSMIMATH_DETAIL_AVX2
    if !consteval {
      if (M >= 8 && N >= 8 && UseAVX2()) {
        TransposeAVX2(source, destination, M, N);
        return;
      }
    }
#endif
    constexpr std::size_t block_size = 4;
    for (std::size_t i = 0; i < M; i += block_size) {
      for (std::size_t j = 0; j < N; j += block_size) {
        const std::size_t limit_i = i + std::min(block_size, M - i);
        const std::size_t limit_j = j + std::min(block_size, N - j);
        for (std::size_t row = i; row < limit_i; ++row) {
          for (std::size_t col = j; col < limit_j; ++col) {
            destination[col * M + row] = source[row * N + col];
          }
        }
      }
    }
  }

  [[nodiscard]] static constexpr const float* AVX2Dot(
    std::size_t M, std::size_t N, std::size_t K, const float* a, const float* b) {
    const std::size_t count = CheckedCount(M, N);
    CheckedCount(M, K);
    CheckedCount(K, N);
    auto result = std::make_unique<float[]>(count);
    if (count == 0 || K == 0) {
      return result.release();
    }

#if OSMIMATH_DETAIL_AVX2
    if !consteval {
      if (N >= 8 && UseAVX2()) {
        DotRowsAVX2(result.get(), a, b, M, N, K);
        return result.release();
      }
    }
#endif
    for (std::size_t row = 0; row < M; ++row) {
      for (std::size_t k = 0; k < K; ++k) {
        for (std::size_t col = 0; col < N; ++col) {
          result[row * N + col] += a[row * K + k] * b[k * N + col];
        }
      }
    }

    return result.release();
  }

  static constexpr void AVX2Inverse(const float* source, float* destination, std::size_t sz) {
    const std::size_t count = CheckedCount(sz, sz);
    if (count == 0) {
      return;
    }

    auto working = std::make_unique_for_overwrite<float[]>(count);
    std::copy_n(source, count, working.get());
#if OSMIMATH_DETAIL_AVX2
    if !consteval {
      if (sz >= 8 && UseAVX2()) {
        auto permutation = std::make_unique_for_overwrite<std::size_t[]>(sz);
        if (!InverseAVX2(working.get(), destination, permutation.get(), sz)) {
          throw std::runtime_error("cannot invert a singular matrix");
        }
        return;
      }
    }
#endif
    auto result = std::make_unique<float[]>(count);
    for (std::size_t i = 0; i < sz; ++i) {
      result[i * sz + i] = 1.0f;
    }

    if (!InverseRows(working.get(), result.get(), sz)) {
      throw std::runtime_error("cannot invert a singular matrix");
    }

    std::copy_n(result.get(), count, destination);
  }

  static constexpr void getCoefficient(
    const float* source, float* coefficients, std::size_t p, std::size_t q, std::size_t n) {
    if (p >= n || q >= n) {
      throw std::out_of_range("minor index is out of range");
    }

    std::size_t index = 0;
    for (std::size_t row = 0; row < n; ++row) {
      for (std::size_t col = 0; col < n; ++col) {
        if (row != p && col != q) {
          coefficients[index++] = source[row * n + col];
        }
      }
    }
  }

  [[nodiscard]] static constexpr float AVX2Det(const float* source, std::size_t sz) {
    const std::size_t count = CheckedCount(sz, sz);
    if (count == 0) {
      return 1.0f;
    }

    auto working = std::make_unique_for_overwrite<float[]>(count);
    std::copy_n(source, count, working.get());
#if OSMIMATH_DETAIL_AVX2
    if !consteval {
      if (sz >= 8 && UseAVX2()) {
        return DeterminantAVX2<0>(working.get(), sz);
      }
    }
#endif
    return DeterminantRows(working.get(), sz);
  }

private:
  static constexpr std::size_t CheckedCount(std::size_t rows, std::size_t cols) {
    const std::size_t max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (rows != 0 && cols > max_elements / rows) {
      throw std::length_error("matrix dimensions are too large");
    }

    return rows * cols;
  }

  static constexpr float Abs(float value) noexcept {
    return std::abs(value);
  }

  static constexpr std::size_t FindPivot(const float* values, std::size_t sz, std::size_t col) {
    std::size_t pivot = col;
    float largest = Abs(values[col * sz + col]);
    for (std::size_t row = col + 1; row < sz; ++row) {
      const float magnitude = Abs(values[row * sz + col]);
      if (magnitude > largest) {
        largest = magnitude;
        pivot = row;
      }
    }

    return pivot;
  }

  static constexpr void SwapRows(float* a, float* b, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
      const float value = a[i];
      a[i] = b[i];
      b[i] = value;
    }
  }

  static constexpr void AddScaledRow(
    float* destination, const float* source, std::size_t count, float factor) {
    for (std::size_t i = 0; i < count; ++i) {
      destination[i] += factor * source[i];
    }
  }

  static constexpr void DivideRow(float* values, std::size_t count, float divisor) {
    for (std::size_t i = 0; i < count; ++i) {
      values[i] /= divisor;
    }
  }

  static constexpr bool InverseRows(float* working, float* result, std::size_t sz) {
    for (std::size_t col = 0; col < sz; ++col) {
      const std::size_t pivot = FindPivot(working, sz, col);
      if (working[pivot * sz + col] == 0.0f) {
        return false;
      }

      float* pivot_row = working + col * sz;
      float* inverse_row = result + col * sz;
      if (pivot != col) {
        SwapRows(pivot_row, working + pivot * sz, sz);
        SwapRows(inverse_row, result + pivot * sz, sz);
      }

      const float divisor = pivot_row[col];
      DivideRow(pivot_row, sz, divisor);
      DivideRow(inverse_row, sz, divisor);
      pivot_row[col] = 1.0f;

      for (std::size_t row = 0; row < sz; ++row) {
        if (row == col) {
          continue;
        }

        float* values = working + row * sz;
        const float factor = values[col];
        if (factor == 0.0f) {
          continue;
        }

        AddScaledRow(values, pivot_row, sz, -factor);
        AddScaledRow(result + row * sz, inverse_row, sz, -factor);
        values[col] = 0.0f;
      }
    }

    return true;
  }

  static constexpr float DeterminantRows(float* working, std::size_t sz) {
    float determinant = 1.0f;
    for (std::size_t col = 0; col < sz; ++col) {
      const std::size_t pivot = FindPivot(working, sz, col);
      if (working[pivot * sz + col] == 0.0f) {
        return 0.0f;
      }

      float* pivot_row = working + col * sz;
      if (pivot != col) {
        SwapRows(pivot_row, working + pivot * sz, sz);
        determinant = -determinant;
      }

      const float divisor = pivot_row[col];
      determinant *= divisor;
      for (std::size_t row = col + 1; row < sz; ++row) {
        float* values = working + row * sz;
        const float factor = values[col] / divisor;
        AddScaledRow(values + col + 1, pivot_row + col + 1, sz - col - 1, -factor);
        values[col] = 0.0f;
      }
    }
    return determinant;
  }

#if OSMIMATH_DETAIL_AVX2
  template<std::size_t Rows, std::size_t Vectors, bool Accumulate = false>
  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline void DotTile(float* result, const float* a, const float* b,
                             std::size_t cols, std::size_t inner, std::size_t a_stride) {
    __m256 sums[Rows][Vectors]{};
    if constexpr (Accumulate) {
      for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t vector = 0; vector < Vectors; ++vector) {
          sums[row][vector] = _mm256_loadu_ps(result + row * cols + vector * 8);
        }
      }
    }
    for (std::size_t k = 0; k < inner; ++k) {
      __m256 rhs[Vectors];
      for (std::size_t vector = 0; vector < Vectors; ++vector) {
        rhs[vector] = _mm256_loadu_ps(b + k * cols + vector * 8);
      }
      for (std::size_t row = 0; row < Rows; ++row) {
        const __m256 lhs = _mm256_broadcast_ss(a + row * a_stride + k);
        for (std::size_t vector = 0; vector < Vectors; ++vector) {
          sums[row][vector] = _mm256_add_ps(sums[row][vector], _mm256_mul_ps(lhs, rhs[vector]));
        }
      }
    }
    for (std::size_t row = 0; row < Rows; ++row) {
      for (std::size_t vector = 0; vector < Vectors; ++vector) {
        _mm256_storeu_ps(result + row * cols + vector * 8, sums[row][vector]);
      }
    }
  }

  template<std::size_t Rows>
  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline void DotRowTile(float* result, const float* a, const float* b,
                                std::size_t cols, std::size_t inner) {
    std::size_t col = 0;
    for (; cols - col >= 16; col += 16) {
      DotTile<Rows, 2>(result + col, a, b + col, cols, inner, inner);
    }
    if (cols - col >= 8) {
      DotTile<Rows, 1>(result + col, a, b + col, cols, inner, inner);
    }
    if (cols % 8 != 0) {
      DotTile<Rows, 1>(result + cols - 8, a, b + cols - 8, cols, inner, inner);
    }
  }

  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline void DotTiles(float* result, const float* a, const float* b,
                              std::size_t rows, std::size_t cols, std::size_t inner) {
    std::size_t row = 0;
    for (; rows - row >= 4; row += 4) {
      DotRowTile<4>(result + row * cols, a + row * inner, b, cols, inner);
    }
    for (; row < rows; ++row) {
      DotRowTile<1>(result + row * cols, a + row * inner, b, cols, inner);
    }
  }

  template<std::size_t Rows, std::size_t Inner, std::size_t Cols>
  OSMIMATH_DETAIL_TARGET
  static void DotRowsAVX2(float* result, const float* a, const float* b) {
    DotTiles(result, a, b, Rows, Cols, Inner);
  }

  OSMIMATH_DETAIL_TARGET
  static void DotRowsAVX2(float* result, const float* a, const float* b,
                          std::size_t rows, std::size_t cols, std::size_t inner) {
    DotTiles(result, a, b, rows, cols, inner);
  }

  static bool UseAVX2() noexcept {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__x86_64__))
    static const bool available = __builtin_cpu_supports("avx2");
    return available;
#else
    return true;
#endif
  }

  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline void AddScaledRowAVX2(
    float* __restrict destination, const float* __restrict source, std::size_t count, float factor) {
    const __m256 factor_vec = _mm256_set1_ps(factor);
    const std::size_t end = count - count % 8;
    std::size_t i = 0;
    for (; i < end; i += 8) {
      const __m256 product = _mm256_mul_ps(factor_vec, _mm256_loadu_ps(source + i));
      _mm256_storeu_ps(destination + i, _mm256_add_ps(_mm256_loadu_ps(destination + i), product));
    }
    AddScaledRow(destination + i, source + i, count - i, factor);
  }

  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline void DivideRowAVX2(float* values, std::size_t count, float divisor) {
    const __m256 divisor_vec = _mm256_set1_ps(divisor);
    const std::size_t end = count - count % 8;
    std::size_t i = 0;
    for (; i < end; i += 8) {
      _mm256_storeu_ps(values + i, _mm256_div_ps(_mm256_loadu_ps(values + i), divisor_vec));
    }
    DivideRow(values + i, count - i, divisor);
  }

  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline std::size_t FindPivotAVX2(const float* values, std::size_t size, std::size_t col) {
    if (size - col <= 16) {
      return FindPivot(values, size, col);
    }
    float largest = Abs(values[col * size + col]);
    __m128 maxima = _mm_set1_ps(largest);
    __m128i indices = _mm_set1_epi32(static_cast<int>(col));
    const __m128 sign = _mm_set1_ps(-0.0f);
    std::size_t row = col + 1;
    __m128i positions = _mm_add_epi32(indices, _mm_setr_epi32(1, 2, 3, 4));
    for (; size - row >= 4; row += 4) {
      const float* first = values + row * size + col;
      const __m128 magnitudes = _mm_andnot_ps(sign, _mm_setr_ps(first[0], first[size], first[2 * size], first[3 * size]));
      const __m128 greater = _mm_cmp_ps(magnitudes, maxima, _CMP_GT_OQ);
      maxima = _mm_blendv_ps(maxima, magnitudes, greater);
      indices = _mm_blendv_epi8(indices, positions, _mm_castps_si128(greater));
      positions = _mm_add_epi32(positions, _mm_set1_epi32(4));
    }
    float magnitudes[4];
    int winners[4];
    _mm_storeu_ps(magnitudes, maxima);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(winners), indices);
    std::size_t pivot = col;
    for (std::size_t lane = 0; lane < 4; ++lane) {
      const auto index = static_cast<std::size_t>(winners[lane]);
      if (magnitudes[lane] > largest || (magnitudes[lane] == largest && index < pivot)) {
        largest = magnitudes[lane];
        pivot = index;
      }
    }
    for (; row < size; ++row) {
      const float magnitude = Abs(values[row * size + col]);
      if (magnitude > largest) {
        largest = magnitude;
        pivot = row;
      }
    }
    return pivot;
  }

  template<std::size_t Size>
  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline void InverseSmallPivot(float* working, float* pivot_row, std::size_t col, float divisor) {
    __m256 masks[Size / 8];
    __m256 pivots[Size / 8];
    const __m256i index = _mm256_set1_epi32(static_cast<int>(col));
    const __m256i lanes = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256 denominator = _mm256_set1_ps(divisor);
    for (std::size_t i = 0; i < Size / 8; ++i) {
      const __m256i positions = _mm256_add_epi32(lanes, _mm256_set1_epi32(static_cast<int>(8 * i)));
      masks[i] = _mm256_castsi256_ps(_mm256_cmpeq_epi32(index, positions));
      const __m256 values = _mm256_loadu_ps(pivot_row + 8 * i);
      pivots[i] = _mm256_div_ps(_mm256_blendv_ps(values, _mm256_set1_ps(1.0f), masks[i]), denominator);
      _mm256_storeu_ps(pivot_row + 8 * i, pivots[i]);
    }
    for (std::size_t row = 0; row < Size; ++row) {
      if (row == col) {
        continue;
      }
      float* values = working + row * Size;
      const float factor = values[col];
      if (factor != 0.0f) {
        const __m256 scale = _mm256_set1_ps(-factor);
        for (std::size_t i = 0; i < Size / 8; ++i) {
          const __m256 current = _mm256_andnot_ps(masks[i], _mm256_loadu_ps(values + 8 * i));
          const __m256 updated = _mm256_add_ps(current, _mm256_mul_ps(scale, pivots[i]));
          _mm256_storeu_ps(values + 8 * i, updated);
        }
      }
    }
  }

  template<std::size_t Size = 0>
  OSMIMATH_DETAIL_TARGET
  static bool InverseAVX2(float* working, float* result, std::size_t* permutation, std::size_t size) {
    if constexpr (Size != 0) {
      size = Size;
    }
    for (std::size_t row = 0; row < size; ++row) {
      permutation[row] = row;
    }
    bool swapped = false;
    for (std::size_t col = 0; col < size; ++col) {
      const std::size_t pivot = FindPivot(working, size, col);
      float* pivot_values = working + pivot * size;
      const float divisor = pivot_values[col];
      if (divisor == 0.0f) {
        return false;
      }
      float* pivot_row = working + col * size;
      if (pivot != col) {
        SwapRows(pivot_row, pivot_values, size);
        std::swap(permutation[col], permutation[pivot]);
        swapped = true;
      }

      if constexpr (Size >= 8 && Size <= 32 && Size % 8 == 0) {
        InverseSmallPivot<Size>(working, pivot_row, col, divisor);
      } else {
        pivot_row[col] = 1.0f;
        DivideRowAVX2(pivot_row, size, divisor);
        const float inverse_pivot = pivot_row[col];
        for (std::size_t row = 0; row < size; ++row) {
          if (row == col) {
            continue;
          }
          float* values = working + row * size;
          const float factor = values[col];
          if (factor != 0.0f) {
            AddScaledRowAVX2(values, pivot_row, size, -factor);
            values[col] = -factor * inverse_pivot;
          }
        }
      }
    }

    if (!swapped) {
      std::copy_n(working, size * size, result);
      return true;
    }
    for (std::size_t row = 0; row < size; ++row) {
      for (std::size_t col = 0; col < size; ++col) {
        result[row * size + permutation[col]] = working[row * size + col];
      }
    }
    return true;
  }

  template<std::size_t Size>
  OSMIMATH_DETAIL_TARGET
  static float DeterminantAVX2(float* working, std::size_t size) {
    if constexpr (Size != 0) {
      size = Size;
    }
    float determinant = 1.0f;
    const std::size_t width = size < 96 ? size : 8;
    for (std::size_t start = 0; start < size; start += width) {
      const std::size_t end = std::min(start + width, size);
      for (std::size_t col = start; col < end; ++col) {
        const std::size_t pivot = FindPivotAVX2(working, size, col);
        float* pivot_values = working + pivot * size;
        const float divisor = pivot_values[col];
        if (divisor == 0.0f) {
          return 0.0f;
        }
        float* pivot_row = working + col * size;
        if (pivot != col) {
          const std::size_t first = width == size ? col : start;
          SwapRows(pivot_row + first, pivot_values + first, size - first);
          determinant = -determinant;
        }
        determinant *= divisor;
        if (width == size) {
          EliminateDirect(working, pivot_row, size, col, divisor);
        } else {
          EliminatePanel(working, pivot_row, size, col, end, divisor);
        }
      }
      if (end < size) {
        UpdateAfterPanel(working, size, start, end);
      }
    }
    return determinant;
  }

  template<std::size_t Rows>
  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline void EliminationFactors(float (&factors)[Rows], const float* values,
                                        std::size_t size, std::size_t col, float divisor) {
    if constexpr (Rows == 8) {
      const __m256 numerators = _mm256_setr_ps(values[col], values[size + col], values[2 * size + col],
          values[3 * size + col], values[4 * size + col], values[5 * size + col], values[6 * size + col],
          values[7 * size + col]);
      _mm256_storeu_ps(factors, _mm256_div_ps(numerators, _mm256_set1_ps(-divisor)));
    } else if constexpr (Rows == 4) {
      const __m128 numerators = _mm_setr_ps(values[col], values[size + col],
                                            values[2 * size + col], values[3 * size + col]);
      _mm_storeu_ps(factors, _mm_div_ps(numerators, _mm_set1_ps(-divisor)));
    } else {
      factors[0] = -values[col] / divisor;
    }
  }

  template<std::size_t Rows>
  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline void EliminateDirectRows(float* __restrict values, const float* __restrict pivot,
                                          std::size_t size, std::size_t col, std::size_t begin,
                                          float divisor) {
    float factors[Rows];
    EliminationFactors(factors, values, size, col, divisor);
    __m256 scales[Rows];
    for (std::size_t row = 0; row < Rows; ++row) {
      scales[row] = _mm256_set1_ps(factors[row]);
    }
    for (std::size_t i = begin; i < size; i += 8) {
      const __m256 source = _mm256_loadu_ps(pivot + i);
      for (std::size_t row = 0; row < Rows; ++row) {
        float* target = values + row * size + i;
        const __m256 current = _mm256_loadu_ps(target);
#if defined(__FMA__)
        const __m256 updated = _mm256_fmadd_ps(source, scales[row], current);
#else
        const __m256 updated = _mm256_add_ps(current, _mm256_mul_ps(source, scales[row]));
#endif
        _mm256_storeu_ps(target, updated);
      }
    }
    for (std::size_t row = 0; row < Rows; ++row) {
      for (std::size_t i = col + 1; i < begin; ++i) {
        values[row * size + i] += factors[row] * pivot[i];
      }
    }
  }

  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline void EliminateDirect(float* working, const float* pivot, std::size_t size,
                                      std::size_t col, float divisor) {
    const std::size_t begin = size - std::min(size / 8, (size - col + 6) / 8) * 8;
    std::size_t row = col + 1;
    for (; size - row >= 8; row += 8) {
      EliminateDirectRows<8>(working + row * size, pivot, size, col, begin, divisor);
    }
    if (size - row >= 4) {
      EliminateDirectRows<4>(working + row * size, pivot, size, col, begin, divisor);
      row += 4;
    }
    for (; row < size; ++row) {
      EliminateDirectRows<1>(working + row * size, pivot, size, col, begin, divisor);
    }
  }

  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline void EliminatePanel(float* working, const float* pivot_row, std::size_t size,
                                    std::size_t col, std::size_t end, float divisor) {
    const std::size_t first = col + 1;
    std::size_t row = first;
    for (; size - row >= 4; row += 4) {
      float* values = working + row * size;
      float factors[4];
      EliminationFactors(factors, values, size, col, divisor);
      for (std::size_t i = 0; i < 4; ++i) {
        float* target = values + i * size;
        target[col] = factors[i];
        AddScaledRowAVX2(target + first, pivot_row + first, end - first, factors[i]);
      }
    }
    for (; row < size; ++row) {
      float* values = working + row * size;
      const float factor = -values[col] / divisor;
      values[col] = factor;
      AddScaledRowAVX2(values + first, pivot_row + first, end - first, factor);
    }
  }

  template<std::size_t Rows>
  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline void UpdateTrailingRows(float* values, const float* pivots, std::size_t size,
                                        std::size_t start, std::size_t end) {
    const std::size_t inner = end - start;
    std::size_t col = end;
    for (; size - col >= 16; col += 16) {
      DotTile<Rows, 2, true>(values + col, values + start, pivots + col, size, inner, size);
    }
    if (size - col >= 8) {
      DotTile<Rows, 1, true>(values + col, values + start, pivots + col, size, inner, size);
      col += 8;
    }
    for (std::size_t row = 0; row < Rows; ++row) {
      for (std::size_t k = 0; k < inner; ++k) {
        AddScaledRow(values + row * size + col, pivots + k * size + col, size - col,
                     values[row * size + start + k]);
      }
    }
  }

  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline void UpdateAfterPanel(float* working, std::size_t size, std::size_t start,
                                      std::size_t end) {
    for (std::size_t row = start + 1; row < end; ++row) {
      float* values = working + row * size;
      for (std::size_t k = start; k < row; ++k) {
        AddScaledRowAVX2(values + end, working + k * size + end, size - end, values[k]);
      }
    }
    const float* pivots = working + start * size;
    std::size_t row = end;
    for (; size - row >= 4; row += 4) {
      UpdateTrailingRows<4>(working + row * size, pivots, size, start, end);
    }
    for (; row < size; ++row) {
      UpdateTrailingRows<1>(working + row * size, pivots, size, start, end);
    }
  }

  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline void X4Transpose(const __m256* rows, __m256* lanes) {
    const __m256 lo0 = _mm256_unpacklo_ps(rows[0], rows[1]);
    const __m256 hi0 = _mm256_unpackhi_ps(rows[0], rows[1]);
    const __m256 lo1 = _mm256_unpacklo_ps(rows[2], rows[3]);
    const __m256 hi1 = _mm256_unpackhi_ps(rows[2], rows[3]);
    lanes[0] = _mm256_shuffle_ps(lo0, lo1, 0x44);
    lanes[1] = _mm256_shuffle_ps(lo0, lo1, 0xee);
    lanes[2] = _mm256_shuffle_ps(hi0, hi1, 0x44);
    lanes[3] = _mm256_shuffle_ps(hi0, hi1, 0xee);
  }

  [[gnu::always_inline]] OSMIMATH_DETAIL_TARGET
  static inline void TransposeTile(
    const float* source, float* destination, std::size_t src_stride, std::size_t dst_stride) {
    __m256 rows[8];
    for (std::size_t i = 0; i < 8; ++i) {
      rows[i] = _mm256_loadu_ps(source + i * src_stride);
    }

    __m256 lanes[8];
    for (std::size_t i = 0; i < 8; i += 4) {
      X4Transpose(rows + i, lanes + i);
    }
    for (std::size_t i = 0; i < 4; ++i) {
      _mm256_storeu_ps(destination + i * dst_stride,
                        _mm256_permute2f128_ps(lanes[i], lanes[i + 4], 0x20));
      _mm256_storeu_ps(destination + (i + 4) * dst_stride,
                        _mm256_permute2f128_ps(lanes[i], lanes[i + 4], 0x31));
    }
  }

  OSMIMATH_DETAIL_TARGET
  static void TransposeAVX2(const float* source, float* destination,
                            std::size_t rows, std::size_t cols) {
    constexpr std::size_t block = 32;
    const std::size_t tiled_rows = rows - rows % 8;
    const std::size_t tiled_cols = cols - cols % 8;
    for (std::size_t i = 0; i < tiled_rows; i += block) {
      for (std::size_t j = 0; j < tiled_cols; j += block) {
        const std::size_t end_row = std::min(i + block, tiled_rows);
        const std::size_t end_col = std::min(j + block, tiled_cols);
        for (std::size_t col = j; col < end_col; col += 8) {
          for (std::size_t row = i; row < end_row; row += 8) {
            TransposeTile(source + row * cols + col, destination + col * rows + row, cols, rows);
          }
        }
      }
    }
    for (std::size_t col = tiled_cols; col < cols; ++col) {
      for (std::size_t row = 0; row < tiled_rows; ++row) {
        destination[col * rows + row] = source[row * cols + col];
      }
    }
    for (std::size_t col = 0; col < cols; ++col) {
      for (std::size_t row = tiled_rows; row < rows; ++row) {
        destination[col * rows + row] = source[row * cols + col];
      }
    }
  }
#endif
};

#undef OSMIMATH_DETAIL_AVX2
#undef OSMIMATH_DETAIL_TARGET

#endif

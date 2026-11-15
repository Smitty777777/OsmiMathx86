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
      if (Cols >= 8 && UseAVX2()) {
        DotRowsAVX2<Rows, Inner, Cols>(result.data(), a.data(), b.data());
        return result;
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
    auto result = matrix<Size, Size>::identity();
    if (!InverseRows(working.data(), result.data(), Size)) {
      return std::nullopt;
    }

    return result;
  }

  template<std::size_t Size>
  [[nodiscard]] static constexpr float AVX2Det(const matrix<Size, Size>& source) {
    auto working = source;
    return DeterminantRows(working.data(), Size);
  }

  static constexpr void AVX2Transpose(
    const float* source, float* destination, std::size_t M, std::size_t N) {
    if (CheckedCount(M, N) == 0) {
      return;
    }

    constexpr std::size_t block_size = 4;
    for (std::size_t i = 0; i < M; i += block_size) {
      for (std::size_t j = 0; j < N; j += block_size) {
        const std::size_t limit_i = i + std::min(block_size, M - i);
        const std::size_t limit_j = j + std::min(block_size, N - j);
#if OSMIMATH_DETAIL_AVX2
        if !consteval {
          if (limit_i - i == block_size && limit_j - j == block_size && UseAVX2()) {
            X4Transpose(source + i * N + j, destination + j * M + i, N, M);
            continue;
          }
        }
#endif
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
    if (count == 0) {
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
    auto result = std::make_unique<float[]>(count);
    std::copy_n(source, count, working.get());
    for (std::size_t i = 0; i < sz; ++i) {
      result[i * sz + i] = 1.0f;
    }

    if (!InverseRows(working.get(), result.get(), sz)) {
      throw std::runtime_error("Cannot invert a singular matrix");
    }

    std::copy_n(result.get(), count, destination);
  }

  static constexpr void getCoefficient(
    const float* source, float* coefficients, std::size_t p, std::size_t q, std::size_t n) {
    if (p >= n || q >= n) {
      throw std::out_of_range("Minor index is out of range");
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
    return DeterminantRows(working.get(), sz);
  }

private:
  static constexpr std::size_t CheckedCount(std::size_t rows, std::size_t cols) {
    const std::size_t max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (rows != 0 && cols > max_elements / rows) {
      throw std::length_error("Matrix dimensions are too large");
    }

    return rows * cols;
  }

  static constexpr float Abs(float value) noexcept {
    return std::abs(value);
  }

  static constexpr std::size_t FindPivot(const float* values, std::size_t sz, std::size_t col) {
    std::size_t pivot = col;
    for (std::size_t row = col + 1; row < sz; ++row) {
      if (Abs(values[row * sz + col]) > Abs(values[pivot * sz + col])) {
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
    std::size_t i = 0;
#if OSMIMATH_DETAIL_AVX2
    if !consteval {
      if (count >= 8 && UseAVX2()) {
        i = AddScaledRowAVX2(destination, source, count, factor);
      }
    }
#endif
    for (; i < count; ++i) {
      destination[i] += factor * source[i];
    }
  }

  static constexpr void DivideRow(float* values, std::size_t count, float divisor) {
    std::size_t i = 0;
#if OSMIMATH_DETAIL_AVX2
    if !consteval {
      if (count >= 8 && UseAVX2()) {
        i = DivideRowAVX2(values, count, divisor);
      }
    }
#endif
    for (; i < count; ++i) {
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
  template<std::size_t Rows, std::size_t Inner, std::size_t Cols>
  OSMIMATH_DETAIL_TARGET
  static void DotRowsAVX2(float* result, const float* a, const float* b) {
    DotRowsAVX2(result, a, b, Rows, Cols, Inner);
  }

  OSMIMATH_DETAIL_TARGET
  static void DotRowsAVX2(float* result, const float* a, const float* b,
    std::size_t rows, std::size_t cols, std::size_t inner) {
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t k = 0; k < inner; ++k) {
        const float factor = a[row * inner + k];
        std::size_t col = AddScaledRowAVX2(result + row * cols, b + k * cols, cols, factor);
        for (; col < cols; ++col) {
          result[row * cols + col] += factor * b[k * cols + col];
        }
      }
    }
  }

  static bool UseAVX2() noexcept {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__x86_64__))
    static const bool available = __builtin_cpu_supports("avx2");
    return available;
#else
    return true;
#endif
  }

  OSMIMATH_DETAIL_TARGET
  static std::size_t AddScaledRowAVX2(
    float* destination, const float* source, std::size_t count, float factor) {
    const __m256 factor_vec = _mm256_set1_ps(factor);
    const std::size_t end = count - count % 8;
    for (std::size_t i = 0; i < end; i += 8) {
      const __m256 product = _mm256_mul_ps(factor_vec, _mm256_loadu_ps(source + i));
      _mm256_storeu_ps(destination + i, _mm256_add_ps(_mm256_loadu_ps(destination + i), product));
    }
    return end;
  }

  OSMIMATH_DETAIL_TARGET
  static std::size_t DivideRowAVX2(float* values, std::size_t count, float divisor) {
    const __m256 divisor_vec = _mm256_set1_ps(divisor);
    const std::size_t end = count - count % 8;
    for (std::size_t i = 0; i < end; i += 8) {
      _mm256_storeu_ps(values + i, _mm256_div_ps(_mm256_loadu_ps(values + i), divisor_vec));
    }
    return end;
  }

  OSMIMATH_DETAIL_TARGET
  static void X4Transpose(
    const float* source, float* destination, std::size_t src_stride, std::size_t dst_stride) {
    __m128 row0 = _mm_loadu_ps(source);
    __m128 row1 = _mm_loadu_ps(source + src_stride);
    __m128 row2 = _mm_loadu_ps(source + 2 * src_stride);
    __m128 row3 = _mm_loadu_ps(source + 3 * src_stride);
    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);
    _mm_storeu_ps(destination, row0);
    _mm_storeu_ps(destination + dst_stride, row1);
    _mm_storeu_ps(destination + 2 * dst_stride, row2);
    _mm_storeu_ps(destination + 3 * dst_stride, row3);
  }
#endif
};

#undef OSMIMATH_DETAIL_AVX2
#undef OSMIMATH_DETAIL_TARGET

#endif

#ifndef OSMIMATH_MODEL_DETAIL_HPP
#define OSMIMATH_MODEL_DETAIL_HPP

#include "../matrix.hpp"
#include "model_options.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace osmimath_detail {

constexpr double Abs(double value) noexcept {
  return std::abs(value);
}

constexpr bool IsFinite(double value) noexcept {
  return std::isfinite(value);
}

constexpr void Validate(const model_options& options) {
  if ((options.penalty != regularization::none && options.penalty != regularization::ridge &&
        options.penalty != regularization::lasso) ||
    !IsFinite(options.alpha) || options.alpha < 0 || !IsFinite(options.tolerance) ||
    options.tolerance <= 0 || options.max_iterations == 0) {
    throw std::invalid_argument("Invalid model options");
  }
}

constexpr double Sqrt(double value) {
  if !consteval {
    return std::sqrt(value);
  }
  if (!(value >= 0.0)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (value == 0.0 || value == std::numeric_limits<double>::infinity()) {
    return value;
  }

  double scale = 1.0;
  while (value < 1.0) {
    value *= 4.0;
    scale *= 0.5;
  }
  while (value >= 4.0) {
    value *= 0.25;
    scale *= 2.0;
  }

  double result = value;
  for (std::size_t iteration = 0; iteration < 16; ++iteration) {
    const double next = 0.5 * (result + value / result);
    if (next == result) {
      break;
    }
    result = next;
  }

  return result * scale;
}

constexpr double SoftThreshold(double value, double threshold) noexcept {
  if (value > threshold) {
    return value - threshold;
  }
  if (value < -threshold) {
    return value + threshold;
  }

  return 0.0;
}

inline double Sigmoid(double value) noexcept {
  if (value >= 0) {
    return 1.0 / (1.0 + std::exp(-value));
  }

  const double exponential = std::exp(value);
  return exponential / (1.0 + exponential);
}

inline double LogisticLoss(double score, double label) noexcept {
  return (label == 1.0 ? std::max(-score, 0.0) : std::max(score, 0.0)) +
    std::log1p(std::exp(-Abs(score)));
}

template<std::size_t Rows, std::size_t Cols>
constexpr double Mean(const AVX2Math::matrix<Rows, Cols>& data, std::size_t col) {
  const double base = data(0, col);
  double sum = 0.0;

  for (std::size_t row = 0; row < Rows; ++row) {
    const double value = data(row, col);
    if (!IsFinite(value)) {
      throw std::invalid_argument("model data must be finite");
    }
    sum += value - base;
  }

  return base + sum / static_cast<double>(Rows);
}

template<std::size_t Rows, std::size_t Cols, std::size_t Target, std::size_t WorkRows = Rows>
struct design {
  static constexpr std::size_t predictors = Cols - 1;

  std::array<double, WorkRows * predictors> values{};
  std::array<double, predictors> means{};
  std::array<double, predictors> scales{};
  std::array<std::size_t, predictors> order{};

  constexpr double& operator()(std::size_t row, std::size_t col) noexcept {
    return values[col * WorkRows + row];
  }

  constexpr bool Prepare(const AVX2Math::matrix<Rows, Cols>& data, bool require_full_rank) {
    if (require_full_rank && predictors >= Rows) {
      return false;
    }

    std::size_t predictor = 0;
    for (std::size_t col = 0; col < Cols; ++col) {
      if (col == Target) {
        continue;
      }

      const double mean = Mean(data, col);
      means[predictor] = mean;
      order[predictor] = predictor;

      double scale = 0.0;
      for (std::size_t row = 0; row < Rows; ++row) {
        (*this)(row, predictor) = double(data(row, col)) - mean;
        scale = std::max(scale, Abs((*this)(row, predictor)));
      }

      if (scale == 0.0) {
        if (require_full_rank) {
          return false;
        }
        scales[predictor++] = 1.0;
        continue;
      }

      double squared_norm = 0.0;
      for (std::size_t row = 0; row < Rows; ++row) {
        (*this)(row, predictor) /= scale;
        squared_norm += (*this)(row, predictor) * (*this)(row, predictor);
      }

      const double norm = Sqrt(squared_norm);
      scales[predictor] = scale * norm;
      for (std::size_t row = 0; row < Rows; ++row) {
        (*this)(row, predictor) /= norm;
      }
      ++predictor;
    }

    return true;
  }
};

template<std::size_t Rows, std::size_t Cols, std::size_t Target, std::size_t WorkRows>
constexpr bool Factorize(design<Rows, Cols, Target, WorkRows>& working,
  std::array<double, WorkRows>& rhs, double tolerance) {
  constexpr std::size_t predictors = Cols - 1;

  for (std::size_t col = 0; col < predictors; ++col) {
    std::size_t pivot = col;
    double squared_norm = 0.0;
    for (std::size_t candidate = col; candidate < predictors; ++candidate) {
      double candidate_norm = 0.0;
      for (std::size_t row = col; row < WorkRows; ++row) {
        candidate_norm += working(row, candidate) * working(row, candidate);
      }
      if (candidate_norm > squared_norm) {
        squared_norm = candidate_norm;
        pivot = candidate;
      }
    }

    if (squared_norm <= tolerance * tolerance) {
      return false;
    }
    if (pivot != col) {
      for (std::size_t row = 0; row < WorkRows; ++row) {
        std::swap(working(row, col), working(row, pivot));
      }
      std::swap(working.order[col], working.order[pivot]);
    }

    const double first = working(col, col);
    const double diagonal = first < 0.0 ? Sqrt(squared_norm) : -Sqrt(squared_norm);
    const double divisor = first - diagonal;
    const double tau = (diagonal - first) / diagonal;
    working(col, col) = diagonal;
    for (std::size_t row = col + 1; row < WorkRows; ++row) {
      working(row, col) /= divisor;
    }

    for (std::size_t k = col + 1; k < predictors; ++k) {
      double projection = working(col, k);
      for (std::size_t row = col + 1; row < WorkRows; ++row) {
        projection += working(row, col) * working(row, k);
      }
      projection *= tau;
      working(col, k) -= projection;
      for (std::size_t row = col + 1; row < WorkRows; ++row) {
        working(row, k) -= working(row, col) * projection;
      }
    }

    double projection = rhs[col];
    for (std::size_t row = col + 1; row < WorkRows; ++row) {
      projection += working(row, col) * rhs[row];
    }
    projection *= tau;
    rhs[col] -= projection;
    for (std::size_t row = col + 1; row < WorkRows; ++row) {
      rhs[row] -= working(row, col) * projection;
    }
  }

  return true;
}

template<std::size_t Rows, std::size_t Cols, std::size_t Target, std::size_t WorkRows>
constexpr std::array<double, Cols> BackSolve(design<Rows, Cols, Target, WorkRows>& working,
  const std::array<double, WorkRows>& rhs, double mean_y) {
  constexpr std::size_t predictors = Cols - 1;
  std::array<double, Cols> result{};
  std::array<double, predictors> solution{};

  for (std::size_t remaining = predictors; remaining > 0; --remaining) {
    const std::size_t col = remaining - 1;
    double value = rhs[col];
    for (std::size_t k = col + 1; k < predictors; ++k) {
      value -= working(col, k) * solution[k];
    }
    solution[col] = value / working(col, col);
    const std::size_t original = working.order[col];
    result[original + 1] = solution[col] / working.scales[original];
  }

  result[0] = mean_y;
  for (std::size_t col = 0; col < predictors; ++col) {
    result[0] -= result[col + 1] * working.means[col];
  }

  return result;
}

}

#endif

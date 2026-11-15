#ifndef OSMIMATH_LINEAR_MODEL_HPP
#define OSMIMATH_LINEAR_MODEL_HPP

#include "model_detail.hpp"

template<std::size_t Rows, std::size_t Cols, std::size_t Target>
class linear_model {
  static_assert(Rows >= 10 && Target < Cols, "target outside column range or insufficient data");

public:
  using data_type = AVX2Math::matrix<Rows, Cols>;
  using coefficient_type = std::array<double, Cols>;

  static constexpr std::size_t predictors = Cols - 1;

  explicit constexpr linear_model(const data_type& data, model_options options = {}) {
    Fit(data, options);
  }

  constexpr void Fit(const data_type& data) {
    Fit(data, options_);
  }

  constexpr void Fit(const data_type& data, model_options options) {
    osmimath_detail::Validate(options);
    const auto result = FitCoefficients(data, options);

    coefficients_ = result.coefficients;
    iterations_ = result.iterations;
    options_ = options;
  }

  [[nodiscard]] constexpr const coefficient_type& Coefficients() const noexcept {
    return coefficients_;
  }

  [[nodiscard]] constexpr double Intercept() const noexcept {
    return coefficients_[0];
  }

  [[nodiscard]] constexpr const model_options& Options() const noexcept {
    return options_;
  }

  [[nodiscard]] constexpr std::size_t Iterations() const noexcept {
    return iterations_;
  }

  [[nodiscard]] constexpr double Predict(
    const std::array<float, predictors>& values) const noexcept {
    double result = Intercept();
    for (std::size_t col = 0; col < predictors; ++col) {
      result += coefficients_[col + 1] * values[col];
    }

    return result;
  }

  template<std::size_t Count>
  [[nodiscard]] constexpr std::array<double, Count> Predict(
    const AVX2Math::matrix<Count, predictors>& values) const noexcept {
    std::array<double, Count> result{};
    for (std::size_t row = 0; row < Count; ++row) {
      result[row] = Intercept();
      for (std::size_t col = 0; col < predictors; ++col) {
        result[row] += coefficients_[col + 1] * values(row, col);
      }
    }

    return result;
  }

  [[nodiscard("do not ignore colinear check")]] static constexpr bool IsColinear(
    const data_type& table) {
    osmimath_detail::design<Rows, Cols, Target> working{};
    if (!working.Prepare(table, true)) {
      return true;
    }
    if constexpr (predictors <= 1) {
      return false;
    }

    std::array<double, Rows> rhs{};
    return !osmimath_detail::Factorize(working, rhs, RankTolerance());
  }

private:
  struct fit_result {
    coefficient_type coefficients{};
    std::size_t iterations = 0;
  };

  coefficient_type coefficients_{};
  model_options options_{};
  std::size_t iterations_ = 0;

  static constexpr double RankTolerance() noexcept {
    return std::numeric_limits<float>::epsilon() * static_cast<double>(std::max(Rows, predictors));
  }

  static constexpr fit_result FitRidge(
    const data_type& data, const model_options& options, double mean_y) {
    osmimath_detail::design<Rows, Cols, Target, Rows + predictors> working{};
    working.Prepare(data, false);
    std::array<double, Rows + predictors> rhs{};

    for (std::size_t row = 0; row < Rows; ++row) {
      rhs[row] = double(data(row, Target)) - mean_y;
    }

    const double penalty_scale =
      osmimath_detail::Sqrt(double(Rows)) * osmimath_detail::Sqrt(options.alpha);
    for (std::size_t col = 0; col < predictors; ++col) {
      const double diagonal = penalty_scale / working.scales[col];
      if (!osmimath_detail::IsFinite(diagonal)) {
        throw std::invalid_argument("ridge penalty exceeds numeric range");
      }

      double squared_norm = 0;
      for (std::size_t row = 0; row < Rows; ++row) {
        squared_norm += working(row, col) * working(row, col);
      }

      const double data_norm = osmimath_detail::Sqrt(squared_norm);
      const double maximum = std::max(data_norm, diagonal);
      const double norm = maximum *
        osmimath_detail::Sqrt((data_norm / maximum) * (data_norm / maximum) +
          (diagonal / maximum) * (diagonal / maximum));
      for (std::size_t row = 0; row < Rows; ++row) {
        working(row, col) /= norm;
      }
      working(Rows + col, col) = diagonal / norm;
      working.scales[col] *= norm;
    }

    const double tolerance = std::numeric_limits<double>::epsilon() * double(Rows + predictors);
    if (!osmimath_detail::Factorize(working, rhs, tolerance)) {
      throw std::runtime_error("ridge design matrix exceeds supported");
    }

    return {osmimath_detail::BackSolve(working, rhs, mean_y), 0};
  }

  static constexpr fit_result FitLasso(
    const data_type& data, const model_options& options, double mean_y) {
    osmimath_detail::design<Rows, Cols, Target> working{};
    working.Prepare(data, false);
    std::array<double, Rows> residual{};
    double target_scale = 0.0;

    for (std::size_t row = 0; row < Rows; ++row) {
      residual[row] = double(data(row, Target)) - mean_y;
      target_scale = std::max(target_scale, osmimath_detail::Abs(residual[row]));
    }

    std::array<double, predictors> solution{};
    std::array<double, predictors> norms{};
    for (std::size_t col = 0; col < predictors; ++col) {
      for (std::size_t row = 0; row < Rows; ++row) {
        norms[col] += working(row, col) * working(row, col);
      }
      norms[col] /= double(Rows);
    }

    for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration) {
      for (std::size_t col = 0; col < predictors; ++col) {
        if (norms[col] == 0) {
          continue;
        }

        double correlation = 0.0;
        for (std::size_t row = 0; row < Rows; ++row) {
          correlation += working(row, col) * residual[row];
        }
        correlation = correlation / double(Rows) + solution[col] * norms[col];
        const double next =
          osmimath_detail::SoftThreshold(correlation, options.alpha / working.scales[col]) /
          norms[col];
        const double change = next - solution[col];
        for (std::size_t row = 0; row < Rows; ++row) {
          residual[row] -= working(row, col) * change;
        }
        solution[col] = next;
      }

      double violation = 0.0;
      for (std::size_t col = 0; col < predictors; ++col) {
        double correlation = 0.0;
        for (std::size_t row = 0; row < Rows; ++row) {
          correlation += working(row, col) * residual[row];
        }
        correlation /= double(Rows);
        const double threshold = options.alpha / working.scales[col];
        const double error = solution[col] == 0
          ? std::max(osmimath_detail::Abs(correlation) - threshold, 0.0)
          : osmimath_detail::Abs(correlation - (solution[col] > 0 ? threshold : -threshold));
        violation = std::max(violation, error);
      }

      if (violation <= options.tolerance * std::max(1.0, target_scale)) {
        fit_result result{};
        result.iterations = iteration + 1;
        result.coefficients[0] = mean_y;

        for (std::size_t col = 0; col < predictors; ++col) {
          result.coefficients[col + 1] = solution[col] / working.scales[col];
          result.coefficients[0] -= result.coefficients[col + 1] * working.means[col];
        }

        return result;
      }
    }

    throw std::runtime_error("did not converge within max_iterations");
  }

  static constexpr fit_result FitCoefficients(const data_type& data, const model_options& options) {
    const double mean_y = osmimath_detail::Mean(data, Target);
    if (options.alpha > 0 && options.penalty == regularization::ridge) {
      return FitRidge(data, options, mean_y);
    }
    if (options.alpha > 0 && options.penalty == regularization::lasso) {
      return FitLasso(data, options, mean_y);
    }

    osmimath_detail::design<Rows, Cols, Target> working{};
    if (!working.Prepare(data, true)) {
      throw std::invalid_argument("design matrix is rank deficient");
    }

    std::array<double, Rows> rhs{};
    for (std::size_t row = 0; row < Rows; ++row) {
      rhs[row] = double(data(row, Target)) - mean_y;
    }

    if constexpr (predictors == 1) {
      double xy = 0.0, xx = 0.0;
      for (std::size_t row = 0; row < Rows; ++row) {
        xy += working(row, 0) * rhs[row];
        xx += working(row, 0) * working(row, 0);
      }

      const double slope = (xy / xx) / working.scales[0];
      return {{{mean_y - slope * working.means[0], slope}}, 0};
    } else {
      if (!osmimath_detail::Factorize(working, rhs, RankTolerance())) {
        throw std::invalid_argument("design matrix is rank deficient");
      }
      return {osmimath_detail::BackSolve(working, rhs, mean_y), 0};
    }
  }
};

#endif

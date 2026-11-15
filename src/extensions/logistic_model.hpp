#ifndef OSMIMATH_LOGISTIC_MODEL_HPP
#define OSMIMATH_LOGISTIC_MODEL_HPP

#include "model_detail.hpp"

template<std::size_t Rows, std::size_t Cols, std::size_t Target>
class logistic_model {
  static_assert(Rows >= 10 && Target < Cols, "target outside column range or insufficient data");

public:
  using data_type = AVX2Math::matrix<Rows, Cols>;
  using coefficient_type = std::array<double, Cols>;

  static constexpr std::size_t predictors = Cols - 1;

  explicit logistic_model(const data_type& data,
    model_options options = {.penalty = regularization::ridge, .alpha = 1e-4}) {
    Fit(data, options);
  }

  void Fit(const data_type& data) {
    Fit(data, options_);
  }

  void Fit(const data_type& data, model_options options) {
    osmimath_detail::Validate(options);
    const auto result = FitCoefficients(data, options);

    coefficients_ = result.coefficients;
    iterations_ = result.iterations;
    options_ = options;
  }

  [[nodiscard]] const coefficient_type& Coefficients() const noexcept {
    return coefficients_;
  }

  [[nodiscard]] double Intercept() const noexcept {
    return coefficients_[0];
  }

  [[nodiscard]] const model_options& Options() const noexcept {
    return options_;
  }

  [[nodiscard]] std::size_t Iterations() const noexcept {
    return iterations_;
  }

  [[nodiscard]] double PredictProbability(const std::array<float, predictors>& values) const {
    double score = Intercept();
    for (std::size_t col = 0; col < predictors; ++col) {
      if (!osmimath_detail::IsFinite(values[col])) {
        throw std::invalid_argument("preedictors must be finite");
      }
      score += coefficients_[col + 1] * values[col];
    }

    return osmimath_detail::Sigmoid(score);
  }

  [[nodiscard]] bool Predict(
    const std::array<float, predictors>& values, double threshold = 0.5) const {
    ValidateThreshold(threshold);
    return PredictProbability(values) >= threshold;
  }

  template<std::size_t Count>
  [[nodiscard]] std::array<double, Count> PredictProbability(
    const AVX2Math::matrix<Count, predictors>& data) const {
    std::array<double, Count> result{};
    for (std::size_t row = 0; row < Count; ++row) {
      std::array<float, predictors> values{};
      for (std::size_t col = 0; col < predictors; ++col) {
        values[col] = data(row, col);
      }
      result[row] = PredictProbability(values);
    }

    return result;
  }

  template<std::size_t Count>
  [[nodiscard]] std::array<bool, Count> Predict(
    const AVX2Math::matrix<Count, predictors>& data, double threshold = 0.5) const {
    ValidateThreshold(threshold);
    const auto probabilities = PredictProbability(data);

    std::array<bool, Count> result{};
    for (std::size_t row = 0; row < Count; ++row) {
      result[row] = probabilities[row] >= threshold;
    }

    return result;
  }

private:
  struct fit_result {
    coefficient_type coefficients{};
    std::size_t iterations = 0;
  };

  coefficient_type coefficients_{};
  model_options options_{};
  std::size_t iterations_ = 0;

  static void ValidateThreshold(double threshold) {
    if (!(threshold >= 0 && threshold <= 1)) {
      throw std::invalid_argument("threshold must be in [0, 1]");
    }
  }

  static fit_result FitCoefficients(const data_type& data, const model_options& options) {
    const double mean_y = osmimath_detail::Mean(data, Target);
    for (std::size_t row = 0; row < Rows; ++row) {
      if (data(row, Target) != 0 && data(row, Target) != 1) {
        throw std::invalid_argument("logistic regression requires binary labels 0 and 1");
      }
    }
    if (mean_y == 0 || mean_y == 1) {
      throw std::invalid_argument("logistic regression requires both classes");
    }

    osmimath_detail::design<Rows, Cols, Target> working{};
    working.Prepare(data, false);
    const double root_rows = std::sqrt(double(Rows));
    for (std::size_t col = 0; col < predictors; ++col) {
      working.scales[col] /= root_rows;
      for (std::size_t row = 0; row < Rows; ++row) {
        working(row, col) *= root_rows;
      }
    }

    coefficient_type solution{}, l1{}, l2{};
    solution[0] = std::log(mean_y) - std::log1p(-mean_y);
    for (std::size_t col = 0; col < predictors; ++col) {
      if (options.penalty == regularization::lasso) {
        l1[col + 1] = options.alpha / working.scales[col];
      }
      if (options.penalty == regularization::ridge) {
        l2[col + 1] = options.alpha / working.scales[col] / working.scales[col];
      }
      if (!osmimath_detail::IsFinite(l1[col + 1]) || !osmimath_detail::IsFinite(l2[col + 1])) {
        throw std::invalid_argument("logistic penalty exceeds numeric range");
      }
    }

    std::array<double, Rows> scores{};
    scores.fill(solution[0]);

    for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration) {
      for (std::size_t col = 0; col < Cols; ++col) {
        double gradient = 0, curvature = 0;
        for (std::size_t row = 0; row < Rows; ++row) {
          const double x = col == 0 ? 1.0 : working(row, col - 1);
          const double p = osmimath_detail::Sigmoid(scores[row]);
          gradient += x * (p - data(row, Target));
          curvature += x * x * p * (1 - p);
        }

        gradient /= double(Rows);
        curvature /= double(Rows);
        const double denominator = curvature + l2[col];
        if (denominator == 0) {
          continue;
        }

        const double next =
          osmimath_detail::SoftThreshold(curvature * solution[col] - gradient, l1[col]) /
          denominator;
        const double delta = next - solution[col];
        if (delta == 0) {
          continue;
        }

        const auto loss = [&](double change) {
          double result = 0;
          for (std::size_t row = 0; row < Rows; ++row) {
            const double x = col == 0 ? 1.0 : working(row, col - 1);
            result += osmimath_detail::LogisticLoss(scores[row] + change * x, data(row, Target));
          }

          result /= double(Rows);
          for (std::size_t k = 0; k < Cols; ++k) {
            const double value = solution[k] + (k == col ? change : 0);
            result += l1[k] * osmimath_detail::Abs(value) + 0.5 * l2[k] * value * value;
          }
          return result;
        };

        const double previous = loss(0);
        double step = 1.0;
        bool accepted = false;
        for (std::size_t attempt = 0; attempt < 60; ++attempt) {
          const double candidate = loss(step * delta);
          if (osmimath_detail::IsFinite(candidate) &&
            candidate <= previous + 1e-14 * std::max(1.0, previous)) {
            accepted = true;
            break;
          }
          step *= 0.5;
        }

        if (!accepted) {
          throw std::runtime_error("logistic line search did not converge");
        }

        solution[col] += step * delta;
        for (std::size_t row = 0; row < Rows; ++row) {
          scores[row] += step * delta * (col == 0 ? 1.0 : working(row, col - 1));
        }
      }

      std::array<double, Rows> errors{};
      for (std::size_t row = 0; row < Rows; ++row) {
        errors[row] = osmimath_detail::Sigmoid(scores[row]) - data(row, Target);
      }

      double violation = 0;
      for (std::size_t col = 0; col < Cols; ++col) {
        double gradient = 0;
        for (std::size_t row = 0; row < Rows; ++row) {
          gradient += (col == 0 ? 1.0 : working(row, col - 1)) * errors[row];
        }
        gradient = gradient / double(Rows) + l2[col] * solution[col];
        const double error = solution[col] == 0
          ? std::max(osmimath_detail::Abs(gradient) - l1[col], 0.0)
          : osmimath_detail::Abs(gradient + (solution[col] > 0 ? l1[col] : -l1[col]));
        violation = std::max(violation, error);
      }

      if (violation <= options.tolerance) {
        fit_result result{solution, iteration + 1};
        for (std::size_t col = 0; col < predictors; ++col) {
          result.coefficients[col + 1] /= working.scales[col];
          result.coefficients[0] -= result.coefficients[col + 1] * working.means[col];
        }

        return result;
      }
    }

    throw std::runtime_error("logitic regression did not converge within max_iterations");
  }
};

#endif

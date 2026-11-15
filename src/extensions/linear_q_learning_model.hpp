#ifndef OSMIMATH_LINEAR_Q_LEARNING_MODEL_HPP
#define OSMIMATH_LINEAR_Q_LEARNING_MODEL_HPP

#include "../matrix.hpp"
#include "q_learning_model.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>

template<std::size_t Features, std::size_t Actions>
class linear_q_learning_model {
  static_assert(Actions > 0, "Q-learning requires actions");
  static_assert(Features < std::numeric_limits<std::size_t>::max(), "Too many features");

public:
  using feature_type = std::array<float, Features>;
  using action_values_type = std::array<float, Actions>;
  using weights_type = AVX2Math::matrix<Features + 1, Actions>;

  static constexpr std::size_t features = Features;
  static constexpr std::size_t actions = Actions;

  explicit constexpr linear_q_learning_model(q_learning_options options = {}) : options_(options) {
    Validate(options);
  }

  [[nodiscard]] constexpr const q_learning_options& Options() const noexcept {
    return options_;
  }

  constexpr void SetOptions(q_learning_options options) {
    Validate(options);
    options_ = options;
  }

  [[nodiscard]] constexpr const weights_type& Weights() const noexcept {
    return weights_;
  }

  [[nodiscard]] constexpr action_values_type QValues(const feature_type& state) const {
    AVX2Math::matrix<1, Features + 1> input{};
    for (std::size_t feature = 0; feature < Features; ++feature) {
      if (!IsFinite(state[feature])) {
        throw std::invalid_argument("Q-learning features must be finite");
      }
      input(0, feature) = state[feature];
    }
    input(0, Features) = 1.0f;

    const auto result = AVX2Math::AVX2Dot(input, weights_);
    for (float value : result.elements) {
      if (!IsFinite(value)) {
        throw std::overflow_error("Q-learning prediction exceeds numeric range");
      }
    }

    return result.elements;
  }

  [[nodiscard]] constexpr float QValue(const feature_type& state, std::size_t action) const {
    ValidateAction(action);

    return QValues(state)[action];
  }

  [[nodiscard]] constexpr std::size_t Predict(const feature_type& state) const {
    return BestAction(QValues(state));
  }

  template<class Generator>
  [[nodiscard]] std::size_t SelectAction(const feature_type& state, Generator& generator) const {
    const auto values = QValues(state);

    if (options_.exploration_rate > 0 &&
      (options_.exploration_rate == 1 ||
        std::bernoulli_distribution(options_.exploration_rate)(generator))) {
      return std::uniform_int_distribution<std::size_t>(0, Actions - 1)(generator);
    }

    return BestAction(values);
  }

  constexpr void Update(const feature_type& state, std::size_t action, double reward,
    const feature_type& next_state, bool terminal = false) {
    ValidateAction(action);

    if (!IsFinite(reward)) {
      throw std::invalid_argument("Q-learning reward must be finite");
    }

    const double current = QValue(state, action);
    double target = reward;
    if (!terminal) {
      const auto next = QValues(next_state);
      target += options_.discount_factor * next[BestAction(next)];
    }

    if (!IsFinite(target)) {
      throw std::overflow_error("Q-learning target exceeds numeric range");
    }

    const double step = options_.learning_rate * (target - current);
    for (std::size_t feature = 0; feature <= Features; ++feature) {
      const double value = feature == Features ? 1.0 : state[feature];
      const double updated = double(weights_(feature, action)) + step * value;
      if (!(updated >= -std::numeric_limits<float>::max() &&
        updated <= std::numeric_limits<float>::max())) {
        throw std::overflow_error("Q-learning weights exceed numeric range");
      }
    }

    for (std::size_t feature = 0; feature <= Features; ++feature) {
      const double value = feature == Features ? 1.0 : state[feature];
      weights_(feature, action) = float(double(weights_(feature, action)) + step * value);
    }
  }

  constexpr void Reset() noexcept {
    weights_ = {};
  }

private:
  weights_type weights_{};
  q_learning_options options_{};

  static constexpr bool IsFinite(double value) noexcept {
    return std::isfinite(value);
  }

  static constexpr void Validate(const q_learning_options& options) {
    if (!(options.learning_rate > 0 && options.learning_rate <= 1) ||
      !(options.discount_factor >= 0 && options.discount_factor <= 1) ||
      !(options.exploration_rate >= 0 && options.exploration_rate <= 1)) {
      throw std::invalid_argument("Invalid Q-learning options");
    }
  }

  static constexpr void ValidateAction(std::size_t action) {
    if (action >= Actions) {
      throw std::out_of_range("Q-learning action is outside the model");
    }
  }

  static constexpr std::size_t BestAction(const action_values_type& values) noexcept {
    std::size_t best = 0;
    for (std::size_t action = 1; action < Actions; ++action) {
      if (values[action] > values[best]) {
        best = action;
      }
    }

    return best;
  }
};

#endif

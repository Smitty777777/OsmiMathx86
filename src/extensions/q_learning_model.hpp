#ifndef OSMIMATH_Q_LEARNING_MODEL_HPP
#define OSMIMATH_Q_LEARNING_MODEL_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>

struct q_learning_options {
  double learning_rate = 0.1;
  double discount_factor = 0.99;
  double exploration_rate = 0.1;
};

template<std::size_t States, std::size_t Actions>
class q_learning_model {
  static_assert(States > 0 && Actions > 0, "Q-learning requires states and actions");

public:
  using action_values_type = std::array<double, Actions>;
  using table_type = std::array<action_values_type, States>;

  static constexpr std::size_t states = States;
  static constexpr std::size_t actions = Actions;

  explicit constexpr q_learning_model(q_learning_options options = {}) : options_(options) {
    Validate(options);
  }

  [[nodiscard]] constexpr const q_learning_options& Options() const noexcept {
    return options_;
  }

  constexpr void SetOptions(q_learning_options options) {
    Validate(options);
    options_ = options;
  }

  [[nodiscard]] constexpr const table_type& QValues() const noexcept {
    return q_values_;
  }

  [[nodiscard]] constexpr double QValue(std::size_t state, std::size_t action) const {
    ValidateState(state);
    ValidateAction(action);

    return q_values_[state][action];
  }

  [[nodiscard]] constexpr std::size_t Predict(std::size_t state) const {
    ValidateState(state);

    std::size_t best = 0;
    for (std::size_t action = 1; action < Actions; ++action) {
      if (q_values_[state][action] > q_values_[state][best]) {
        best = action;
      }
    }

    return best;
  }

  template<class Generator>
  [[nodiscard]] std::size_t SelectAction(std::size_t state, Generator& generator) const {
    ValidateState(state);

    if (options_.exploration_rate > 0 &&
      (options_.exploration_rate == 1 ||
        std::bernoulli_distribution(options_.exploration_rate)(generator))) {
      return std::uniform_int_distribution<std::size_t>(0, Actions - 1)(generator);
    }

    return Predict(state);
  }

  constexpr void Update(std::size_t state, std::size_t action, double reward,
    std::size_t next_state, bool terminal = false) {
    ValidateState(state);
    ValidateAction(action);

    if (!IsFinite(reward)) {
      throw std::invalid_argument("Q-learning reward must be finite");
    }

    double target = reward;
    if (!terminal) {
      const std::size_t best = Predict(next_state);
      target += options_.discount_factor * q_values_[next_state][best];
    }

    if (!IsFinite(target)) {
      throw std::overflow_error("Q-learning target exceeds numeric range");
    }

    q_values_[state][action] =
      std::lerp(q_values_[state][action], target, options_.learning_rate);
  }

  constexpr void Reset() noexcept {
    q_values_ = {};
  }

private:
  table_type q_values_{};
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

  static constexpr void ValidateState(std::size_t state) {
    if (state >= States) {
      throw std::out_of_range("Q-learning state is outside the table");
    }
  }

  static constexpr void ValidateAction(std::size_t action) {
    if (action >= Actions) {
      throw std::out_of_range("Q-learning action is outside the table");
    }
  }
};

#endif

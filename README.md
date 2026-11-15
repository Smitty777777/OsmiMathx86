# OsmiMath

```cpp
#include "matrix.hpp"

constexpr AVX2Math::matrix<2, 2> a{{4, 7, 2, 6}};
constexpr auto transposed = AVX2Math::AVX2Transpose(a);
constexpr auto product = AVX2Math::AVX2Dot(a, transposed);
constexpr auto inverse = AVX2Math::AVX2Inverse(a);

static_assert(transposed[0, 1] == 2);
static_assert(product[0, 0] == 65);
static_assert(AVX2Math::AVX2Det(a) == 10);
static_assert(inverse.has_value());

auto identity = AVX2Math::matrix<4, 4>::identity();
```

```cpp
#include "extensions/linear_model.hpp"

#include <cstddef>
#include <iostream>

int main() {
  AVX2Math::matrix<10, 2> data{};
  for (std::size_t row = 0; row < 10; ++row) {
    data(row, 0) = float(row);
    data(row, 1) = 1.0f + 2.0f * data(row, 0);
  }

  const linear_model<10, 2, 1> model(data);
  const linear_model<10, 2, 1> ridge(data, {
    .penalty = regularization::ridge,
    .alpha = 0.1
  });
  const linear_model<10, 2, 1> lasso(data, {
    .penalty = regularization::lasso,
    .alpha = 0.1
  });

  std::cout << model.Predict({10.0f}) << '\n';
  std::cout << ridge.Predict({10.0f}) << '\n';
  std::cout << lasso.Predict({10.0f}) << '\n';
  std::cout << model.Intercept() << ' ' << model.Coefficients()[1] << '\n';

  const AVX2Math::matrix<3, 1> inputs{{10.0f, 11.0f, 12.0f}};
  for (double prediction : model.Predict(inputs)) {
    std::cout << prediction << '\n';
  }
}
```

```cpp
#include "extensions/logistic_model.hpp"

#include <cstddef>
#include <iostream>

int main() {
  AVX2Math::matrix<20, 2> data{};
  for (std::size_t row = 0; row < 20; ++row) {
    data(row, 0) = row < 10 ? -1.0f : 1.0f;
    data(row, 1) = row < 10 ? float(row < 2) : float(row < 18);
  }

  const logistic_model<20, 2, 1> model(data, {
    .penalty = regularization::ridge,
    .alpha = 0.1
  });

  std::cout << model.PredictProbability({1.0f}) << '\n';
  std::cout << model.Predict({1.0f}, 0.5) << '\n';

  const AVX2Math::matrix<2, 1> inputs{{-1.0f, 1.0f}};
  const auto probabilities = model.PredictProbability(inputs);
  const auto labels = model.Predict(inputs);
  for (std::size_t row = 0; row < inputs.rows; ++row) {
    std::cout << probabilities[row] << ' ' << labels[row] << '\n';
  }
}
```

```cpp
#include "extensions/lightgbm_model.hpp"

#include <cstddef>
#include <iostream>

int main() {
  AVX2Math::matrix<20, 2> data{};
  for (std::size_t row = 0; row < 20; ++row) {
    data(row, 0) = float(row);
    data(row, 1) = row < 10 ? 0.0f : 10.0f;
  }

  const lightgbm_model<20, 2, 1> regressor(data, {
    .iterations = 100,
    .learning_rate = 0.1,
    .num_leaves = 2,
    .max_bins = 20
  });

  std::cout << regressor.Predict({15.0f}) << '\n';
  std::cout << regressor.TreeCount() << '\n';

  const AVX2Math::matrix<2, 1> inputs{{5.0f, 15.0f}};
  for (double prediction : regressor.Predict(inputs)) {
    std::cout << prediction << '\n';
  }

  auto binary_data = data;
  for (std::size_t row = 0; row < 20; ++row) {
    binary_data(row, 1) = float(row >= 10);
  }

  const lightgbm_model<20, 2, 1> classifier(binary_data, {
    .objective = boosting_objective::binary_logloss,
    .iterations = 100,
    .learning_rate = 0.1,
    .num_leaves = 2,
    .max_bins = 20,
    .seed = 42
  });

  std::cout << classifier.Predict({15.0f}) << '\n';
  std::cout << classifier.PredictClass({15.0f}) << '\n';
}
```

```cpp
#include "extensions/q_learning_model.hpp"

q_learning_model<4, 2> agent({
  .learning_rate = 0.25,
  .discount_factor = 0.9,
  .exploration_rate = 0.2
});
std::mt19937 generator(42);

for (std::size_t episode = 0; episode < 2000; ++episode) {
  std::size_t state = 0;

  for (std::size_t step = 0; step < 64 && state != 3; ++step) {
    const auto action = agent.SelectAction(state, generator);
    const auto next = action == 1 ? state + 1 : (state == 0 ? 0 : state - 1);
    const bool terminal = next == 3;

    agent.Update(state, action, terminal ? 1.0 : 0.0, next, terminal);
    state = next;
  }
}

const auto action = agent.Predict(0);
const auto value = agent.QValue(0, action);
```

```cpp
#include "extensions/linear_q_learning_model.hpp"

linear_q_learning_model<3, 4> agent({
  .learning_rate = 0.01,
  .discount_factor = 0.99,
  .exploration_rate = 0.1
});
std::mt19937 generator(42);
std::array<float, 3> state{0.2f, -0.4f, 0.8f};

const auto action = agent.SelectAction(state, generator);
std::array<float, 3> next_state{0.3f, -0.2f, 0.7f};
agent.Update(state, action, 1.0, next_state);

const auto values = agent.QValues(state);
const auto best_action = agent.Predict(state);
```

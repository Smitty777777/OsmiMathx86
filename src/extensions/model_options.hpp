#ifndef OSMIMATH_MODEL_OPTIONS_HPP
#define OSMIMATH_MODEL_OPTIONS_HPP

#include <cstddef>

enum class regularization {
  none,
  ridge,
  lasso
};

struct model_options {
  regularization penalty = regularization::none;
  double alpha = 1.0;
  double tolerance = 1e-8;
  std::size_t max_iterations = 10000;
};

#endif

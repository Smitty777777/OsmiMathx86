#ifndef OSMIMATH_LIGHTGBM_MODEL_HPP
#define OSMIMATH_LIGHTGBM_MODEL_HPP

#include "model_detail.hpp"

#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

enum class boosting_objective {
  squared_error,
  binary_logloss
};

struct lightgbm_options {
  boosting_objective objective = boosting_objective::squared_error;
  std::size_t iterations = 100;
  double learning_rate = 0.1;

  std::size_t num_leaves = 31;
  std::size_t max_depth = 0;
  std::size_t max_bins = 63;
  std::size_t min_data_in_leaf = 2;
  double min_sum_hessian_in_leaf = 1e-3;

  double lambda_l1 = 0;
  double lambda_l2 = 1;
  double min_gain_to_split = 0;
  double max_delta_step = 0;

  bool enable_goss = false;
  double top_rate = 0.2;
  double other_rate = 0.1;

  bool enable_bundle = true;
  std::uint32_t seed = 0;
};

template<std::size_t Rows, std::size_t Cols, std::size_t Target>
class lightgbm_model {
  static_assert(Rows >= 10 && Target < Cols, "target outside column range or insufficient data");

public:
  using data_type = AVX2Math::matrix<Rows, Cols>;

  static constexpr std::size_t predictors = Cols - 1;
  static constexpr std::size_t no_node = std::numeric_limits<std::size_t>::max();

  struct tree_node {
    std::size_t feature = predictors;
    float threshold = 0;
    double value = 0;
    double gain = 0;
    std::size_t left = no_node;
    std::size_t right = no_node;
    std::size_t depth = 0;

    [[nodiscard]] bool IsLeaf() const noexcept {
      return left == no_node;
    }
  };

  struct tree {
    std::vector<tree_node> nodes;

    [[nodiscard]] double Predict(const std::array<float, predictors>& values) const noexcept {
      std::size_t index = 0;
      while (!nodes[index].IsLeaf()) {
        const auto& node = nodes[index];
        index = values[node.feature] <= node.threshold ? node.left : node.right;
      }

      return nodes[index].value;
    }
  };

  explicit lightgbm_model(const data_type& data, lightgbm_options options = {}) {
    Fit(data, options);
  }

  void Fit(const data_type& data) {
    Fit(data, options_);
  }

  void Fit(const data_type& data, lightgbm_options options) {
    Validate(options);
    for (float value : data.elements) {
      if (!osmimath_detail::IsFinite(value)) {
        throw std::invalid_argument("boosting data must be finite");
      }
    }

    const double mean = osmimath_detail::Mean(data, Target);
    if (options.objective == boosting_objective::binary_logloss) {
      for (std::size_t row = 0; row < Rows; ++row) {
        if (data(row, Target) != 0 && data(row, Target) != 1) {
          throw std::invalid_argument("Binary boosting requires labels 0 and 1");
        }
      }
      if (mean == 0 || mean == 1) {
        throw std::invalid_argument("binary boosting requires both classes");
      }
    }

    const double base = options.objective == boosting_objective::squared_error
      ? mean
      : std::log(mean) - std::log1p(-mean);
    auto binned = Bin(data, options);
    std::array<double, Rows> scores{}, gradients{}, hessians{}, weights{};
    scores.fill(base);
    std::mt19937 generator(options.seed);
    std::vector<tree> trees;

    for (std::size_t iteration = 0; iteration < options.iterations; ++iteration) {
      for (std::size_t row = 0; row < Rows; ++row) {
        const double prediction = options.objective == boosting_objective::squared_error
          ? scores[row]
          : osmimath_detail::Sigmoid(scores[row]);
        gradients[row] = prediction - data(row, Target);
        hessians[row] = options.objective == boosting_objective::squared_error
          ? 1.0
          : std::max(prediction * (1 - prediction), 1e-12);
      }

      const auto sample = Sample(gradients, weights, options, generator);
      auto next = BuildTree(data, binned, sample, gradients, hessians, weights, options);
      if (next.nodes.size() == 1 && osmimath_detail::Abs(next.nodes[0].value) <= 1e-15) {
        break;
      }

      for (std::size_t row = 0; row < Rows; ++row) {
        std::array<float, predictors> values{};
        for (std::size_t col = 0; col < predictors; ++col) {
          values[col] = data(row, SourceColumn(col));
        }
        scores[row] += options.learning_rate * next.Predict(values);
        if (!osmimath_detail::IsFinite(scores[row])) {
          throw std::runtime_error("boosting exceeded numeric range");
        }
      }
      trees.push_back(std::move(next));
    }

    trees_ = std::move(trees);
    base_score_ = base;
    bundles_ = binned.bundles.size();
    options_ = options;
  }

  [[nodiscard]] const lightgbm_options& Options() const noexcept {
    return options_;
  }

  [[nodiscard]] const std::vector<tree>& Trees() const noexcept {
    return trees_;
  }

  [[nodiscard]] std::size_t TreeCount() const noexcept {
    return trees_.size();
  }

  [[nodiscard]] std::size_t NumBundles() const noexcept {
    return bundles_;
  }

  [[nodiscard]] double BaseScore() const noexcept {
    return base_score_;
  }

  [[nodiscard]] double PredictRaw(const std::array<float, predictors>& values) const {
    for (float value : values) {
      if (!osmimath_detail::IsFinite(value)) {
        throw std::invalid_argument("predictors must be finite");
      }
    }

    double result = base_score_;
    for (const auto& fitted_tree : trees_) {
      result += options_.learning_rate * fitted_tree.Predict(values);
    }

    return result;
  }

  [[nodiscard]] double Predict(const std::array<float, predictors>& values) const {
    const double raw = PredictRaw(values);
    return options_.objective == boosting_objective::squared_error
      ? raw
      : osmimath_detail::Sigmoid(raw);
  }

  [[nodiscard]] bool PredictClass(
    const std::array<float, predictors>& values, double threshold = 0.5) const {
    if (options_.objective != boosting_objective::binary_logloss) {
      throw std::logic_error("classification requires binary_logloss");
    }
    if (!(threshold >= 0 && threshold <= 1)) {
      throw std::invalid_argument("threshold must be in [0, 1]");
    }

    return Predict(values) >= threshold;
  }

  template<std::size_t Count>
  [[nodiscard]] std::array<double, Count> Predict(
    const AVX2Math::matrix<Count, predictors>& data) const {
    std::array<double, Count> result{};
    for (std::size_t row = 0; row < Count; ++row) {
      std::array<float, predictors> values{};
      for (std::size_t col = 0; col < predictors; ++col) {
        values[col] = data(row, col);
      }
      result[row] = Predict(values);
    }

    return result;
  }

  [[nodiscard]] std::array<double, predictors> FeatureImportances() const noexcept {
    std::array<double, predictors> result{};
    double total = 0;
    for (const auto& fitted_tree : trees_) {
      for (const auto& node : fitted_tree.nodes) {
        if (!node.IsLeaf()) {
          result[node.feature] += node.gain;
          total += node.gain;
        }
      }
    }

    if (total > 0) {
      for (double& value : result) {
        value /= total;
      }
    }

    return result;
  }

private:
  struct statistics {
    double gradient = 0;
    double hessian = 0;
    std::size_t count = 0;

    statistics& operator+=(const statistics& other) noexcept {
      gradient += other.gradient;
      hessian += other.hessian;
      count += other.count;
      return *this;
    }

    statistics& operator-=(const statistics& other) noexcept {
      gradient -= other.gradient;
      hessian -= other.hessian;
      count -= other.count;
      return *this;
    }
  };

  struct bundle {
    std::vector<std::size_t> features;
    std::vector<std::size_t> codes;
    std::size_t bins = 1;
  };

  struct bin_data {
    std::array<std::vector<float>, predictors> cuts;
    std::array<std::size_t, predictors> defaults{};
    std::array<std::size_t, predictors> offsets{};
    std::vector<bundle> bundles;
  };

  struct split {
    std::size_t feature = predictors;
    float threshold = 0;
    double gain = 0;
    statistics left{}, right{};
  };

  using histogram = std::vector<std::vector<statistics>>;

  struct leaf {
    std::size_t node = 0;
    std::vector<std::size_t> rows;
    statistics total{};
    histogram bins;
    split best{};
  };

  std::vector<tree> trees_;
  lightgbm_options options_{};
  double base_score_ = 0;
  std::size_t bundles_ = 0;

  static constexpr std::size_t SourceColumn(std::size_t predictor) noexcept {
    return predictor < Target ? predictor : predictor + 1;
  }

  static void Validate(const lightgbm_options& o) {
    if ((o.objective != boosting_objective::squared_error &&
          o.objective != boosting_objective::binary_logloss) ||
      o.iterations == 0 || !(o.learning_rate > 0 && o.learning_rate <= 1) || o.num_leaves == 0 ||
      o.max_bins < 2 || o.max_bins > 65535 || o.min_data_in_leaf == 0 ||
      !osmimath_detail::IsFinite(o.min_sum_hessian_in_leaf) || o.min_sum_hessian_in_leaf < 0 ||
      !osmimath_detail::IsFinite(o.lambda_l1) || o.lambda_l1 < 0 ||
      !osmimath_detail::IsFinite(o.lambda_l2) || o.lambda_l2 < 0 ||
      !osmimath_detail::IsFinite(o.min_gain_to_split) || o.min_gain_to_split < 0 ||
      !osmimath_detail::IsFinite(o.max_delta_step) || o.max_delta_step < 0 ||
      !(o.top_rate > 0 && o.other_rate > 0 && o.top_rate + o.other_rate <= 1)) {
      throw std::invalid_argument("bad histogram boosting options");
    }
  }

  static bin_data Bin(const data_type& data, const lightgbm_options& options) {
    bin_data result;
    std::array<std::vector<std::size_t>, predictors> bins;
    std::array<std::size_t, predictors> active{}, order{};

    for (std::size_t feature = 0; feature < predictors; ++feature) {
      std::vector<float> values(Rows);
      for (std::size_t row = 0; row < Rows; ++row) {
        values[row] = data(row, SourceColumn(feature));
      }
      std::sort(values.begin(), values.end());

      auto& cuts = result.cuts[feature];
      const std::size_t count = std::min(options.max_bins, Rows);
      for (std::size_t bin = 1; bin < count; ++bin) {
        const std::size_t position = (Rows / count) * bin + ((Rows % count) * bin) / count;
        const float cut = values[position - 1];
        if (cut < values.back() && (cuts.empty() || cut > cuts.back())) {
          cuts.push_back(cut);
        }
      }

      result.defaults[feature] =
        std::size_t(std::lower_bound(cuts.begin(), cuts.end(), 0.0f) - cuts.begin());
      bins[feature].resize(Rows);
      for (std::size_t row = 0; row < Rows; ++row) {
        bins[feature][row] =
          std::size_t(std::lower_bound(cuts.begin(), cuts.end(), data(row, SourceColumn(feature))) -
            cuts.begin());
        active[feature] += bins[feature][row] != result.defaults[feature];
      }
      order[feature] = feature;
    }

    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return active[a] > active[b];
    });

    for (std::size_t feature : order) {
      std::size_t destination = result.bundles.size();
      if (options.enable_bundle) {
        for (std::size_t index = 0; index < result.bundles.size(); ++index) {
          bool conflict = false;
          for (std::size_t row = 0; row < Rows; ++row) {
            if (bins[feature][row] != result.defaults[feature] &&
              result.bundles[index].codes[row] != 0) {
              conflict = true;
              break;
            }
          }
          if (!conflict) {
            destination = index;
            break;
          }
        }
      }
      if (destination == result.bundles.size()) {
        result.bundles.push_back({{}, std::vector<std::size_t>(Rows), 1});
      }

      auto& target = result.bundles[destination];
      result.offsets[feature] = target.bins;
      target.features.push_back(feature);
      target.bins += result.cuts[feature].size();
      for (std::size_t row = 0; row < Rows; ++row) {
        const std::size_t bin = bins[feature][row], zero = result.defaults[feature];
        if (bin != zero) {
          target.codes[row] = result.offsets[feature] + bin - (bin > zero ? 1 : 0);
        }
      }
    }

    return result;
  }

  static std::vector<std::size_t> Sample(const std::array<double, Rows>& gradients,
    std::array<double, Rows>& weights, const lightgbm_options& options, std::mt19937& generator) {
    std::vector<std::size_t> rows(Rows);
    std::iota(rows.begin(), rows.end(), std::size_t(0));
    weights.fill(1.0);
    if (!options.enable_goss) {
      return rows;
    }

    std::stable_sort(rows.begin(), rows.end(), [&](std::size_t a, std::size_t b) {
      return osmimath_detail::Abs(gradients[a]) > osmimath_detail::Abs(gradients[b]);
    });

    const std::size_t top =
      std::clamp(std::size_t(options.top_rate * double(Rows)), std::size_t(1), Rows - 1);
    const std::size_t other =
      std::clamp(std::size_t(options.other_rate * double(Rows)), std::size_t(1), Rows - top);
    std::shuffle(rows.begin() + top, rows.end(), generator);
    rows.resize(top + other);
    weights.fill(0.0);
    for (std::size_t index = 0; index < rows.size(); ++index) {
      weights[rows[index]] = index < top ? 1.0 : double(Rows - top) / double(other);
    }

    return rows;
  }

  static histogram MakeHistogram(const bin_data& binned, const std::vector<std::size_t>& rows,
    const std::array<double, Rows>& gradients, const std::array<double, Rows>& hessians,
    const std::array<double, Rows>& weights) {
    histogram result(binned.bundles.size());
    for (std::size_t index = 0; index < binned.bundles.size(); ++index) {
      const auto& group = binned.bundles[index];
      result[index].resize(group.bins);
      for (std::size_t row : rows) {
        result[index][group.codes[row]] +=
          {gradients[row] * weights[row], hessians[row] * weights[row], 1};
      }
    }

    return result;
  }

  static double LeafValue(const statistics& total, const lightgbm_options& options) {
    const double denominator = total.hessian + options.lambda_l2;
    double result = denominator > 0
      ? -osmimath_detail::SoftThreshold(total.gradient, options.lambda_l1) / denominator
      : 0;
    if (options.max_delta_step > 0) {
      result = std::clamp(result, -options.max_delta_step, options.max_delta_step);
    }

    return result;
  }

  static double Score(const statistics& total, const lightgbm_options& options) {
    const double value = LeafValue(total, options);
    return -(2 * total.gradient * value + (total.hessian + options.lambda_l2) * value * value +
      2 * (options.lambda_l1 * osmimath_detail::Abs(value)));
  }

  static split BestSplit(
    const leaf& current, const bin_data& binned, const lightgbm_options& options) {
    split best{};
    best.gain = options.min_gain_to_split;
    const double parent_score = Score(current.total, options);

    for (std::size_t group = 0; group < binned.bundles.size(); ++group) {
      for (std::size_t feature : binned.bundles[group].features) {
        const std::size_t zero = binned.defaults[feature];
        const std::size_t count = binned.cuts[feature].size();
        const std::size_t offset = binned.offsets[feature];
        const auto& bins = current.bins[group];
        statistics default_bin = current.total;
        for (std::size_t bin = 0; bin <= count; ++bin) {
          if (bin == zero) {
            continue;
          }
          default_bin -= bins[offset + bin - (bin > zero ? 1 : 0)];
        }

        statistics left{};
        for (std::size_t bin = 0; bin < count; ++bin) {
          left += bin == zero ? default_bin : bins[offset + bin - (bin > zero ? 1 : 0)];
          statistics right = current.total;
          right -= left;
          if (left.count < options.min_data_in_leaf || right.count < options.min_data_in_leaf ||
            left.hessian < options.min_sum_hessian_in_leaf ||
            right.hessian < options.min_sum_hessian_in_leaf) {
            continue;
          }

          const double gain = 0.5 * (Score(left, options) + Score(right, options) - parent_score);
          if (gain > best.gain) {
            best = {feature, binned.cuts[feature][bin], gain, left, right};
          }
        }
      }
    }

    return best;
  }

  static tree BuildTree(const data_type& data, const bin_data& binned,
    const std::vector<std::size_t>& sample, const std::array<double, Rows>& gradients,
    const std::array<double, Rows>& hessians, const std::array<double, Rows>& weights,
    const lightgbm_options& options) {
    tree result;
    result.nodes.push_back({});

    leaf root;
    root.rows = sample;
    for (std::size_t row : sample) {
      root.total += {gradients[row] * weights[row], hessians[row] * weights[row], 1};
    }
    root.bins = MakeHistogram(binned, root.rows, gradients, hessians, weights);
    root.best = BestSplit(root, binned, options);
    result.nodes[0].value = LeafValue(root.total, options);

    std::vector<leaf> leaves;
    leaves.push_back(std::move(root));

    while (leaves.size() < options.num_leaves) {
      std::size_t best = leaves.size();
      double gain = options.min_gain_to_split;
      for (std::size_t index = 0; index < leaves.size(); ++index) {
        const auto& candidate = leaves[index];
        if (options.max_depth > 0 && result.nodes[candidate.node].depth >= options.max_depth) {
          continue;
        }
        if (candidate.best.feature < predictors && candidate.best.gain > gain) {
          best = index;
          gain = candidate.best.gain;
        }
      }
      if (best == leaves.size()) {
        break;
      }

      leaf parent = std::move(leaves[best]);
      const auto choice = parent.best;
      leaf left, right;
      left.total = choice.left;
      right.total = choice.right;
      for (std::size_t row : parent.rows) {
        (data(row, SourceColumn(choice.feature)) <= choice.threshold ? left.rows : right.rows)
          .push_back(row);
      }

      leaf& smaller = left.rows.size() <= right.rows.size() ? left : right;
      leaf& larger = left.rows.size() <= right.rows.size() ? right : left;
      smaller.bins = MakeHistogram(binned, smaller.rows, gradients, hessians, weights);
      larger.bins = std::move(parent.bins);
      for (std::size_t group = 0; group < larger.bins.size(); ++group) {
        for (std::size_t bin = 0; bin < larger.bins[group].size(); ++bin) {
          larger.bins[group][bin] -= smaller.bins[group][bin];
        }
      }

      const std::size_t depth = result.nodes[parent.node].depth + 1;
      left.node = result.nodes.size();
      result.nodes.push_back(
        {predictors, 0, LeafValue(left.total, options), 0, no_node, no_node, depth});
      right.node = result.nodes.size();
      result.nodes.push_back(
        {predictors, 0, LeafValue(right.total, options), 0, no_node, no_node, depth});

      auto& node = result.nodes[parent.node];
      node.feature = choice.feature;
      node.threshold = choice.threshold;
      node.gain = choice.gain;
      node.left = left.node;
      node.right = right.node;

      left.best = BestSplit(left, binned, options);
      right.best = BestSplit(right, binned, options);
      leaves[best] = std::move(left);
      leaves.push_back(std::move(right));
    }

    return result;
  }
};

#endif

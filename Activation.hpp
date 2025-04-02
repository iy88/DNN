#ifndef _ACTIVATION_
#define _ACTIVATION_
#include "Function.hpp"
#include "common.h"
class Activation : Function {
 public:
  Activation() = default;
  void bp(ld grad) {}
  void initializeWeights(size_t outputLength) {}
  vector<ld> getWeights() { return vector<ld>(); }
  void clearGrad() {}
};

class Sigmoid : public Activation {
 public:
  Sigmoid() = default;
  ld exec(vector<ld> &inputs) { return 1.0 / (1 + exp(-inputs[0])); }
  vector<ld> grad(vector<ld> &inputs) {
    return {exec(inputs) * (1 - exec(inputs))};
  }
  Sigmoid *newInstance() { return new Sigmoid; }
};

class Sgn : public Activation {
 public:
  Sgn() = default;
  ld exec(vector<ld> &inputs) { return 2.0 / (1 + exp(-inputs[0])) - 1; }
  vector<ld> grad(vector<ld> &inputs) {
    return {2 * exp(-inputs[0]) /
            (2 * exp(-inputs[0]) + (exp(-inputs[0]) * exp(-inputs[0])) + 1)};
  }
};

class LeakyReLU : public Activation {
 public:
  LeakyReLU() = default;
  ld exec(vector<ld> &inputs) {
    return inputs[0] > 0 ? inputs[0] : 1 / 5.5 * inputs[0];
  }
  vector<ld> grad(vector<ld> &inputs) {
    return {ld(inputs[0] >= 0 ? 1 : 1 / 5.5)};
  }
};

class ReLU : public Activation {
 public:
  ReLU() = default;
  ld exec(vector<ld> &inputs) { return inputs[0] > 0 ? inputs[0] : 0; }
  vector<ld> grad(vector<ld> &inputs) { return {ld(inputs[0] >= 0 ? 1 : 0)}; }
};
#endif
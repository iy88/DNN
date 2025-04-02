#ifndef _LINEAR_
#define _LINEAR_
#include "Function.hpp"
#include "common.h"
class Linear : public Function {
 public:
  bool isWeighted() { return true; }
  vector<ld> W;
  vector<ld> grads;
  vector<ld> _inputs;
  Linear(size_t inputLength) {
    // for (size_t i = 0; i <= inputLength; i++) {
    //   W.push_back(1+i);
    // }
    W.resize(inputLength + 1);  //  plus 1 for bias
  }
  void initializeWeights(size_t outputLength) {
    ld bound_weight = 1.0 / sqrt(W.size());
    for (auto &it : W) {
      ld rand_num = (ld)rand() / RAND_MAX * 2 * bound_weight - bound_weight;
      it = rand_num;
    }
    // // W[W.size() - 1] = 0.0;  // zero for bias
  }
  ld exec(vector<ld> &inputs) {
    _inputs = inputs;
    _inputs.push_back(1);  // for bias, extra input unit 1
    ld res = 0;
    for (size_t i = 0; i < W.size(); i++) {
      res += _inputs[i] * W[i];
    }
    return res;
  }
  vector<ld> grad(vector<ld> &inputs) {
    vector<ld> d;
    for (size_t i = 0; i < inputs.size(); i++) {
      d.push_back(W[i]);
    }
    return d;
  }
  void bp(ld grad) {
    if (!grads.size()) grads = vector<ld>(W.size(), 0);
    for (size_t i = 0; i < W.size(); i++) {
      grads[i] += grad * _inputs[i];
    }
  }
  void step(ld lr) {
    for (size_t i = 0; i < W.size(); i++) {
      W[i] -= grads[i] * lr;
    }
  }
  vector<ld> getWeights() { return W; }
  void clearGrad() {
    for (auto &it : grads) it = 0;
  }
};
#endif
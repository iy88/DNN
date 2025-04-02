#ifndef _NEURON_
#define _NEURON_
#include "ComputationalGraph.hpp"
#include "Linear.hpp"
#include "common.h"
class Neuron : public ComputeNode {
 public:
  Neuron(size_t inputLength, size_t id)
      : ComputeNode(inputLength, new Linear(inputLength), id) {};
  Neuron(size_t inputLength, Function *fp, size_t id)
      : ComputeNode(inputLength, fp, id) {};
  Neuron(size_t id) : ComputeNode(0, id) {}
  void input(ld v) {  // for input node only
    value = v;
  }
  void bp();
  void step(ld lr);
  void clearGrad() {
    grad = 0;
    if (fp) fp->clearGrad();
  }
};

void Neuron::bp() {
  if (fp != nullptr) fp->bp(grad);
  /*
   * Neuron's grad is a temp var. depend on each input, accumulate the grad in
   * the detailed function instead
   */
  grad = 0;
}

void Neuron::step(ld lr) {
  if (fp) fp->step(lr);
}
#endif
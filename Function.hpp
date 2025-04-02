#ifndef _FUNCTION_
#define _FUNCTION_
#include "common.h"
class Function {
 public:
  virtual bool isWeighted(){return false;}
  virtual ld exec(vector<ld> &inputs) = 0;
  virtual vector<ld> grad(vector<ld> &inputs) = 0;
  virtual ~Function() = default;
  virtual void bp(ld grad) {}
  virtual void step(ld lr) {}
  virtual void initializeWeights(size_t outputLength) {}
  virtual vector<ld> getWeights() = 0;
  virtual void clearGrad() {}
};
#endif
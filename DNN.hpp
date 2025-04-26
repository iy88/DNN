#ifndef _DNN_
#define _DNN_
#include "Activation.hpp"
#include "ComputationalGraph.hpp"
#include "Function.hpp"
#include "Linear.hpp"
#include "Neuron.hpp"
#include "common.h"

class DNN : public ComputationalGraph {
 public:
  DNN() { srand(time(nullptr)); };
  vector<vector<Neuron *>> Layers;
  void addLayer(size_t outputLength);
  Neuron *addNeuron(size_t inputLength);
  Neuron *addNeuron();
  Neuron *addNeuron(Activation *fp);
  void bp(vector<ld> &labels);
  vector<ld> eval(vector<ld> &inputs);
  void training(ld lr, size_t epoch, vector<vector<ld>> &inputs,
                vector<vector<ld>> &labels);
  template <typename ActivationType>
  void addLayer();  // activation layer
  void step(ld lr);
  vector<vector<vector<ld>>> getWeights();
  vector<ld> MSELoss(vector<ld> labels, vector<ld> outputs, size_t n);
  void manual_seed(unsigned int seed = 42) { srand(seed); }
};

Neuron *DNN::addNeuron() {
  Neuron *nd = new Neuron(Nodes.size());
  Nodes.push_back(nd);
  return nd;
}

Neuron *DNN::addNeuron(Activation *fp) {
  Neuron *nd = new Neuron(1, (Function *)fp, Nodes.size());
  Nodes.push_back(nd);
  return nd;
}

Neuron *DNN::addNeuron(size_t inputLength) {
  Neuron *nd = new Neuron(inputLength, Nodes.size());
  Nodes.push_back(nd);
  return nd;
}

void DNN::addLayer(size_t outputLength) {
  size_t inputLength = Layers.size() > 0 ? Layers[Layers.size() - 1].size() : 0;
  vector<Neuron *> Layer;
  vector<ComputeNode *> lastLayer;
  if (inputLength)
    lastLayer = vector<ComputeNode *>(Layers[Layers.size() - 1].begin(),
                                      Layers[Layers.size() - 1].end());
  for (size_t i = 0; i < outputLength; i++) {
    Neuron *nd = inputLength != 0 ? addNeuron(inputLength) : addNeuron();
    if (nd->fp != nullptr) nd->fp->initializeWeights(outputLength);
    Layer.push_back(nd);
    if (inputLength) bindNode(lastLayer, nd);
  }
  Layers.push_back(Layer);
}

template <typename ActivationType>
void DNN::addLayer() {
  vector<Neuron *> Layer;
  vector<ComputeNode *> lastLayer = vector<ComputeNode *>(
      Layers[Layers.size() - 1].begin(), Layers[Layers.size() - 1].end());
  for (size_t i = 0; i < lastLayer.size(); i++) {
    Neuron *nd = addNeuron(new ActivationType);
    Layer.push_back(nd);
    bindNode({lastLayer[i]}, nd);
  }
  Layers.push_back(Layer);
}

void DNN::bp(vector<ld> &outputGrad) {
  grad(outputGrad);
  Neuron *n;
  for (auto nd : Nodes) {
    n = (Neuron *)nd;
    n->bp();
  }
}

void DNN::step(ld lr) {
  Neuron *n;
  for (auto nd : Nodes) {
    n = (Neuron *)nd;
    n->step(lr);
  }
}

vector<ld> DNN::eval(vector<ld> &inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    Layers[0][i]->input(inputs[i]);
  }
  // cout << "Forward::" << endl;  // use to print forward output of each node
  return compute();
}

vector<ld> DNN::MSELoss(vector<ld> labels, vector<ld> outputs, size_t n) {
  vector<ld> grad(outputs.size(), 0);
  for (size_t j = 0; j < labels.size(); j++) {
    // 1/n Σ(y_pd-y_tg)^2  --- standard MSE
    grad[j] = 2 * (outputs[j] - labels[j]) / n;
    // 1/2n Σ(y_tg-y_pd)^2  --- use extra 1/2 eliminate the "2" factor
    // grad[j] = -(labels[j] - outputs[j]) / n;
  }
  return grad;
}

void DNN::training(ld lr, size_t epoch, vector<vector<ld>> &inputs,
                   vector<vector<ld>> &labels) {
  for (size_t i = 0; i < epoch; i++) {
    clearGrad();
    for (size_t t = 0; t < inputs.size(); t++) {
      // clearGrad();
      vector<ld> res = eval(inputs[t]);
      vector<ld> lossGrad(MSELoss(labels[t], res, inputs.size()));
      bp(lossGrad);
      // step(lr);
    }
    step(lr);
  }
}

vector<vector<vector<ld>>> DNN::getWeights() {
  vector<vector<vector<ld>>> W;
  for (auto &layer : Layers) {
    vector<vector<ld>> w;
    for (auto &node : layer) {
      if (!node->fp) continue;
      if (node->fp->isWeighted()) w.push_back(node->fp->getWeights());
    }
    W.push_back(w);
  }
  return W;
}
#endif
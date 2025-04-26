#ifndef _COMPUTATIONALGRAPH_
#define _COMPUTATIONALGRAPH_
#include "Function.hpp"
#include "common.h"
class ComputeNode {
 public:
  size_t inputLength = 0;
  ld value = 0;
  vector<ld> pfpx;
  vector<ComputeNode *> nextNodes;
  vector<ComputeNode *> operands;
  Function *fp = nullptr;
  size_t id = 999;
  ld grad = 0;
  ComputeNode(size_t _inputLength, Function *_fp, size_t _id)
      : inputLength(_inputLength), fp(_fp), id(_id) {}
  ComputeNode(ld val, size_t _id) : value(val), id(_id) {};
  virtual ~ComputeNode() {
    if (fp != nullptr) delete fp;
  };
  virtual ld compute();
  virtual void clearGrad() {};
};

ld ComputeNode::compute() {
  if (inputLength == 0) return value;
  vector<ld> inputs;
  for (size_t i = 0; i < inputLength; ++i) {
    inputs.push_back(operands[i]->value);
  }
  value = fp->exec(inputs);
  pfpx = fp->grad(inputs);
  return value;
}

class ComputationalGraph {
 public:
  vector<ComputeNode *> Nodes;
  vector<ComputeNode *> outNodes;
  vector<ComputeNode *> inNodes;
  vector<ComputeNode *> topo_forward;
  vector<ComputeNode *> topo_backward;
  ComputationalGraph() = default;
  ComputeNode *addNode(size_t inputLength, Function *fp);
  ComputeNode *addNode(ld value);
  void bindNode(vector<ComputeNode *> ops, ComputeNode *ComputeNodePtr);
  vector<ld> compute();
  vector<ld> grad();
  void grad(vector<ld> &outputGrad);
  void clearGrad() {
    for (auto nd : Nodes) nd->clearGrad();
  }
  virtual ~ComputationalGraph() {
    for (auto &it : Nodes) delete it;
  }
};

ComputeNode *ComputationalGraph::addNode(size_t inputLength, Function *fp) {
  ComputeNode *nd = new ComputeNode(inputLength, fp, Nodes.size());
  Nodes.push_back(nd);
  return nd;
}

ComputeNode *ComputationalGraph::addNode(ld value) {
  ComputeNode *nd = new ComputeNode(value, Nodes.size());
  Nodes.push_back(nd);
  return nd;
}

void ComputationalGraph::bindNode(vector<ComputeNode *> ops,
                                  ComputeNode *ComputeNodePtr) {
  for (auto nd : ops) {
    ComputeNodePtr->operands.push_back(nd);
    nd->nextNodes.push_back(ComputeNodePtr);
  }
}

vector<ld> ComputationalGraph::compute() {
  if (!topo_forward.size()) {
    map<ComputeNode *, ll> indegree;
    bool flag = false;
    for (auto nd : Nodes) {
      for (auto nnd : nd->nextNodes) {
        indegree[nnd]++;
      }
    }
    if (outNodes.size() != 0) flag = true;
    queue<ComputeNode *> q;
    if (inNodes.size() == 0) {
      for (auto nd : Nodes)
        if (indegree[nd] == 0) inNodes.push_back(nd);
      sort(inNodes.begin(), inNodes.end(), [&](auto a, auto b) {
        return a->id < b->id;
      });  // ensure ascending by id
    }
    for (auto nd : inNodes) q.push(nd);
    while (!q.empty()) {
      auto nd = q.front();
      q.pop();
      topo_forward.push_back(nd);
      for (auto nnd : nd->nextNodes) {
        --indegree[nnd];
        if (indegree[nnd] == 0) q.push(nnd);
      }
      if (nd->nextNodes.size() == 0 && !flag) {
        outNodes.push_back(nd);
      }
    }
    if (!flag) {
      sort(outNodes.begin(), outNodes.end(),
           [&](auto a, auto b) { return a->id < b->id; });
    }
    topo_backward = topo_forward;
    reverse(topo_backward.begin(), topo_backward.end());
  }
  for (auto nd : topo_forward) {
    // use to print forward output of each node
    // cout << nd->id << " " << nd->compute() << endl;
    nd->compute();
  }
  vector<ld> res;
  for (auto nd : outNodes) {
    res.push_back(nd->value);
  }
  return res;
}

void ComputationalGraph::grad(vector<ld> &outputGrad) {
  // cout << "Backward::" << endl; // use to print backward grad of each node
  for (size_t i = 0; i < outNodes.size(); ++i)
    outNodes[i]->grad = outputGrad[i];
  for (auto nd : topo_backward) {
    for (size_t i = 0; i < nd->inputLength; ++i)
      nd->operands[i]->grad += nd->grad * nd->pfpx[i];
    // use to print backward grad of each node
    // cout << nd->id << " " << nd->grad << endl;
  }
}

vector<ld> ComputationalGraph::grad() {
  clearGrad();
  for (auto nd : outNodes) nd->grad = 1;
  for (auto nd : topo_backward)
    for (size_t i = 0; i < nd->inputLength; ++i)
      nd->operands[i]->grad += nd->grad * nd->pfpx[i];
  vector<ld> res;
  for (auto nd : inNodes) {
    res.push_back(nd->grad);
  }
  return res;
}
#endif
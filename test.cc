#include <bits/stdc++.h>

#include "DNN.hpp"

int main() {
  cout << fixed << setprecision(6);
  DNN net;
  // net.manual_seed(42);
  net.addLayer(2);
  net.addLayer(3);
  net.addLayer<Sigmoid>();
  net.addLayer(1);
  net.addLayer<Sigmoid>();
  vector<vector<ld>> inputs = {{1, 2}};
  vector<vector<ld>> labels = {{1}};

  net.manual_seed(42);
  // net.addLayer(2);
  // net.addLayer(2);
  // net.addLayer<Sgn>();
  // net.addLayer(1);
  // net.addLayer<Sgn>();

  // vector<vector<ld>> inputs = {{1, 1}, {-1, -1}, {-1, 1}, {1, -1}};
  // vector<vector<ld>> labels = {{-1}, {-1}, {1}, {1}};
  cout << "[before training] --------------------" << endl;
  for (size_t i = 0; i < net.Layers.size(); i++) {
    if (i == 0 || i == 2 || i == 4) continue;
    if (i == 1) {
      for (size_t t = 0; t < net.Layers[i].size(); t++) {
        auto &node = net.Layers[i][t];
        Linear *fp = (Linear *)node->fp;
        if (t == 0) fp->W = {1.0, 0.2, 0.1};
        if (t == 1) fp->W = {0.3, 0.4, -0.2};
        if (t == 2) fp->W = {0.5, 0.3, 0.3};
      }
    }
    if (i == 3) {
      for (size_t t = 0; t < net.Layers[i].size(); t++) {
        auto &node = net.Layers[i][t];
        Linear *fp = (Linear *)node->fp;
        if (t == 0) fp->W = {0.2, 0.8, 1.0, 0.1};
      }
    }
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    vector<ld> res = net.eval(inputs[i]);
    cout << "output: " << res[0] << " target: " << labels[i][0] << endl;
  }
  vector<vector<vector<ld>>> W = net.getWeights();
  for (auto &layer : W) {
    for (auto &node : layer) {
      for (auto &w : node) {
        cout << w << " ";
      }
      cout << endl;
    }
    cout << "---------------------" << endl;
  }

  net.training(0.1, 1, inputs, labels);
  cout << "[after  training] --------------------" << endl;
  for (size_t i = 0; i < inputs.size(); i++) {
    vector<ld> res = net.eval(inputs[i]);
    cout << "output: " << res[0] << " target: " << labels[i][0] << endl;
  }
  W = net.getWeights();
  for (auto &layer : W) {
    for (auto &node : layer) {
      for (auto &w : node) {
        cout << w << " ";
      }
      cout << endl;
    }
    cout << "---------------------" << endl;
  }

  return 0;
}
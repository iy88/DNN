#include <bits/stdc++.h>

#include "DNN.hpp"

int main() {
  cout << fixed << setprecision(4);
  DNN net;
  // net.manual_seed(42);
  net.addLayer(2);
  net.addLayer(2);
  net.addLayer<Sigmoid>();
  net.addLayer(1);
  // net.addLayer<Sigmoid>();
  net.addLayer<LeakyReLU>();

  vector<vector<ld>> inputs = {{1, 1}, {0, 0}, {0, 1}, {1, 0}};
  vector<vector<ld>> labels = {{0}, {0}, {1}, {1}};
  cout << "[before training] --------------------" << endl;
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

  net.training(0.1, 10000, inputs, labels);
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
#pragma once

#include "opencv2/core/core.hpp"
#include <string>

class StereoPair {
public:
  cv::Mat left, right;
  cv::Mat true_disparity_left, true_disparity_right;
  cv::Mat disparity_left, disparity_right;

  int base_offset;
  int rows, cols;
  int min_disparity_left, max_disparity_left;
  int min_disparity_right, max_disparity_right;

  std::string name;

  void resize(float scale);

  StereoPair(cv::Mat _left, cv::Mat _right,
    cv::Mat _true_left, cv::Mat _true_right,
    int _base_offset, std::string _name);
};
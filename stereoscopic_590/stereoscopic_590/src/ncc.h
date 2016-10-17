#pragma once
#include "disparity-algorithm.h"

class NCCDisparity : public DisparityAlgorithm {
private:
  StereoPair *pair;
  cv::Mat get_template(int i, int j, bool left);
  cv::Mat get_row(int i, cv::Mat im);
  cv::Mat get_magnitude(cv::Mat im);
  int disparity(cv::Mat t, cv::Mat row, cv::Mat magnitude, int j, bool left);
  int window_size;
public:
  NCCDisparity(int _window_size) : window_size(_window_size) {}
  NCCDisparity& compute(StereoPair &pair);
};

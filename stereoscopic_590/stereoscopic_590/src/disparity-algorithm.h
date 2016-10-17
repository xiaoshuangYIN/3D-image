#pragma once
#include "stereo-pair.h"

class DisparityAlgorithm {
public:
  virtual DisparityAlgorithm& compute(StereoPair &pair) = 0;
};
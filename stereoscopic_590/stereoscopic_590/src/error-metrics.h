#pragma once
#include "opencv2/core/core.hpp"
#include <math.h>
#include <tuple>

#define EVAL_BAD_THRESH 5

/*
  rms of non-occluded regions
  rms of textured regions

  accuracy of occluded vs. non-occluded classification
*/
class ErrorMetrics {

public:
	static double get_rms_error_all (const cv::Mat gold_disparity, const cv::Mat guess_disparity) ;
	static double get_bad_matching_all (const cv::Mat gold_disparity, const cv::Mat guess_disparity) ;

  static cv::Mat get_unoccluded (const cv::Mat gold_disparity, const cv::Mat guess_disparity) ;
  static std::tuple<cv::Mat, cv::Mat, int> get_unoccluded_diff (const cv::Mat gold_disparity, const cv::Mat guess_disparity) ;

  static double get_bad_matching_unoccluded (const cv::Mat gold_disparity, const cv::Mat guess_disparity, int thresh);
  static double get_rms_error_unoccluded (const cv::Mat gold_disparity, const cv::Mat guess_disparity) ;
  static double get_correlation_unoccluded (const cv::Mat gold_disparity, const cv::Mat guess_disparity) ;
  static double get_bias_unoccluded (const cv::Mat gold_disparity, const cv::Mat guess_disparity) ;
  static double get_r_squared_unoccluded (const cv::Mat gold_disparity, const cv::Mat guess_disparity) ;
  static std::vector<int> get_occlusion_confusion_matrix (const cv::Mat gold_disparity, const cv::Mat guess_disparity);
};
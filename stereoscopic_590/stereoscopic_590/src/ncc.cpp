#include "ncc.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"

using namespace std;

/**
 * Return template of window_size centered at (i, j)
 */
cv::Mat NCCDisparity::get_template(int i, int j, bool left) {
  int r = (window_size - 1) /  2;
  int min_i = i - r;
  int min_j = j - r;

  cv::Mat t;

  cv::Rect roi(min_j, min_i, window_size, window_size);

  if (left) {
    pair->left(roi).copyTo(t);
  } else {
    pair->right(roi).copyTo(t);
  }

  // subtract mean
  cv::Scalar mean = cv::mean(t);
  cv::subtract(t, mean, t);


  return t;
}

/**
 * Get entire row of height window_size centered at i
 */
cv::Mat NCCDisparity::get_row(int i, cv::Mat im) {
  int r = (window_size - 1) /  2;
  int min_i = i - r;

  cv::Rect roi(0, min_i, pair->cols, window_size);
  return im(roi);
}

/**
 * Calculate disparity of template t within the row.
 * Use the search space defined by min_disparity and max_disparity on the StereoPair
 * centered around the template's original location j.
 *
 * left flag determines whether we expect to find the template to the right or left of
 * its original location, and how to report the disparity.
 */
int NCCDisparity::disparity(cv::Mat t, cv::Mat row, cv::Mat magnitude, int j, bool left) {
  vector<cv::Mat> row_rgb(3);
  vector<cv::Mat> t_rgb(3);
  vector<cv::Mat> detections_rgb(3);

  // Calculate search region
  int r = (window_size - 1) /  2;
  int min_j, max_j;
  if (left) {
    // right = left - disparity
    min_j = j - pair->max_disparity_left - r;
    max_j = j - pair->min_disparity_left + r;
  } else {
    // left = right + disparity
    min_j = j + pair->min_disparity_right - r;
    max_j = j + pair->max_disparity_right + r;
  }

  if (min_j < 0) min_j = 0;
  if (max_j < 0) max_j = 0;
  if (min_j >= pair->cols) min_j = pair->cols - 1;
  if (max_j >= pair->cols) max_j = pair->cols - 1;

  /* If search region is too small for the template, report
   * an occlusion */
  int bounds_width = max_j - min_j + 1;
  if (bounds_width < window_size)
    return 0;

  // Get an ROI of the search region and crop the row to that size
  cv::Rect bounded_roi(min_j, 0, bounds_width, window_size);  

  row = row(bounded_roi);
  magnitude = magnitude(bounded_roi);

  // Split to perform template matching on each channel individually
  split(row, row_rgb);
  split(row, detections_rgb);
  split(t, t_rgb);

  // For each image channel
  for (int c = 0; c < 3; c++) {
    // Get the correlation with the mean-subtracted template
    cv::filter2D(row_rgb[c], detections_rgb[c], -1, t_rgb[c]);
    /* Since the template is already mean-subtracted, we do not have
     * to mean-subtract the original image. Also, since we are only comparing
     * detections from the same template, we do not need to normalize the magnitude
     * of the template because all detections will be scaled by some constant
     * factor.
     */
  }

  // Get the 1-pixel tall center stripe that contains the correlation values
  cv::Mat detections;
  merge(detections_rgb, detections);
  cv::Rect roi(0, r, bounds_width - window_size + 1, 1);
  detections(roi).copyTo(detections);

  /* Divide by the 1-pixel tall stripe of magnitudes
   * to get the normalized correlation.
   */
  cv::divide(detections, magnitude(roi), detections);

  // Average the different detection strengths from all color channels
  cv::cvtColor(detections, detections, CV_BGR2GRAY);

  // Find the maximum
  cv::Point maxLoc;
  cv::minMaxLoc(detections, NULL, NULL, NULL, &maxLoc);

  // Transform from the search region back to the original image coordinates
  int max_loc_orig = maxLoc.x + min_j + r;

  // disparity = left - right
  if (left) {
    return j - max_loc_orig;
  } else {
    return max_loc_orig - j;
  }
}


/**
 * Return a matrix of the standard deviation of im
 * within a square region of window_size. This is used
 * for the normalization of the normalized cross correlation.
 */
cv::Mat NCCDisparity::get_magnitude(cv::Mat im) {
  // std = (sum(x^2) - sum(x)^2/n)/n
  cv::Mat im_sq, mean_sq; // mean of x^2
  cv::pow(im, 2, im_sq);
  cv::boxFilter(im_sq, mean_sq, -1,
    cv::Size(window_size, window_size), cv::Point(-1,-1), true, cv::BORDER_CONSTANT);

  cv::Mat mean, sq_mean; // (mean of x)^2
  cv::boxFilter(im, mean, -1,
    cv::Size(window_size, window_size), cv::Point(-1,-1), true, cv::BORDER_CONSTANT);
  cv::pow(mean, 2, sq_mean);

  cv::Mat var, std_dev;
  // var = mean of x^2 - (mean of x)^2
  cv::subtract(mean_sq, sq_mean, var);
  // std = sqrt(var)
  cv::sqrt(var, std_dev);

  return std_dev;
}

NCCDisparity& NCCDisparity::compute(StereoPair &_pair) {
  pair = &_pair;

  pair->disparity_left = cv::Mat(pair->rows, pair->cols, CV_8U);
  pair->disparity_right = cv::Mat(pair->rows, pair->cols, CV_8U);

  pair->disparity_left.setTo(0);
  pair->disparity_right.setTo(0);

  cv::Mat magnitude_left = get_magnitude(pair->left);
  cv::Mat magnitude_right = get_magnitude(pair->right);

  int r = (window_size- 1) / 2;
  for (int i = r; i < (pair->rows - r); i++) {
    // Print progress
    if ((i % 20) == 0)
      cout << i << endl;

    // Get original image row and magnitude of row for normalization
    cv::Mat row_left = get_row(i, pair->left);
    cv::Mat row_right = get_row(i, pair->right);
    cv::Mat mag_row_left = get_row(i, magnitude_left);
    cv::Mat mag_row_right = get_row(i, magnitude_right);

    // For each pixel in the row, calculate a disparity
    for (int j = r; j < (pair->cols - r); j++) {
      // Get a mean-subtracted template
      cv::Mat t_left = get_template(i, j, true);
      cv::Mat t_right = get_template(i, j, false);

      // Calculate disparity by NCC
      int d_left = disparity(t_left, row_right, mag_row_right, j, true);
      int d_right = disparity(t_right, row_left, mag_row_left, j, false);

      // Save in disparity image
      pair->disparity_left.at<uchar>(i, j) = d_left;
      pair->disparity_right.at<uchar>(i, j) = d_right;
    }
  }

  return *this;
}

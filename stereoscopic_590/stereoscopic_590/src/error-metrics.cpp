#include "error-metrics.h"
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

/***********
 * Helpers */

/*
 * Returns a mask of pixels that are unoccluded in both images
 */
Mat ErrorMetrics::get_unoccluded (const Mat gold_disparity, const Mat guess_disparity) {
	Mat gold_unoccluded = (gold_disparity != 0);
	Mat guess_unoccluded = (guess_disparity != 0);

	return (gold_unoccluded) & (guess_unoccluded);
}

/*
 * Returns a tuple of
 *	(
 *		residuals where unoccluded,
 *		unoccluded mask,
 *		number of unoccluded pixels
 *	)
 */
tuple<Mat, Mat, int> ErrorMetrics::get_unoccluded_diff (const Mat gold_disparity, const Mat guess_disparity) {
	Mat unoccluded_mask = get_unoccluded(gold_disparity, guess_disparity);
	int num_pixel = countNonZero(unoccluded_mask);

	Mat guess_disparity_f, gold_disparity_f;

	guess_disparity.convertTo(guess_disparity_f, CV_32FC1);
	gold_disparity.convertTo(gold_disparity_f, CV_32FC1);

	Mat diff = guess_disparity_f - gold_disparity_f;
	diff.setTo(0, unoccluded_mask == 0);
	return make_tuple(diff, unoccluded_mask, num_pixel);
}

/**
 * Percentage of unoccluded pixels that are mislabeled */
double ErrorMetrics::get_bad_matching_unoccluded (const Mat gold_disparity, const Mat guess_disparity, int thresh) {
	Mat diff, unoccluded_mask;
	int num_pixel;
	tie(diff, unoccluded_mask, num_pixel) = get_unoccluded_diff(gold_disparity, guess_disparity);

	double score = (double) countNonZero(abs(diff) > thresh) / num_pixel;
	return score;
}

/**************
 * Unoccluded */

/**
 * RMSE of disparity in units of pixels
 * for pixels that are unoccluded in both the ground truth and the computed image */
double ErrorMetrics::get_rms_error_unoccluded (const Mat gold_disparity, const Mat guess_disparity) {
	Mat diff, unoccluded_mask;
	int num_pixel;
	tie(diff, unoccluded_mask, num_pixel) = get_unoccluded_diff(gold_disparity, guess_disparity);

	double score = 1.0/sqrt(num_pixel)* norm(diff,NORM_L2);
	return score;
}

/**
 * Mean bias in units of pixels for pixels that are unoccluded in both the
 * ground truth and the computed image. */
double ErrorMetrics::get_bias_unoccluded (const Mat gold_disparity, const Mat guess_disparity) {
	Mat diff, unoccluded_mask;
	int num_pixel;
	tie(diff, unoccluded_mask, num_pixel) = get_unoccluded_diff(gold_disparity, guess_disparity);

	double score = sum(diff)[0] / (double) num_pixel;

	return score;
}

/**
 * Correlation coefficient for pixels unoccluded in both disparity maps */
double ErrorMetrics::get_correlation_unoccluded (const Mat gold_disparity, const Mat guess_disparity) {
	Mat diff, unoccluded_mask;
	int num_pixel;
	tie(diff, unoccluded_mask, num_pixel) = get_unoccluded_diff(gold_disparity, guess_disparity);

	Mat guess_disparity_f, gold_disparity_f;

	guess_disparity.convertTo(guess_disparity_f, CV_32FC1);
	gold_disparity.convertTo(gold_disparity_f, CV_32FC1);

	Scalar gold_mean, guess_mean;
	Scalar gold_std, guess_std;

	meanStdDev(gold_disparity_f, gold_mean, gold_std, unoccluded_mask);  	
	meanStdDev(guess_disparity_f, guess_mean, guess_std, unoccluded_mask);

	Mat guess_gold_mul = guess_disparity_f.mul(gold_disparity_f);
	guess_gold_mul.setTo(0, unoccluded_mask == 0);
	double num = sum(guess_gold_mul)[0] - num_pixel*guess_mean[0]*gold_mean[0];
	double denom = (num_pixel-1)*guess_std[0]*gold_std[0];

	double score = num / denom;

	return score;
}

/**
 * R^2 coefficient for pixels unoccluded in both disparity maps */
double ErrorMetrics::get_r_squared_unoccluded (const Mat gold_disparity, const Mat guess_disparity)
{
	Mat diff, unoccluded_mask;
	int num_pixel;
	tie(diff, unoccluded_mask, num_pixel) = get_unoccluded_diff(gold_disparity, guess_disparity);

	
	Mat mean_diff;
	gold_disparity.convertTo(mean_diff, CV_32FC1);
	mean_diff.setTo(0, unoccluded_mask == 0);
	double guess_mean = sum(mean_diff)[0] / (double) num_pixel;
	mean_diff = mean_diff - guess_mean;
	mean_diff.setTo(0, unoccluded_mask == 0);

	Mat mean_diff_sq, diff_sq;
	pow(mean_diff, 2, mean_diff_sq);
	pow(diff, 2, diff_sq);

	double ss_res = sum(diff_sq)[0];
	double ss_tot = sum(mean_diff_sq)[0];

	double score = 1 - ss_res / ss_tot;

	return score;
}

// Helper function
inline int count_and(Mat a, Mat b) {
	Mat c = a & b;
	return countNonZero(c);
}

/****************************
 * Occlusion Classification */

/**
 * With occlusions representing 'positive's, this function returns a vector of
 * counts of (true negative, false positive, false negative, true positive)
 */
vector<int> ErrorMetrics::get_occlusion_confusion_matrix (const Mat gold_disparity, const Mat guess_disparity)
{
	Mat gold_unoccluded = (gold_disparity != 0);
	Mat gold_occluded = (gold_disparity == 0);
	Mat guess_unoccluded = (guess_disparity != 0);	
	Mat guess_occluded = (guess_disparity == 0);

	vector<int> scores;
	scores.push_back(countNonZero(gold_unoccluded & guess_unoccluded));
	scores.push_back(countNonZero(gold_unoccluded & guess_occluded));
	scores.push_back(countNonZero(gold_occluded & guess_unoccluded));
	scores.push_back(countNonZero(gold_occluded & guess_occluded));

	return scores;
}

/**************
 * All Pixels */

/**
 * RMSE over all pixels. Prone to errors related to occlusions */
double ErrorMetrics::get_rms_error_all (const Mat gold_disparity, const Mat guess_disparity) {
	int num_pixel = gold_disparity.rows * gold_disparity.cols;

	Mat diff = gold_disparity - guess_disparity;
	double score = 1/sqrt(num_pixel)* norm(diff,NORM_L2);

	return score;
}

/**
 * Fraction of poorly labeled pixels over the entire image */
double ErrorMetrics::get_bad_matching_all (const Mat gold_disparity, const Mat guess_disparity)  {
	int num_pixel = gold_disparity.rows * gold_disparity.cols;

	Mat abs_diff;
	absdiff(gold_disparity, guess_disparity, abs_diff);
	Mat is_bad = (abs_diff > EVAL_BAD_THRESH) / 255;
	double num_bad_pixel = sum(is_bad)[0];
	double score = num_bad_pixel/num_pixel;

	return score;
}


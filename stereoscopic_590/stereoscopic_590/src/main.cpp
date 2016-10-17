#include "stereo-dataset.h"
#include "algorithms.h"
#include "error-metrics.h"
#include <opencv2/opencv.hpp>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, const char *argv[]) {
  StereoDataset dataset;
  srand (time(NULL));

  if (argc < 3) {
    cerr << "Must enter scale and either ncc or gc" << endl;
    exit(1);
  }

  float scale = atof(argv[1]);

  string alg_name(argv[2]);


  bool use_gc;
  if (alg_name == "ncc") {
    use_gc = false;
  } else if (alg_name == "gc") {
    use_gc = true;
  } else {
    cerr << "Must enter either ncc or gc" << endl;
    exit(1);
  }

  int Cp = 0;
  int V = 0;
  int window_size = 0;
  int param1 = 0;
  int param2 = 0;

  stringstream ss;
  string base_name;
  DisparityAlgorithm *alg;

  if (use_gc) {
    if (argc < 5) {
      cerr << "Must enter Cp and V" << endl;
      exit(1);
    }
    Cp = atoi(argv[3]);
    V = atoi(argv[4]);
    param1 = Cp;
    param2 = V;
    alg = new GraphCutDisparity(Cp, V);
    ss << "results/gc-scale-" << scale
      << "-Cp-" << Cp << "-V-" << V;
    base_name = ss.str();
  } else {
    if (argc < 4) {
      cerr << "Must enter window size" << endl;
      exit(1);
    }
    window_size = atoi(argv[3]);
    param1 = window_size;
    alg = new NCCDisparity(window_size);
    ss << "results/ncc-scale-" << scale
      << "-w-" << window_size;
    base_name = ss.str();
  }

  string stats_file = base_name + "-stats.csv";
  ofstream stats_stream;
  stats_stream.open(stats_file);
  stats_stream << "Scale,Algorithm,"
    << "Param1,Param2,"
    << "Name,Elapsed Time,"
    << "Left RMSE,Right RMSE,"
    << "Left BM_Unocc,Right BM_Unocc,"
    << "Left Bias,Right Bias,"
    << "Left Corr,Right Corr,"
    << "Left R2,Right R2,"
    << "Left tn,Left fp,Left fn,Left tp,"
    << "Right tn,Right fp,Right fn,Right tp"
    << endl;

  for (string name : dataset.get_all_datasets()) {
    StereoPair pair = dataset.get_stereo_pair(name);
    pair.resize(scale);

    clock_t start_time = clock();
    alg->compute(pair);
    clock_t end_time = clock();
    double elapsed_time = (double) (end_time - start_time) / (double) CLOCKS_PER_SEC;

    double rmse_left = ErrorMetrics::get_rms_error_unoccluded(pair.true_disparity_left, pair.disparity_left);
    double rmse_right = ErrorMetrics::get_rms_error_unoccluded(pair.true_disparity_right, pair.disparity_right);

    double bm_unocc_left = ErrorMetrics::get_bad_matching_unoccluded(pair.true_disparity_left, pair.disparity_left, 3);
    double bm_unocc_right = ErrorMetrics::get_bad_matching_unoccluded(pair.true_disparity_right, pair.disparity_right, 3);

    double bias_left = ErrorMetrics::get_bias_unoccluded(pair.true_disparity_left, pair.disparity_left);
    double bias_right = ErrorMetrics::get_bias_unoccluded(pair.true_disparity_right, pair.disparity_right);

    double corr_left = ErrorMetrics::get_correlation_unoccluded(pair.true_disparity_left, pair.disparity_left);
    double corr_right = ErrorMetrics::get_correlation_unoccluded(pair.true_disparity_right, pair.disparity_right);

    double r_squared_left = ErrorMetrics::get_r_squared_unoccluded(pair.true_disparity_left, pair.disparity_left);
    double r_squared_right = ErrorMetrics::get_r_squared_unoccluded(pair.true_disparity_right, pair.disparity_right);

    vector<int> confusion_left = ErrorMetrics::get_occlusion_confusion_matrix(pair.true_disparity_left, pair.disparity_left);
    vector<int> confusion_right = ErrorMetrics::get_occlusion_confusion_matrix(pair.true_disparity_right, pair.disparity_right);

    stats_stream << scale << ","
      << alg_name << ","
      << param1 << ","
      << param2 << ","
      << pair.name << "," << elapsed_time << ","
      << rmse_left << "," << rmse_right << ","
      << bm_unocc_left << "," << bm_unocc_right << ","
      << bias_left << "," << bias_right << ","
      << corr_left << "," << corr_right << ","
      << r_squared_left << "," << r_squared_right << ","
      << confusion_left[0] << "," << confusion_left[1] << ","
      << confusion_left[2] << "," << confusion_left[3] << ","
      << confusion_right[0] << "," << confusion_right[1] << ","
      << confusion_right[2] << "," << confusion_right[3]
      << endl;

    string left_file = base_name + "-" + pair.name + "-left.png";
    string right_file = base_name + "-" + pair.name + "-right.png";
    string true_left_file = base_name + "-" + pair.name + "-left-true.png";
    string true_right_file = base_name + "-" + pair.name + "-right-true.png";
    cv::imwrite(left_file, pair.disparity_left);
    cv::imwrite(right_file, pair.disparity_right);
    cv::imwrite(true_left_file, pair.true_disparity_left);
    cv::imwrite(true_right_file, pair.true_disparity_right);
  }
  stats_stream.close();
}
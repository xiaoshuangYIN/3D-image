#pragma once

#include "stereo-pair.h"
#include <vector>
#include <string>

class StereoDataset {
private:
  const char *left_format = "./data/%s/Illum%d/Exp%d/view1.png";
  const char *right_format = "./data/%s/Illum%d/Exp%d/view5.png";
  const char *true_left_format = "./data/%s/disp1.png";
  const char *true_right_format = "./data/%s/disp5.png";
  const char *offset_format = "./data/%s/dmin.txt";
public:
  StereoPair get_stereo_pair(
    const std::string dataset = "Bowling1",
    int illumination=1,
    int exposure=1);

  std::vector<std::string> get_all_datasets();
  std::vector<int> get_all_illuminations();
  std::vector<int> get_all_exposures();

  std::string get_random_dataset();
  int get_random_illumination();
  int get_random_exposure();

  StereoPair get_random_stereo_pair();
};
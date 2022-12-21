//
// Created by ljx on 2022/12/21.
//

#ifndef STEREONET_RKNN_UTILS_H
#define STEREONET_RKNN_UTILS_H
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"

void ReadImages(const std::string imagesPath, std::vector<std::string> &lImg, std::vector<std::string> &rImg);

bool isContain(std::string str1, std::string str2);


#endif //STEREONET_RKNN_UTILS_H

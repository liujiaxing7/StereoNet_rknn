//
// Created by ljx on 2022/12/21.
//

#include "utils.h"
bool isContain(std::string str1, std::string str2)
{
    if (str1.find(str2)!= std::string::npos)
    {
        return true;
    } else
    {
        return false;
    }

}

void ReadImages(const std::string imagesPath, std::vector<std::string> &lImg, std::vector<std::string> &rImg)
{
    std::vector<cv::String> allFiles;
    cv::glob(imagesPath, allFiles, true);

    if (allFiles.size() == 0)
    {
        std::cout<< "there in no file;"<<std::endl;
    }
    for (int i =0 ; i < allFiles.size(); ++i)
    {
        std::string filename = allFiles[i];
        if (isContain(allFiles[i],"L.png"))
        {
            lImg.push_back(filename);
        }
        else if (isContain(allFiles[i], "R.png"))
        {
            rImg.push_back(filename);
        }
    }

    std::cout<<"load images"<<std::endl;
}
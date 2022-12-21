#include <iostream>
//#include "npu_rk/model_npu_rk.h"
#include <opencv2/highgui.hpp>
#include<sys/time.h>
#include <ctime>
#include <unistd.h>
#include <rknn_api.h>
#include <opencv2/imgproc.hpp>
#include "fstream"
#include "utils.h"

bool Read(const char *filename, unsigned char **data, int &size) {
    *data = nullptr;
    ::FILE *fp;
    const int offset = 0;
    int ret = 0;
    unsigned char *dataTemp;

    fp = fopen(filename, "rb");

    fseek(fp, 0, SEEK_END);
    size = ftell(fp);

    ret = fseek(fp, offset, SEEK_SET);

    dataTemp = (unsigned char *) malloc(size);
    ret = fread(dataTemp, 1, size, fp);

    *data = dataTemp;
    fclose(fp);

    return true;

    exit:
    return false;
}
int main(int argc, char **argv) {

    if (argc < 2) {
        std::cout << "modelpath: mnnpath:\n"
                  << "data_path: images_path\n"
                  << std::endl;
        return -1;
    }

    int img_w = 640;
    int img_h = 400;

    cv::Mat dstL, dstR;
    rknn_context ctx;

    //load_time
    float load_time_use = 0;
    struct timeval load_start;
    struct timeval load_end;
    gettimeofday(&load_start,NULL);

    std::string modelFile =  argv[1];
    int modelSize = 0;
    unsigned char *data;
    int ret = Read(modelFile.c_str(), &data, modelSize);
    ret = rknn_init(&ctx, data, modelSize, 0, NULL);
    std::cout<<"load success"<<std::endl;

    gettimeofday(&load_end,NULL);
    load_time_use=(load_end.tv_sec-load_start.tv_sec)*1000000+(load_end.tv_usec-load_start.tv_usec);
    std::cout<<"load model time : "<<load_time_use/1000.0<<"ms"<<std::endl;

    //加载图像
    std::string imagespath = argv[2];

    std::vector<std::string> limg;
    std::vector<std::string> rimg;

    ReadImages(imagespath, limg, rimg);

    const size_t size = limg.size();

    for (size_t imgid = 0; imgid < size; ++imgid)
    {
        auto imgL = limg.at(imgid);
        auto imgR = rimg.at(imgid);
        cv::Mat image_in_L = cv::imread(imgL);
        cv::Mat image_in_R = cv::imread(imgR);

        cv::resize(image_in_L, dstL, cv::Size(img_w
                , img_h));
        cv::resize(image_in_R, dstR, cv::Size(img_w
                , img_h));

        using TYPE = uint8_t;
        cv::cvtColor(dstL, dstL, cv::COLOR_BGR2RGB);
        TYPE *ptrL = (TYPE *) malloc(dstL.rows * dstL.cols * dstL.channels() * sizeof(TYPE));
        memcpy(ptrL, dstL.data , dstL.rows * dstL.cols * dstL.channels() * sizeof(TYPE));

        cv::cvtColor(dstR, dstR, cv::COLOR_BGR2RGB);
        TYPE *ptrR = (TYPE *) malloc(dstR.rows * dstR.cols * dstR.channels() * sizeof(TYPE));
        memcpy(ptrR, dstR.data , dstR.rows * dstR.cols * dstR.channels() * sizeof(TYPE));

        //data_to gpu
        float data_to_gpu = 0;
        struct timeval data_gpu_start;
        struct timeval data_gpu_end;
        gettimeofday(&data_gpu_start,NULL);

        rknn_input inputs[1];
        rknn_output outputs[1];
        inputs[0].buf = ptrL;
        inputs[0].index = 0;
        inputs[0].size = dstL.cols * dstL.rows * dstL.channels();
        inputs[0].pass_through = false;
        inputs[0].type = RKNN_TENSOR_UINT8;
        inputs[0].fmt = RKNN_TENSOR_NHWC;

        inputs[1].buf = ptrR;
        inputs[1].index = 0;
        inputs[1].size = dstL.cols * dstL.rows * dstL.channels();
        inputs[1].pass_through = false;
        inputs[1].type = RKNN_TENSOR_UINT8;
        inputs[1].fmt = RKNN_TENSOR_NHWC;


        ret = rknn_inputs_set(ctx, 1, inputs);
        if(ret < 0) {
            printf("rknn_input_set fail! ret=%d\n", ret);
            if(ctx > 0)         rknn_destroy(ctx);
            return ret;
        }
        gettimeofday(&data_gpu_end,NULL);
        data_to_gpu=(data_gpu_end.tv_sec-data_gpu_start.tv_sec)*1000000+(data_gpu_end.tv_usec-data_gpu_start.tv_usec);
        std::cout<<"data_to_gpu time : "<<data_to_gpu/1000.0<<"ms"<<std::endl;


        //forward time
        float forward_time_use = 0;
        struct timeval forward_start;
        struct timeval forward_end;
        gettimeofday(&forward_start,NULL);
        ret = rknn_run(ctx, nullptr);
        if(ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
            if(ctx > 0)         rknn_destroy(ctx);
            return ret;
        }
        gettimeofday(&forward_end,NULL);
        forward_time_use=(forward_end.tv_sec-forward_start.tv_sec)*1000000+(forward_end.tv_usec-forward_start.tv_usec);
        std::cout<<"forward time : "<<forward_time_use/1000.0<<"ms"<<std::endl;


        //data to cpu time
        float data_to_cpu = 0;
        struct timeval data_cpu_start;
        struct timeval data_cpu_end;
        gettimeofday(&data_cpu_start,NULL);


        outputs[0].want_float = true;
        outputs[0].is_prealloc = false;
        ret = rknn_outputs_get(ctx, 1, outputs, nullptr);
        if(ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
            if(ctx > 0)         rknn_destroy(ctx);
            return ret;
        }
        std::vector<float*> network_outputs {
                (float*)outputs[0].buf,
        };

        gettimeofday(&data_cpu_end,NULL);
        data_to_cpu=(data_cpu_end.tv_sec-data_cpu_start.tv_sec)*1000000+(data_cpu_end.tv_usec-data_cpu_start.tv_usec);
        std::cout<<"data_to_cpu time : "<<data_to_cpu/1000.0<<"ms"<<std::endl;

        //可视化图像
        int moutSize_w = dstL.cols;
        int moutSize_h = dstL.rows;
        cv::Mat moutimg;
        moutimg.create(cv::Size(moutSize_w, moutSize_h), CV_32FC1);

        cv::Mat showImg;

        for (int i=0; i<moutSize_h; ++i) {
            {
                for (int j=0; j<moutSize_w; ++j)
                {
                    //                std::cout<<"index: " << i*outSize_w+j <<std::endl;
                    //                std::cout<<"value:" <<network_outputs[0][i*outSize_w+j]<<std::endl;
                    moutimg.at<float>(i,j) = network_outputs[0][i*moutSize_w+j];
                }
            }
        }

        cv::Mat outimg;
        int outSize_w = dstL.cols;
        int outSize_h = dstL.rows;
        cv::resize(moutimg, outimg, cv::Size(outSize_w
                , outSize_h),0,0,cv::INTER_CUBIC);
        //可视化
        double minv = 0.0, maxv = 0.0;
        double* minp = &minv;
        double* maxp = &maxv;
        minMaxIdx(outimg,minp,maxp);
        float minvalue = (float)minv;
        float maxvalue = (float)maxv;

        for (int i=0; i<outSize_h; ++i) {
            {
                for (int j=0; j<outSize_w; ++j)
                {

                    outimg.at<float>(i,j) = 255* (outimg.at<float>(i,j) - minvalue)/(maxvalue-minvalue);
                }
            }
        }

        outimg.convertTo(showImg,CV_8U);
        cv::Mat colorimg;
        cv::Mat colorimgfinal;
        //        cv2.applyColorMap(cv2.convertScaleAbs(norm_disparity_map,1), cv2.COLORMAP_MAGMA)
        cv::convertScaleAbs(showImg,colorimg);
        cv::applyColorMap(colorimg,colorimgfinal,cv::COLORMAP_PARULA);
//        cv::applyColorMap(colorimg,colorimgfinal,cv::COLORMAP_HOT);
        //    namedWindow("image", cv::WINDOW_AUTOSIZE);
        //    imshow("image", colorimgfinal);
        cv::imwrite("../result_"+std::to_string(imgid)+".png",colorimgfinal);
    }

    return 0;
}

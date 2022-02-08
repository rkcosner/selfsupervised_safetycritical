#include <linux/videodev2.h>
#include <linux/v4l2-common.h>
#include <linux/v4l2-controls.h>
#include <linux/videodev2.h>
#include <opencv4/opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include "elas/elas_opencv_bridge.h"
#include <thread> 


#include "Eigen/Dense"
#include "Eigen/Core"
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <chrono>

// Camera Array Parameters
float baseline = 0.052; 
float c_x = 40;
float focus = 221.28672543;
constexpr float postRectificationRescale = 1.;
int rectification_shift1_2 = 14; 
int rectification_shift2_3 = 3; 

float err_threshold = 0.99;
float calib_data[16] = {1*postRectificationRescale, 0, 0, -142.27140427/2,
                        0, 1*postRectificationRescale, 0, -102.50337029/2,
                        0, 0, 0, -focus*postRectificationRescale, 
                        0, 0, -1.0/0.052*0.525, 0};

cv::Mat Q_(4,4, CV_32FC1, calib_data);

// Image Properties 
constexpr int width = 5120;
constexpr int height= 800;
constexpr float img_scale = 0.25;
const int depth = 10; 
const int cvt_code = 46; 
const int convert2rgb = 0;
constexpr int img_width = int(width*img_scale);
constexpr int frame_width = img_width / 4;
constexpr int img_height = int(height*img_scale);
cv::Mat disp1_rmap[2][2];
cv::Mat disp2_rmap[2][2];
cv::Mat disp3_rmap[2][2];

// Stereo Vars 
StereoEfficientLargeScale elas1_(0,32); 
StereoEfficientLargeScale elas2_(0,32); 
StereoEfficientLargeScale elas3_(0,64); 


cv::Mat disp1_;
cv::Mat disp2_;
cv::Mat disp3_;
cv::Mat disp3plusErr_; 
cv::Mat disp3minusErr_;
cv::Mat d3_maxErr_(img_height*postRectificationRescale, img_width/4*postRectificationRescale, CV_8U, cv::Scalar(0));
cv::Mat disp3hat_( img_height*postRectificationRescale, img_width/4*postRectificationRescale, CV_8U, cv::Scalar(0));

cv::Mat imgL_; 
cv::Mat imgC_; 
cv::Mat imgR_; 

constexpr int err_depth = 65;



void getCVimage(const sensor_msgs::Image rosImg, cv::Mat &img){
    cv_bridge::CvImagePtr cvPtr = cv_bridge::toCvCopy(rosImg, sensor_msgs::image_encodings::TYPE_8UC1); 
    cv::Mat(cvPtr->image).convertTo(img, CV_8UC1);
}






// Image Rectification Values D1
// cv::Mat D1_cameraMatrix_left = (cv::Mat_<float>(3,3) << 200.13893387  , 0.,         140.76444227,
//                                                 0.     ,    200.13893387,  93.37328257,
//                                               0.        ,   0.        ,   1.     );
// cv::Mat D1_distortionCoefficients_left = (cv::Mat_<float>(1,5) <<  -0.1255048 ,  0.19631011 , 0.    ,      0.  ,       -0.11851587);
// cv::Mat D1_R1 = (cv::Mat_<float>(3,3) <<    0.98395951 , 0.17665223, -0.02485322,
//                                             -0.1770544  , 0.98408644 ,-0.0150203 ,
//                                              0.02180435 , 0.01917974 , 0.99957827  );
// cv::Mat D1_P1 = (cv::Mat_<float>(3,4) <<    200.13893387  , 0.    ,     143.32390594 ,  0.  ,
//                                              0.       ,  200.13893387, 103.14701462  , 0.,
//                                              0.        ,   0.        ,   1.        ,   0.  );

// cv::Mat D1_cameraMatrix_right = (cv::Mat_<float>(3,3) << 200.13893387 ,  0.    ,     136.80080128,
//                                                    0.  ,       200.13893387, 107.47382618,
//                                                     0.        ,   0.        ,   1.     );
// cv::Mat D1_distortionCoefficients_right = (cv::Mat_<float>(1,5) <<  -0.07347874 , 0.05198529 , 0. ,         0.  ,        0.01706871);
// cv::Mat D1_R2 = (cv::Mat_<float>(3,3) <<     0.98440286 , 0.17563299 ,-0.01020129,
//                                              -0.17543106 , 0.98432462 , 0.01813819,
//                                              0.01322704 ,-0.01606567 , 0.99978345);
// cv::Mat D1_P2 = (cv::Mat_<float>(3,4) <<   200.13893387 ,   0.      ,    143.60647392 ,-443.56273572,
//                                              0.     ,     200.13893387 , 103.14701462 ,   0.      ,
//                                              0.   ,         0.  ,          1.      ,      0.    );




cv::Mat D1_cameraMatrix_left = (cv::Mat_<float>(3,3) << 213.38477548 ,  0.,         160.38018106,
                                               0.    ,     213.38477548 , 90.42188151,
                                              0.        ,   0.        ,   1.     );
cv::Mat D1_distortionCoefficients_left = (cv::Mat_<float>(1,5) <<  -0.01021809 , 0.19241332 , 0.   ,       0. ,        -0.33436766);
cv::Mat D1_R1 = (cv::Mat_<float>(3,3) <<    0.99676252,  0.03304748 ,-0.07329633,
                                            -0.03419278,  0.99931114, -0.01442593,
                                            0.0727691 ,  0.01688543  ,0.99720587  );
cv::Mat D1_P1 = (cv::Mat_<float>(3,4) <<    213.38477548 ,  0.   ,      185.98413086  , 0. ,
                                             0.    ,     213.38477548 , 99.90508556 ,  0. ,
                                             0.        ,   0.        ,   1.        ,   0.  );

cv::Mat D1_cameraMatrix_right = (cv::Mat_<float>(3,3) << 213.38477548  , 0. ,        154.06712743,
                                                    0. ,        213.38477548, 106.43590771,
                                                    0.        ,   0.        ,   1.     );
cv::Mat D1_distortionCoefficients_right = (cv::Mat_<float>(1,5) <<  0.04534914 ,-0.13212098 , 0.    ,      0.   ,       0.16346267);
cv::Mat D1_R2 = (cv::Mat_<float>(3,3) <<      0.9959765 ,  0.03299123, -0.08332097,
                                             -0.03168035,  0.99935336 , 0.01700663,
                                               0.08382816 ,-0.01429856 , 0.99637763 );
cv::Mat D1_P2 = (cv::Mat_<float>(3,4) <<   213.38477548  ,  0.  ,        182.41349792 ,-497.26944683,
                                              0.   ,       213.38477548  , 99.90508556 ,   0.       ,
                                             0.   ,         0.  ,          1.      ,      0.    );






// Image Rectification Values D3
cv::Mat D2_cameraMatrix_left = (cv::Mat_<float>(3,3) << 216.85273609 ,  0.    ,     154.40694161,
                                              0.  ,       216.85273609 ,105.71968046,
                                              0.        ,   0.        ,   1.     );
cv::Mat D2_distortionCoefficients_left = (cv::Mat_<float>(1,5) <<  0.02738416, -0.05257071 , 0.,          0. ,         0.07259553);
cv::Mat D2_R1 = (cv::Mat_<float>(3,3) <<       0.99911503, -0.00219649 , 0.04200392,
                                             0.00229048  ,0.99999498, -0.00218977,
                                            -0.0419989  , 0.00228404 , 0.99911505 );
cv::Mat D2_P1 = (cv::Mat_<float>(3,4) <<      216.85273609 ,  0.    ,     140.92088318 ,  0.  , 
                                              0.  ,       216.85273609 ,105.3395071  ,  0.   ,
                                             0.        ,   0.        ,   1.        ,   0.  );

cv::Mat D2_cameraMatrix_right = (cv::Mat_<float>(3,3) <<216.85273609  , 0.     ,    152.07872685,
                                                   0.     ,    216.85273609 ,104.04086618,
                                                    0.        ,   0.        ,   1.     );
cv::Mat D2_distortionCoefficients_right = (cv::Mat_<float>(1,5) << -0.01407029,  0.12369421 , 0.,          0.  ,       -0.19781305);
cv::Mat D2_R2 = (cv::Mat_<float>(3,3) <<       0.99903657 , 0.00484119,  0.04361749,
                                            -0.00493879 , 0.99998554 , 0.00213018,
                                              -0.04360654 ,-0.00234354,  0.99904603  );
cv::Mat D2_P2 = (cv::Mat_<float>(3,4) <<     216.85273609    ,0.     ,     132.90382004, -494.29453342,
                                            0.    ,      216.85273609 , 105.3395071  ,   0.        ,
                                             0.        ,   0.        ,   1.        ,   0.  );





// Image Rectification Values D3
cv::Mat D3_cameraMatrix_left = (cv::Mat_<float>(3,3) << 216.0422918 ,   0.        , 161.25731835,
                                              0.        , 216.0422918 ,  89.79627252,
                                              0.        ,   0.        ,   1.     );
cv::Mat D3_distortionCoefficients_left = (cv::Mat_<float>(1,5) << -0.01682369,  0.22907142,  0.        ,  0.        , -0.39213378);
cv::Mat D3_R1 = (cv::Mat_<float>(3,3) <<        0.99980977,  0.01748846, -0.00863564,
                                             -0.01763787,  0.99969059, -0.01753911,
                                               0.00832624,  0.01768808,  0.99980888  );
cv::Mat D3_P1 = (cv::Mat_<float>(3,4) <<        216.0422918 ,   0.        , 165.85542297, 0, 
                                             0.        , 216.0422918 ,  99.9402914 ,   0. ,
                                             0.        ,   0.        ,   1.        ,   0.  );
cv::Mat D3_cameraMatrix_right = (cv::Mat_<float>(3,3) << 216.0422918 ,   0.        , 151.88891564,
                                                    0.        , 216.0422918 , 104.14842486,
                                                    0.        ,   0.        ,   1.     );
cv::Mat D3_distortionCoefficients_right = (cv::Mat_<float>(1,5) << -0.00733585,  0.10949051,  0.        ,  0.        , -0.182561238);
cv::Mat D3_R2 = (cv::Mat_<float>(3,3) <<        0.99954373,  0.02270267, -0.01992291,
                                             -0.02234817,  0.99959108,  0.01783911,
                                              0.02031975, -0.01738573,  0.99964236  );
cv::Mat D3_P2 = (cv::Mat_<float>(3,4) <<        216.0422918 ,    0.        ,  155.36833954, -990.02692196,
                                            0.        ,  216.0422918 ,   99.9402914 ,    0.      ,
                                             0.        ,   0.        ,   1.        ,   0.  );




void set_rmap_calibration_matrices(){

    // GET STEREORECTIFICATION REMAP MATRICES
    // Stereorectification Maps for D1
    cv::initUndistortRectifyMap(D1_cameraMatrix_left, D1_distortionCoefficients_left,   D1_R1, D1_P1, cv::Size(img_width*img_scale, img_height) ,  CV_16SC2, disp1_rmap[0][0], disp1_rmap[0][1]);
    cv::initUndistortRectifyMap(D1_cameraMatrix_right, D1_distortionCoefficients_right, D1_R2, D1_P2, cv::Size(img_width*img_scale ,img_height), CV_16SC2,   disp1_rmap[1][0], disp1_rmap[1][1]);
    // Stereorectification Maps for D2
    cv::initUndistortRectifyMap(D2_cameraMatrix_left, D2_distortionCoefficients_left,   D2_R1, D2_P1, cv::Size(img_width*img_scale, img_height) ,  CV_16SC2, disp2_rmap[0][0], disp2_rmap[0][1]);
    cv::initUndistortRectifyMap(D2_cameraMatrix_right, D2_distortionCoefficients_right, D2_R2, D2_P2, cv::Size(img_width*img_scale ,img_height), CV_16SC2,   disp2_rmap[1][0], disp2_rmap[1][1]);
    // Stereorectification Maps for D3
    cv::initUndistortRectifyMap(D3_cameraMatrix_left, D3_distortionCoefficients_left,   D3_R1, D3_P1, cv::Size(img_width*img_scale, img_height) ,  CV_16SC2, disp3_rmap[0][0], disp3_rmap[0][1]);
    cv::initUndistortRectifyMap(D3_cameraMatrix_right, D3_distortionCoefficients_right, D3_R2, D3_P2, cv::Size(img_width*img_scale ,img_height), CV_16SC2,   disp3_rmap[1][0], disp3_rmap[1][1]);

}




void getDisparity1(cv::Mat &disp_imgL, cv::Mat &disp_imgR, cv::Mat &disp, int edge_max){
        
        cv::Mat imgL_rect1; 
        cv::Mat imgR_rect1; 
        
        cv::remap(disp_imgL, imgL_rect1, disp1_rmap[0][0], disp1_rmap[0][1], cv::INTER_LINEAR, cv::BORDER_DEFAULT, cv::Scalar());
        cv::remap(disp_imgR, imgR_rect1, disp1_rmap[1][0], disp1_rmap[1][1], cv::INTER_LINEAR, cv::BORDER_DEFAULT, cv::Scalar());

        cv::resize(imgL_rect1, imgL_rect1, cv::Size(),postRectificationRescale, postRectificationRescale); 
        cv::resize(imgR_rect1, imgR_rect1, cv::Size(),postRectificationRescale,postRectificationRescale); 

        elas1_(imgL_rect1, imgR_rect1, disp, edge_max); 
        
        disp =  (disp/16);
        cv::flip(disp, disp, 1);
        disp.convertTo(disp, CV_8U); 

}


void getDisparity2(cv::Mat &disp_imgL, cv::Mat &disp_imgR, cv::Mat &disp, int edge_max){

        cv::Mat imgL_rect2; 
        cv::Mat imgR_rect2; 

        cv::remap(disp_imgL, imgL_rect2, disp2_rmap[0][0], disp2_rmap[0][1], cv::INTER_LINEAR, cv::BORDER_DEFAULT, cv::Scalar());
        cv::remap(disp_imgR, imgR_rect2, disp2_rmap[1][0], disp2_rmap[1][1], cv::INTER_LINEAR, cv::BORDER_DEFAULT, cv::Scalar());

        cv::resize(imgL_rect2, imgL_rect2, cv::Size(), postRectificationRescale, postRectificationRescale); 
        cv::resize(imgR_rect2, imgR_rect2, cv::Size(), postRectificationRescale, postRectificationRescale); 

        elas2_(imgL_rect2, imgR_rect2, disp, edge_max); 
        
        disp =  (disp/16);
        cv::flip(disp, disp, 1);
        disp.convertTo(disp, CV_8U); 

} 

void getDisparity3(cv::Mat &disp_imgL, cv::Mat &disp_imgR, cv::Mat &disp, int edge_max){

        cv::Mat imgL_rect3; 
        cv::Mat imgR_rect3; 
        
        cv::remap(disp_imgL, imgL_rect3, disp3_rmap[0][0], disp3_rmap[0][1], cv::INTER_LINEAR, cv::BORDER_DEFAULT, cv::Scalar());
        cv::remap(disp_imgR, imgR_rect3, disp3_rmap[1][0], disp3_rmap[1][1], cv::INTER_LINEAR, cv::BORDER_DEFAULT, cv::Scalar());

        cv::resize(imgL_rect3, imgL_rect3, cv::Size(), postRectificationRescale, postRectificationRescale); 
        cv::resize(imgR_rect3, imgR_rect3, cv::Size(), postRectificationRescale, postRectificationRescale); 

        elas3_(imgL_rect3, imgR_rect3, disp, edge_max);

        disp =  (disp/16);
        cv::flip(disp, disp, 1);
        disp.convertTo(disp, CV_8U); 
}

void getD3hat(cv::Mat &disp1, cv::Mat &disp2, cv::Mat &disp3_hat){
    disp3_hat.setTo(cv::Scalar(0)); 
    int height = img_height*postRectificationRescale; 
    int width = img_width/4*postRectificationRescale; 

    int j_hat; 

    for (int i=0; i<height; i++){
        for(int j=0; j+rectification_shift2_3<width; j++){
            j_hat = j + disp1.at<uint8_t>(i,j); 
            if (j_hat + rectification_shift1_2< width){ 

                int d2 = disp2.at<uint8_t>(i,j_hat + rectification_shift1_2); 
                int d1 = j_hat-j;

                disp3_hat.at<uint8_t>(i,j+rectification_shift2_3) = d2 + d1; 
            }
        }
    }

//     for (int i=0; i<height; i++){
//         for(int j=0; j<width; j++){
//             j_hat = j + disp2.at<uint8_t>(i,j); 
//             if (j_hat < width){ 

//                 int d1 = disp1.at<uint8_t>(i,j_hat); 
//                 int d2 = j_hat-j;

//                 disp3_hat.at<uint8_t>(i,j) = d2 + d1; 
//             }
//         }
//     }


}



void initErrTensor(Eigen::Tensor<float,3, Eigen::RowMajor> &et){
    // Set the error tensor with a decaying exponential that sums to 1 
    for (int i=0; i<img_height*postRectificationRescale; ++i){
        for (int j=0; j<frame_width*postRectificationRescale; ++j){
            for (int k=0; k<err_depth; ++k){
                et(i,j,k) = exp(-k)/1.5819767068693267;
            }
        }
    }   
}

void get_max_err(const Eigen::Tensor<float,3, Eigen::RowMajor> &et, cv::Mat &maxErr){
    maxErr.setTo(cv::Scalar(0));
    for (int i=0; i<img_height*postRectificationRescale; ++i){
        for (int j=0; j<frame_width*postRectificationRescale; ++j){
            float prob_sum = et(i,j,err_depth-1);
            for (int k=0; k<err_depth; ++k){
                prob_sum += et(i, j, k);
                if (prob_sum > err_threshold){
                    maxErr.at<uint8_t>(i, j) = k;
                    break;
                }
            }
        }
    }
}


 void view_slice(const Eigen::Tensor<float,3, Eigen::RowMajor> &et, int slice_num){
     cv::Mat slice( img_height*postRectificationRescale, frame_width*postRectificationRescale, CV_8U, cv::Scalar(0));
     for (int i=0; i<img_height*postRectificationRescale; ++i){
         for (int j=0; j<frame_width*postRectificationRescale; ++j){
            slice.at<uint8_t>(i,j) = 255*et(i,j,slice_num);
         }
     }
//     slice.convertTo(slice, CV_8UC3);
//     cv::applyColorMap(slice, slice, cv::COLORMAP_JET);
     std::string winname = "Slice ";

     cv::imshow(winname + std::to_string(slice_num), slice);
     cv::waitKey(5);

 }

std::string get_current_time(){
    using namespace std::chrono;

    // get current time
    auto now = system_clock::now();

    // get number of milliseconds for the current second
    // (remainder after division into seconds)
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    // convert to std::time_t in order to convert to std::tm (broken time)
    auto timer = system_clock::to_time_t(now);

    // convert to broken time
    std::tm bt = *std::localtime(&timer);

    std::ostringstream time_stream;

    time_stream << std::put_time(&bt, "%H:%M:%S"); // HH:MM:SS
    time_stream << '.' << std::setfill('0') << std::setw(3) << ms.count();

    return time_stream.str();
}

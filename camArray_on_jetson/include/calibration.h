#include <opencv4/opencv2/opencv.hpp>

// Camera Array Parameters
float baseline = 0.052; 
float c_x = 40;
float focus = 221.28672543;
float rescale = 1; 
// float calib_data[16] = {1, 0, 0, -160,
//                         0, 1, 0, -100,
//                         0, 0, 0, -f,
//                         0, 0, -1/baseline/3/5*4, 0};
// float calib_data[16] = {1, 0, 0, -142.27140427,
//                         0, 1, 0, -102.50337029,
//                         0, 0, 0, -221.28672543,
//                         0, 0,  -2.0501144,-0.07451122}; // -0.07451122

// float calib_data[16] = {1, 0, 0, -142.27140427,
//                         0, 1, 0, -102.50337029,
//                         0, 0, 0, -focus, 
//                         0, 0, -1/0.052*0.575, 0};

float calib_data[16] = {1*rescale, 0, 0, -142.27140427/2,
                        0, 1*rescale, 0, -102.50337029/2,
                        0, 0, 0, -focus*rescale, 
                        0, 0, -1/0.052*0.575, 0};

cv::Mat Q_(4,4, CV_32FC1, calib_data);

// Image Properties 
const int width = 5120; 
const int height= 800; 
const float img_scale = 0.25; 
const int depth = 10; 
const int cvt_code = 46; 
const int convert2rgb = 0;
const int img_width = int(width*img_scale); 
const int img_height = int(height*img_scale); 
cv::Mat disp1_rmap[2][2];
cv::Mat disp2_rmap[2][2];
cv::Mat disp3_rmap[2][2];


// Image Rectification Values D3
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
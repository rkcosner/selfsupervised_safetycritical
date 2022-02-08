#include "ros/ros.h"
#include "std_msgs/String.h"
#include <linux/videodev2.h>
#include <linux/v4l2-common.h>
#include <linux/v4l2-controls.h>
#include <linux/videodev2.h>
#include <opencv4/opencv2/opencv.hpp>
#include <sstream>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <signal.h>
#include <fcntl.h>
#include <future>
#include <iomanip>

#include "calibration.h"

int node_frequency = 20; // Hz

void getResizedGray(cv::Mat &img){
        cv::resize(img, img, cv::Size(), rescale, rescale); 
}

// Multithread Controller Input 
std::string GetLineFromCin() {
    std::string line;
    std::getline(std::cin, line);
    return line;
}
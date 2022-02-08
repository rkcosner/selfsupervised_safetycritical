#include "ros/ros.h"
#include "std_msgs/String.h"

#include <iostream>
#include <sstream>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Twist.h>
#include "uncertainty_estimator/triple_image.h"

#include "cam_array.h"
#include "cbf.h"
#include "pointclouds.h"

#define VISUALIZE
int experiment_type; 


void inputCallback(const geometry_msgs::Twist::ConstPtr & msg){ 
  std::cout<<"Filtered Input: u = \t [ " <<  msg->linear.x << ", " << msg->angular.y << " ]\n"; 
}


void imageCallback(const uncertainty_estimator::triple_image::ConstPtr& msg)
{
    getCVimage(msg->left, imgL_); 
    getCVimage(msg->center, imgC_); 
    getCVimage(msg->right, imgR_); 

    std::thread disp1_thread(getDisparity1, std::ref(imgL_), std::ref(imgC_), std::ref(disp1_), 0); 
    std::thread disp2_thread(getDisparity2, std::ref(imgC_), std::ref(imgR_), std::ref(disp2_), 0);
    std::thread disp3_thread(getDisparity3, std::ref(imgL_), std::ref(imgR_), std::ref(disp3_), 0);  // 80 ms 

    disp1_thread.join(); 
    disp2_thread.join();
    disp3_thread.join();

    disp1_*=255./32;
    disp1_.convertTo(disp1_, CV_8UC3);
    cv::applyColorMap(disp1_, disp1_, cv::COLORMAP_JET);
    
    disp2_*=255./32;
    disp2_.convertTo(disp2_, CV_8UC3);
    cv::applyColorMap(disp2_, disp2_, cv::COLORMAP_JET);

    disp3_*=255./64;
    disp3_.convertTo(disp3_, CV_8UC3);
    cv::applyColorMap(disp3_, disp3_, cv::COLORMAP_JET);

    cv::imshow("left image", imgL_);
    cv::imshow("center image", imgC_); 
    cv::imshow("right image", imgR_); 
    cv::imshow("d1", disp1_); 
    cv::imshow("d2", disp2_);
    cv::imshow("d3", disp3_); 
    cv::waitKey(10);
}

int main(int argc, char **argv){
  set_rmap_calibration_matrices();

  ros::init(argc, argv, "monitor");
  ros::NodeHandle n;
  ros::Subscriber image_sub = n.subscribe("triple_img", 1, imageCallback);
  ros::Subscriber u_star_sub = n.subscribe("u_star", 1, inputCallback);

  ros::spin();

  

  return 0;
}
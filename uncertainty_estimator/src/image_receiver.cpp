#include "ros/ros.h"
#include "std_msgs/String.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/Twist.h>
#include "uncertainty_estimator/triple_image.h"

#include "cam_array.h"
#include "cbf.h"
#include "pointclouds.h"
#include "uncertainty_predictor.h"
#include "fstream" 

#define VISUALIZE
#define RECORD_DATA
int experiment_type; 
float lr;
// Data Logging
std::ofstream inputLog; 
std::ofstream pointsLog; 
cv::VideoWriter vidWriter; 



bool train_model;
int batch_size;

#ifdef DEBUG_LEARNING_STEPS
  bool learning_step_flag = false; 
#endif



// Controller Vars
std::vector<float> u_des{0, 0}; 
std::vector<float> u_star = {0, 0};
cv::Mat floatLeft, floatRight, floatDisparity;
Eigen::Tensor<float,3, Eigen::RowMajor> errorTensor_(img_height, frame_width, err_depth);
using UncPredType = UncertaintyPredictor<img_height, frame_width, err_depth>;
std::unique_ptr<UncPredType>  unc_pred_ptr;
using CVMatMap  = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

void inputDesCallback(const geometry_msgs::Twist::ConstPtr& msg){
    u_des[0] = msg->linear.x; 
    u_des[1] = msg->angular.z; 
    std::cout << "Recevied input" << u_des[0] << ", " << u_des[1] << std::endl;
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

    getD3hat(disp1_, disp2_, disp3hat_);

    disp3_thread.join();
    imgL_.convertTo(floatLeft, CV_32FC1);
    imgR_.convertTo(floatRight, CV_32FC1);
    if (experiment_type==1) {
        if (train_model) {
            floatDisparity = abs((disp3_ - disp3hat_));
            floatDisparity.convertTo(floatDisparity, CV_32FC1);
            if (batch_size == 1) {
              errorTensor_ = unc_pred_ptr->make_uncertainty_prediction_and_refine(
                      CVMatMap(floatLeft.ptr<float>(), img_height, frame_width),
                      CVMatMap(floatRight.ptr<float>(), img_height, frame_width),
                      CVMatMap(floatDisparity.ptr<float>(), img_height, frame_width)
              );
            }
            else if (batch_size > 1) {
              errorTensor_ = unc_pred_ptr->batch_make_uncertainty_prediction_and_refine(
                      CVMatMap(floatLeft.ptr<float>(), img_height, frame_width),
                      CVMatMap(floatRight.ptr<float>(), img_height, frame_width),
                      CVMatMap(floatDisparity.ptr<float>(), img_height, frame_width)
              );
            }
            else {
              throw std::invalid_argument("[ERROR] Invalid Batch Size of 0.");
            }
            #ifdef DEBUG_LEARNING_STEPS
              learning_step_flag = true; 
            #endif 
        } else {
            errorTensor_ = unc_pred_ptr->make_uncertainty_prediction(
                    CVMatMap(floatLeft.ptr<float>(), img_height, frame_width),
                    CVMatMap(floatRight.ptr<float>(), img_height, frame_width));
        }
    }
    get_max_err(errorTensor_, d3_maxErr_);

    if(experiment_type==0){
        disp3plusErr_ = disp3_;  // Points are closer (min h)
    }else if (experiment_type==1){
        disp3plusErr_ = disp3_ + d3_maxErr_;  // Points are closer (min h)
    }
    disp3minusErr_ = disp3_ - d3_maxErr_; // Points are farther away (max h)

    cv::reprojectImageTo3D(disp3plusErr_, cvWorstCaseCloud_, Q_); // points are closer
    cv::reprojectImageTo3D(disp3minusErr_, cvBestCaseCloud_, Q_); // points are further away

    // Get point clouds
    pclBestCaseCloud_ = cvMatToPcl(cvBestCaseCloud_); // 1 ms 
    pclWorstCaseCloud_ = cvMatToPcl(cvWorstCaseCloud_); // 1 ms 

    // Get the max h
    // Foreground Filter
    foreground_filter(pclBestCaseCloud_); 
    foreground_filter(pclWorstCaseCloud_); 

    float r_max_h; 
    int K_points; 
    if(experiment_type==0){
      r_max_h = 2;
      K_points = 1; 
    }else if (experiment_type==1){
      r_max_h = get_worst_case_r();
      K_points = 4000;//img_height*frame_width;
    }      

    // std::cout << "Worst case r = " << r_max_h << std::endl;

    std::vector<Eigen::Vector3f> points_safe_wrt; 
    float r_worstCaseClosest = get_K_closest_r(points_safe_wrt, r_max_h, K_points);

    // std::cout << "Worst case r2 = " << r_worstCaseClosest << std::endl;

    if(points_safe_wrt.size()>0){

      Eigen::Vector3f average=average_points_list(points_safe_wrt);

      if (turn==0){
        argmin_cbf_ecos_multiple_points(u_des, points_safe_wrt, u_star);
        // DATA LOGGING
        inputLog << get_current_time() << ','
                    << u_des[0] << ','
                    << u_des[1] << ','
                    << u_star[0] << ','
                    << u_star[1] << ','
                    <<  1.0/2*(pow(r_worstCaseClosest,2) - pow(C,2))  << "\n"; 
        std::cout << "No turning" << std::endl;
      }else{
        argmin_turn_cbf_ecos_multiple_points(u_des, points_safe_wrt, u_star); 
        // DATA LOGGING
        inputLog << get_current_time() << ','
                  << u_des[0] << ','
                  << u_des[1] << ','
                  << u_star[0] << ','
                  << u_star[1] << ','
                  <<  turnCBF(points_safe_wrt[0])  << "\n"; 
        std::cout << "Turning" << std::endl;
      }

     std::cout << "CBF value : h = " << 1.0/2*(pow(r_worstCaseClosest,2) - pow(C,2))  << std::endl;
     std::cout << "Average closest point z : " << points_safe_wrt[0][2]*39.37 << " in, x = " << points_safe_wrt[0][0]*39.37  <<  std::endl;
     std::cout << "Filtered u_des : v = " << u_star[0] << " w = " << u_star[1] << std::endl;



      std::string time_now = get_current_time();

      for (size_t i=0; i<points_safe_wrt.size(); i++){
        pointsLog << time_now << ',' << points_safe_wrt[i][0] << ',' << points_safe_wrt[i][1] <<  ',' << points_safe_wrt[i][2] << "\n"; 
      }

    }else{
        std::cout << "[WARNING]: empty pointcloud" << std::endl;
        std::string cmd = "walk_mpc_idqp(vx = "+std::to_string(0)+",vrz = " + std::to_string(0)+")";
        // std::cout << "Calling Python Command : "  << cmd.c_str() << std::endl;
        // PyRun_SimpleString(cmd.c_str()); 
        u_star[0] = u_des[0];
        u_star[1] = u_des[1];
    }

//    for( size_t slice_i(0); slice_i < err_depth; slice_i++){
//        view_slice(errorTensor_, slice_i);
//    }

      cv::Mat d3_diff;
      cv::Mat d3h_viewer;
      cv::Mat err_viewer;

      cv::absdiff(disp3_, disp3hat_, d3_diff); 
      d3_diff *=255./34; 
      cv::applyColorMap(d3_diff, d3_diff, cv::COLORMAP_JET);

      disp1_*=255./32;
      disp1_.convertTo(disp1_, CV_8UC3);
      cv::applyColorMap(disp1_, disp1_, cv::COLORMAP_JET);
      
      disp2_*=255./32;
      disp2_.convertTo(disp2_, CV_8UC3);
      cv::applyColorMap(disp2_, disp2_, cv::COLORMAP_JET);

      disp3_*=255./64;
      disp3_.convertTo(disp3_, CV_8UC3);
      cv::applyColorMap(disp3_, disp3_, cv::COLORMAP_JET);

      disp3hat_*=255./64;
      disp3hat_.convertTo(d3h_viewer, CV_8UC3);
      cv::applyColorMap(d3h_viewer, d3h_viewer, cv::COLORMAP_JET);

      d3_maxErr_*=255./err_depth;
      d3_maxErr_.convertTo(err_viewer, CV_8UC3);
      cv::applyColorMap(err_viewer, err_viewer, cv::COLORMAP_JET);

    #ifdef VISUALIZE

      cv::imshow("left image", imgL_);
      cv::imshow("center image", imgC_); 
      cv::imshow("right image", imgR_); 
      cv::imshow("d1", disp1_); 
      cv::imshow("d2", disp2_);
      cv::imshow("d3", disp3_); 
      cv::imshow("disp3hat", d3h_viewer); 
      cv::imshow("diff", d3_diff);
      cv::imshow("err_viewer", err_viewer);

      cv::waitKey(10);
    #endif 

    // DATA LOGGING

    cv::cvtColor(imgL_, imgL_,cv::COLOR_GRAY2BGR);
    cv::cvtColor(imgC_, imgC_,cv::COLOR_GRAY2BGR);
    cv::cvtColor(imgR_, imgR_,cv::COLOR_GRAY2BGR);

    cv::Mat matConcat[] = {imgL_, imgC_, imgR_, disp1_, disp2_, disp3_, d3_diff, d3h_viewer, err_viewer}; 
    cv::Mat stackToWrite; 
    cv::hconcat(matConcat, 9, stackToWrite);
    vidWriter.write(stackToWrite);


}

int main(int argc, char **argv){

// Data Logging
  auto t = std::time(nullptr); 
  auto tm = *std::localtime(&t); 
  std::ostringstream oss; 
  oss << std::put_time(&tm, "%Y_%m_%d_%H_%M_%S_");
  std::string log_path = "/home/drew/Aeronvironment/catkin_ws_rkc/src/uncertainty_estimator/bags/dataLogs/"+oss.str(); 

  inputLog.open(log_path+"inputs.csv");
  pointsLog.open(log_path+"points.csv");

  int codec = cv::VideoWriter::fourcc('M','J','P','G');
  std::string vid_path = log_path+"video.avi";
  std::cout << vid_path << std::endl;
  vidWriter.open(vid_path, codec, 10,cv::Size(9*320,200)); 

  // Initialize origins
  origin.x = 0;
  origin.y = 0;
  origin.z = 0;
  eigOrigin << 0,0,0; 

 initErrTensor(errorTensor_);

  set_rmap_calibration_matrices();

  ros::init(argc, argv, "image_receiver");
  ros::NodeHandle n;
  ros::param::param<int>("image_receiver/experiment_type", experiment_type,0);
  ros::param::param<bool>("image_receiver/train_model", train_model, false);
  ros::param::param<int>("image_receiver/batch_size", batch_size, 1);
  ros::param::param<float>("image_receiver/learning_rate", lr, 1e-3);
  ros::param::param<float>("image_receiver/error_threshold", err_threshold, 0.99);
  ros::param::param<int>("image_receiver/turn", turn, 0);

  unc_pred_ptr = std::make_unique<UncPredType>(lr, batch_size);

  if (experiment_type==0){
    ROS_INFO("Experiment type: Standard Barrier");
  }else if (experiment_type==1){
    ROS_INFO("Experiment type: Robust Barrier");
  }else{
    ROS_INFO("Uknown Experiment type. Running Standard Barrier");
    experiment_type=0;
  }
 if(train_model) {
     ROS_INFO("Training Model");
 }
 else{
     ROS_INFO("Not Training Model");
 }
  ROS_INFO("Learning rate: %e", lr);
  ROS_INFO("Error Threshold: %e", err_threshold);

  ros::Rate loop_rate(100);

  ros::Subscriber image_sub = n.subscribe("triple_img", 1, imageCallback);
  ros::Subscriber input_sub = n.subscribe("u_des", 1, inputDesCallback);
  ros::Publisher input_pub = n.advertise<geometry_msgs::Twist>("u_star", 1);
  
  #ifdef DEBUG
    ros::Publisher learning_step = n.advertise<geometry_msgs::Twist>("learning_step", 1); 
  #endif

  while(ros::ok()){
    geometry_msgs::Twist msg;
    msg.linear.x = u_star[0]; 
    msg.angular.z= u_star[1]; 
    input_pub.publish(msg); 
    ros::spinOnce();
    #ifdef DEBUG_LEARNING_STEPS
      if (learning_step_flag){
        learning_step.publish(msg); 
        learning_step_flag = false; 
      }
    #endif

    #ifdef DEBUG
      learning_step.publish(msg); 
    #endif 

    loop_rate.sleep(); 
  }

  inputLog.close();
  pointsLog.close();
  vidWriter.release();
  return 0;
}
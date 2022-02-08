#include "main.h"
#include "uncertainty_estimator/triple_image.h"
#include <geometry_msgs/Twist.h>


/* Video Reader Node ///////////////////////////////////////////////////////////////////////////

    Input: 

    Output: 
        - video stream over rostopic "triple_img"
        - desired input over rostopic "u_des"
*///////////////////////////////////////////////////////////////////////////////////////////////


/* Environment Variables ///////////////////////////////////////////////////////////////////////
    - RECORD        :   record video to .avi file
    - VISUALIZE     :   imshow the videos live on the jetson
*///////////////////////////////////////////////////////////////////////////////////////////////
// #define RECORD
// #define VISUALIZE 

/* Open CV Declarations ////////////////////////////////////////////////////////////////////////
    - cap           :   video capture object
    - deviceID      :   device ID on computer, example 0 inticates that the arducam is at /dev/video0
    - apiID         :   video api 
    - frame         :   read camera frame
    - frame_resized :   resized camera frame to reduce data transfer
    - imgL_         :   image from left camera
    - imgC_         :   image from center camera
    - imgR_         :   image from right camera
    - vidWriter     :   video writing object if recording data
    - future        :   asynchoronously result of the keystroke recording function GetLineFromCin
*///////////////////////////////////////////////////////////////////////////////////////////////
cv::VideoCapture cap; 
int deviceID = 0; 
int apiID = cv::CAP_V4L2; 
cv::Mat frame; 
cv::Mat frame_resized; 
cv::Mat imgL_; 
cv::Mat imgC_; 
cv::Mat imgR_;
#ifdef RECORD
    cv::VideoWriter vidWriter;
#endif 
auto future = std::async(std::launch::async, GetLineFromCin);


/* Desired Controller Variables ////////////////////////////////////////////////////////////////
    - input         :   desired input 
*///////////////////////////////////////////////////////////////////////////////////////////////
std::vector<float> input={0,0};




/* Helper Functions ////////////////////////////////////////////////////////////////////////////
    - kill_cv       :   release video writer (vidWriter) and close openCV windows
*///////////////////////////////////////////////////////////////////////////////////////////////
void kill_cv(){
    #ifdef RECORD
        vidWriter.release(); 
    #endif 
    cv::destroyAllWindows(); 
}

void mySigIntHandler(int sig)
{
    kill_cv();
    ros::shutdown();
}




int main(int argc, char **argv)
{
    auto t = std::time(nullptr); 
    auto tm = *std::localtime(&t); 
    
    #ifdef RECORD
        // Open Video Writing object
        std::ostringstream oss; 
        oss << std::put_time(&tm, "/home/amber/DataLogs/videos/%Y_%m_%d_%H_%M_%S_experiment_vid.avi");
        std::string vid_filename = oss.str(); 
        std::cout << "Writing Video to : " << vid_filename << std::endl; 
        int codec = cv::VideoWriter::fourcc('M','J','P','G');
        vidWriter.open(vid_filename, codec,20,cv::Size(img_width,img_height));
    #endif 
    
    /* Initialize videoReader ROS node 
        Publishers: 
            - image_pub     :   publish triple_img message of the three multi-baseline stereo images
            - u_des         :   publish the system desired input (forward velocity and angular rate)
    */
    ros::init(argc, argv, "videoReader");
    ros::NodeHandle n;
    ros::Publisher image_pub = n.advertise<uncertainty_estimator::triple_image>("triple_img", 1);
    ros::Publisher udes_pub = n.advertise<geometry_msgs::Twist>("u_des", 1);
    ros::Rate loop_rate(node_frequency);

    // Open video feed and set parameters
    cap.open(deviceID, apiID ); 
    cap.set(cv::CAP_PROP_CONVERT_RGB, convert2rgb); 
    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1); 
    // Verify that the video feed is connected
    if(cap.isOpened()){
        std::cout << "Opened Camera" << std::endl; 
    }else{ 
        std::cerr << "Error! Couldn't open camera" << std::endl;
        return -1; 
    }


    // Main Loop
    while (ros::ok())
    {
        /* Read commandline input once per loop with nonlocking thread 
                Set input using the convention: 
                        w   q                       v+=0.01     v+=0.05 
                    a   s   d           w+=0.01     v=w=0       w-=0.01
                        x                           v-=0.01
                Hit "k" to kill the CamArray which writes data and stops the quad
        */
        if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            auto line = future.get();
            future = std::async(std::launch::async, GetLineFromCin);
            if (line.compare("w")==0){
                input[0] +=0.01; 
            }else if (line.compare("s")==0){
                input[0] = 0.0;
                input[1] = 0.0;  
            }else if (line.compare("x")==0){
                input[0] -= 0.01;
            }else if (line.compare("a")==0){
                input[1] += 0.01; 
            }else if (line.compare("d")==0){
                input[1] -= 0.01; 
	    }else if (line.compare("q")==0){
		input[0] += 0.1; 
            }else if (line.compare("k")==0){
                std::cout << "CamArray killed, writing data and stopping quad.  \n\n"; 
                input[0] = 0; 
                input[1] = 0; 
                #ifdef RECORD_DATA
                    kill_cv();
                #endif 
                return 0; 
            }
            std::cout << "Inputs set to : \n \t v : " << input[0] << "\n \t w : " << input[1] << std::endl; 
        }

        /* Quick Image Processing 
                - read frame
                - convert to 8 bit
                - convert bayerBG2BGR 
                - downsize image
                - flip to put in robot perspective, not mirrored
                - alert if missing frame
                - record downsized color frame 
                - convert to gray
                - split into left, right, and center images
        */
        cap.read(frame); 
        cv::resize(frame, frame, cv::Size(), img_scale, img_scale); 
        cv::convertScaleAbs(frame, frame, 256.0/1024);
	cv::cvtColor(frame, frame, cvt_code); 
        cv::flip(frame, frame, 0 ); 
        if (frame.empty()){
            std::cerr << "Error! blank frame grabbed\n"; 
        }
        #ifdef RECORD
            vidWriter.write(frame);
        #endif 
        #ifdef VISUALIZE
            cv::imshow("Frame", frame); 
            cv::waitKey(1); 
        #endif
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        imgL_ = frame(cv::Rect(0,0,img_width/4, img_height));
        imgC_ = frame(cv::Rect(img_width/4,0,img_width/4, img_height));
        imgR_ = frame(cv::Rect(img_width/2,0,img_width/4, img_height));

        // Construct and sent the triple_image message
        uncertainty_estimator::triple_image msg; 
        sensor_msgs::ImagePtr imgLmsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", imgL_).toImageMsg();
        sensor_msgs::ImagePtr imgCmsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", imgC_).toImageMsg();
        sensor_msgs::ImagePtr imgRmsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", imgR_).toImageMsg();
        msg.header.stamp = ros::Time::now();
        msg.left = *imgLmsg; 
        msg.center = *imgCmsg;
        msg.right = *imgRmsg;
        image_pub.publish(msg);


        // Construct and send the input message
        geometry_msgs::Twist udes_msg;
        udes_msg.linear.x = input[0]; 
        udes_msg.angular.z= input[1]; 
        udes_pub.publish(udes_msg);

        // Wait for the next loop 
        loop_rate.sleep();
    }


    return 0;
}

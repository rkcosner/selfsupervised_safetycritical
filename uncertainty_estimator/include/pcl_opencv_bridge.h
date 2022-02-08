#include <iostream>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree_search.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

typedef pcl::PointXYZ myPoint;



//  PCl <--> OpenCV Conversions
pcl::PointCloud<pcl::PointXYZ>::Ptr cvMatToPcl(const cv::Mat &mat) {
    auto cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    // std::cout << "begin parsing file" << std::endl;
    for (int ki = 0; ki < mat.rows; ki++) {
        for (int kj = 0; kj < mat.cols; kj++) {
            pcl::PointXYZ pointXYZ;

            pointXYZ.x = mat.at<cv::Point3f>(ki, kj).x;
            pointXYZ.y = mat.at<cv::Point3f>(ki, kj).y;
            pointXYZ.z = mat.at<cv::Point3f>(ki, kj).z;

            if(pointXYZ.z <= 0)
                continue;
            cloud->points.push_back(pointXYZ);
        }

    }
    return cloud;
}

cv::Mat PoinXYZToMat(pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr){

    cv::Mat OpenCVPointCloud(3, point_cloud_ptr->points.size(), CV_64FC1);
    for(size_t i=0; i < point_cloud_ptr->points.size();i++){
        OpenCVPointCloud.at<double>(0,i) = point_cloud_ptr->points.at(i).x;
        OpenCVPointCloud.at<double>(1,i) = point_cloud_ptr->points.at(i).y;
        OpenCVPointCloud.at<double>(2,i) = point_cloud_ptr->points.at(i).z;
    }

    return OpenCVPointCloud;
}


#ifdef COLOR
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cvMatToPclColor(cv::Mat &mat, cv::Mat &img) {

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = boost::shared_ptr<pcl::PointCloud<pcl::PointXYZRGB>>(
                new pcl::PointCloud<pcl::PointXYZRGB>);
        // std::cout << "begin parsing file" << std::endl;
        for (int ki = 0; ki < mat.rows; ki++) {
            for (int kj = 0; kj < mat.cols; kj++) {
                pcl::PointXYZRGB pointXYZRGB;

                pointXYZRGB.x = mat.at<cv::Point3f>(ki, kj).x;
                pointXYZRGB.y = mat.at<cv::Point3f>(ki, kj).y;
                pointXYZRGB.z = mat.at<cv::Point3f>(ki, kj).z;

                if(pointXYZRGB.z <= 0)
                    continue;

                cv::Vec3b color = img.at<cv::Vec3b>(cv::Point(ki, kj));
                uint8_t r = (color[2]);
                uint8_t g = (color[1]);
                uint8_t b = (color[0]);


                int32_t rgb = (r << 16) | (g << 8) | b;
                pointXYZRGB.rgb = *reinterpret_cast<float*>(&rgb);
                if(pointXYZRGB.z <= 0)
                    continue;
                cloud->points.push_back(pointXYZRGB);
            }

        }
        return cloud;
    }
#endif
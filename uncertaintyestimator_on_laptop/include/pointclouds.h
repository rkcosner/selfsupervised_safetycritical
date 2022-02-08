#include <opencv4/opencv2/opencv.hpp>
#include "pcl_opencv_bridge.h"

cv::Mat cvCloud_; 
cv::Mat cvWorstCaseCloud_; 
cv::Mat cvBestCaseCloud_; 


// PCL Initializers
pcl::PointCloud<myPoint>::Ptr pclCloud_ (new pcl::PointCloud<myPoint>);
pcl::PointCloud<myPoint>::Ptr pclWorstCaseCloud_ (new pcl::PointCloud<myPoint>);
pcl::PointCloud<myPoint>::Ptr pclBestCaseCloud_ (new pcl::PointCloud<myPoint>);

// PCL Filtering
pcl::VoxelGrid<myPoint> sor_;
pcl::StatisticalOutlierRemoval<myPoint> stat_;
pcl::PassThrough<myPoint> pass_;
pcl::PassThrough<myPoint> passY_;


float factor1;
float factor2;  



// Comparison Points
myPoint origin;
Eigen::Vector3f eigOrigin; 


float get_worst_case_r(){
    float r_worst; 
    if (pclBestCaseCloud_->size()>0){
        pcl::KdTreeFLANN<myPoint> kdtree_r_max_h;
        kdtree_r_max_h.setInputCloud(pclBestCaseCloud_);

        std::vector<int>  pointIdxKNNSearch(1); 
        std::vector<float> pointKNNSquaredDistance(1); 


        if ( kdtree_r_max_h.nearestKSearch (origin, 1, pointIdxKNNSearch, pointKNNSquaredDistance) > 0 )
        {
            for (std::size_t i = 0; i < pointIdxKNNSearch.size (); ++i)
                r_worst = pointKNNSquaredDistance[i];
        }
    }else{
        r_worst = 10; 
        std::cout << "[WARNING]: Empty Pointcloud bestcase\n "; 
    }

    return r_worst;

}

float get_K_closest_r(std::vector<Eigen::Vector3f> &points_safe_wrt, float r_hmax, int K){
    float r;
    if (pclWorstCaseCloud_->size()>0){
        pcl::KdTreeFLANN<myPoint> kdtree_r_max_h2;
        kdtree_r_max_h2.setInputCloud(pclWorstCaseCloud_);

        std::vector<int>  pointIdxKNNSearch2(K); 
        std::vector<float> pointKNNSquaredDistance2(K); 
        if ( kdtree_r_max_h2.nearestKSearch (origin, K, pointIdxKNNSearch2, pointKNNSquaredDistance2) > 0 )
        {
            for (std::size_t i = 0; i < pointIdxKNNSearch2.size (); ++i){
                if (pointKNNSquaredDistance2[i] < r_hmax){
                    Eigen::Vector3f add_point; 
                    add_point << (*pclWorstCaseCloud_)[ pointIdxKNNSearch2[i] ].x , (*pclWorstCaseCloud_)[ pointIdxKNNSearch2[i] ].y, (*pclWorstCaseCloud_)[ pointIdxKNNSearch2[i] ].z; 
                    points_safe_wrt.push_back(add_point); 
                    r = sqrt(pointKNNSquaredDistance2[i]); 
                }
            }
        }
        return r; 
    }else{
        std::cout << "[WARNING]: Empty Pointcloud worstcase \n "; 
        return 10; 
    }
}


void foreground_filter(const pcl::PointCloud<myPoint>::Ptr &cloud){
    pass_.setInputCloud(cloud); // 1 ms 
    pass_.setFilterFieldName("z");
    pass_.setFilterLimits(0.0, 2);///0.33);
    pass_.filter(*cloud);

    pass_.setInputCloud(cloud); // 1 ms 
    pass_.setFilterFieldName("y");
    pass_.setFilterLimits(-0.2, 1);//1);
    pass_.filter(*cloud);
}

Eigen::Vector3f average_points_list(const std::vector<Eigen::Vector3f> &points_safe_wrt){
    Eigen::Vector3f ave=eigOrigin;
    int denom = 0;  
    for (size_t m_idx=0; m_idx<points_safe_wrt.size(); ++m_idx){
        denom++; 
        ave+=points_safe_wrt[m_idx]; 
    }
    if (denom == 0){
        ave = eigOrigin + Eigen::Matrix3Xf::Ones(3,1)*100; 
    }else{
        ave /=denom; 
    }
    return ave;
}
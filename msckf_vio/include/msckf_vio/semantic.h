#ifndef MSCKF_VIO_SEMANTIC_H
#define MSCKF_VIO_SEMANTIC_H

#include <vector>
#include <queue>
#include <mutex>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
// #include <opencv2/dnn.hpp>

#include <mutex>
#include <condition_variable>
#include <msckf_vio/CameraMeasurement.h>
// #include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>

#include <map>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace cv;
using namespace dnn;
using namespace Ort;

namespace msckf_vio{

class Semantic
{
public:
    Semantic(ros::NodeHandle &n);
    Semantic(const Semantic &) = delete;
    Semantic operator = (const Semantic &) = delete;

    ~Semantic() {}
    bool initialize();

    // YOLOv5
    typedef boost::shared_ptr<Semantic> Ptr;
    typedef boost::shared_ptr<const Semantic> ConstPtr;

private:
    bool loadParameters();

    bool createRosIO();

    void publish();
    
    // Callback function.
    void feature_Callback(const CameraMeasurementConstPtr& msg);

    void image_Callback(const sensor_msgs::ImageConstPtr& cam0_img);

    // Detect function
    bool Detect();

    void UndistortFeaturePoints(std::vector<cv::Point2f>& feature_points);

    void undistortPoints(
        const std::vector<cv::Point2f>& pts_in,
        const cv::Vec4d& intrinsics,
        const std::string& distortion_model,
        const cv::Vec4d& distortion_coeffs,
        std::vector<cv::Point2f>& pts_out,
        const cv::Matx33d &rectification_matrix = cv::Matx33d::eye(),
        const cv::Vec4d &new_intrinsics = cv::Vec4d(1,1,0,0));

    // Publish function.
    void publish(const CameraMeasurementPtr& msg);
    void drawFeaturesStereo();
    //
    std::queue<CameraMeasurementPtr> feature_queue_;
    std::queue<sensor_msgs::ImageConstPtr> image_queue;
    sensor_msgs::ImageConstPtr image_ptr,image_ptr_2;
    std::mutex mutex_;
    typedef unsigned long long int FeatureIDType;

    // CameraMeasurementPtr feature_ptr_(new CameraMeasurement);

    // FeaturesPtrVector  
    // std::vector<CameraMeasurementConstPtr> feature_ptr_vector;

    // Rgb images
    cv_bridge::CvImageConstPtr cv_ptr;
    cv_bridge::CvImageConstPtr cv_ptr1;
    // Ros node handle
    ros::NodeHandle nh;

    // Subscribers and publishers.
    ros::Subscriber feature_sub;
    ros::Subscriber cam0_rgb_img_sub;
    ros::Publisher image_pub; 
    ros::Publisher feature_pub;

    image_transport::Publisher debug_stereo_pub;    
    
    // YOLOv5
    float Sigmoid(float x){
        return static_cast<float>(1.f / 1.f + exp(-x));
    }
    //anchors
	const float netAnchors[3][6] = { { 10.0, 13.0, 16.0, 30.0, 33.0, 23.0 },{ 30.0, 61.0, 62.0, 45.0, 59.0, 119.0 },{ 116.0, 90.0, 156.0, 198.0, 373.0, 326.0 } };
    //stride
	const float netStride[3] = { 8.0, 16.0, 32.0 };
	const int netWidth = 640; //网络模型输入大小
	const int netHeight = 640;
	float nmsThreshold = 0.45; 
	float boxThreshold = 0.35;
	float classThreshold = 0.80;
    std::vector<std::string> className = {"Car", "Pedestrian", "Truck", "Van", "Cyclist", "Tram"};
    uint8_t is_first_img;
    // 
    struct Output{
        int id;
        float confidence;
        cv::Rect box;
    };
    
    std::vector<Output> output;
    cv::dnn::Net net;
    std::string netPath;

};

typedef Semantic::Ptr SemanticPtr;
typedef Semantic::ConstPtr SemanticConstPtr;

}

#endif
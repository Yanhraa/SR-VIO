/**
 * @brief 用于动态点去除
 * @date 2023/03/07
 * 
*/
#include <iostream>

#include <msckf_vio/semantic.h> 
namespace msckf_vio
{

Semantic::Semantic(ros::NodeHandle &n) : 
nh(n),
is_first_img(0)
{   
    return;
}

/**
Semantic::~Semantic()
{
    return;
}
*/

bool Semantic::loadParameters()
{
    nh.param<std::string>("net_Path", netPath, "/home/r/src/msckf_ws/src/msckf_vio/Thirdparty/yolo_model/best.onnx");
    net = cv::dnn::readNetFromONNX(netPath);

    if (net.empty()) {
        // std::cerr << "Error: Could not read the ONNX model." << std::endl;
        ROS_WARN("Filed to readNetFromONNX.");
        return false;
    }

    ROS_INFO("ONNX model loaded successfully.");

    // cuda
    bool isCuda = false;
    if(isCuda){ //gpu
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }else{  //cpu
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    }
    return true;
}


bool Semantic::createRosIO()
{
    feature_sub = nh.subscribe("features", 10, &Semantic::feature_Callback, this);
    cam0_rgb_img_sub = nh.subscribe("cam0_rgb_image", 10, &Semantic::image_Callback, this);

    image_pub = nh.advertise<sensor_msgs::Image>("/detected_image", 1);
    feature_pub = nh.advertise<CameraMeasurement>("features_", 10);

    image_transport::ImageTransport it(nh);
    debug_stereo_pub = it.advertise("debug_stereo_image", 10);

    return true;
}

bool Semantic::initialize()
{
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    if(!loadParameters()) return false;
    ROS_INFO("Finish loading Semantic node parameters.");

    if(!createRosIO()) return false;
    ROS_INFO("Finish loading Semantic node IO.");

    return true;
}

void Semantic::feature_Callback(const CameraMeasurementConstPtr& msg)
{

    // ROS_INFO("*Receive feature ptr.");
    std::lock_guard<std::mutex> lock_(mutex_);
    CameraMeasurementPtr temp_ptr(new CameraMeasurement(*msg));   
    feature_queue_.push(temp_ptr);

    if(!image_queue.empty()) {
        Detect();
    }

}

void Semantic::image_Callback
(const sensor_msgs::ImageConstPtr& cam0_img
)
{
    image_queue.push(cam0_img);
   
}

bool Semantic::Detect()
{

    cv::Vec4d intrinsics(7.070493e+02, 7.070493e+02, 6.040814e+02, 1.805066e+02); 
    std::string distortion_model = "radtan"; 
    cv::Vec4d distortion_coeffs (0.0, 0.0, 0.0, 0.0); 
    output.clear(); 

    image_ptr = image_queue.front();
    image_queue.pop();
    cv_ptr = cv_bridge::toCvShare(image_ptr, sensor_msgs::image_encodings::RGB8);
    cv_bridge::CvImagePtr cv_ptr1 = cv_bridge::toCvCopy(image_ptr, sensor_msgs::image_encodings::MONO8);
    cv::Mat image = cv_ptr -> image;
    cv::Mat blob;
    int col = image.cols;
    int row = image.rows;
    int maxLen = MAX(col, row);
    cv::Mat net_input_img = image.clone();

    if (maxLen > 1.2 * col || maxLen > 1.2 * row){
        cv::Mat resize_image = cv::Mat::zeros(maxLen, maxLen, CV_8UC3);
        image.copyTo(resize_image(cv::Rect(0, 0, col, row)));
        net_input_img = resize_image;
    }

    cv::dnn::blobFromImage(net_input_img, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(104, 117,123), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> net_output_img;
    net.forward(net_output_img, net.getUnconnectedOutLayersNames());
    std::vector<int> classIds; 
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    float ratio_h = (float)net_input_img.rows / netHeight;
	float ratio_w = (float)net_input_img.cols / netWidth;
	int net_width = className.size() + 5;  //输出的网络宽度是类别数+5
	float* pdata = (float*)net_output_img[0].data;

	for (int stride = 0; stride < 3; stride++) {    //stride
		int grid_x = (int)(netWidth / netStride[stride]);
		int grid_y = (int)(netHeight / netStride[stride]);
		for (int anchor = 0; anchor < 3; anchor++) { //anchors
			const float anchor_w = netAnchors[stride][anchor * 2];
			const float anchor_h = netAnchors[stride][anchor * 2 + 1];
			for (int i = 0; i < grid_y; i++) {
				for (int j = 0; j < grid_y; j++) {
					float box_score = pdata[4]; 
					if (box_score > boxThreshold) {
						cv::Mat scores(1,className.size(), CV_32FC1, pdata+5);
						Point classIdPoint;
						double max_class_socre;
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						max_class_socre = (float)max_class_socre;
						if (max_class_socre > classThreshold) {

							float x = pdata[0]; 
							float y = pdata[1];
							float w = pdata[2];
							float h = pdata[3];

							int left = (x - 0.5*w)*ratio_w;
							int top = (y - 0.5*h)*ratio_h;
							classIds.push_back(classIdPoint.x);
							confidences.push_back(max_class_socre*box_score);
							boxes.push_back(Rect(left, top, int(w*ratio_w), int(h*ratio_h)));
						}
					}
					pdata += net_width;
				}
			}
		}
	}

    std::vector<int> nms_result;
	NMSBoxes(boxes, confidences, classThreshold, nmsThreshold, nms_result);
	for (int i = 0; i < nms_result.size(); i++) {
		int idx = nms_result[i];
		Output result;
		result.id = classIds[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		output.push_back(result);
	}

    CameraMeasurementPtr feature_ptr = feature_queue_.front();
    feature_queue_.pop();
    CameraMeasurementPtr feature_ptr_(new CameraMeasurement);

    cv::Mat detected_image = image.clone();
    for (const auto& result : output) {
    cv::rectangle(detected_image, result.box, cv::Scalar(255, 0, 0), 2);
    cv::putText(detected_image, className[result.id-1], cv::Point(result.box.x, result.box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);

    }

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", detected_image).toImageMsg();

    if(!output.empty()){
    ROS_INFO("feature_ptr->header.stamp: %f", feature_ptr->header.stamp.toSec());
    ROS_INFO("image_ptr->header.stamp: %f", image_ptr->header.stamp.toSec());
        // ROS_INFO("Detected dynamic object.");   
        //去除特征点
        for(const auto& feature : feature_ptr->features){
            bool isInsideDynamicObject = false;  // 是否在动态目标中的标志位
            for(const auto& object : output){
                std::vector<cv::Point2f> pts_out;
                std::vector<cv::Point2f> pts_in = {
                    cv::Point2f(object.box.x, object.box.y),
                    cv::Point2f(object.box.x + object.box.width, object.box.y),
                    cv::Point2f(object.box.x, object.box.y + object.box.height),
                    cv::Point2f(object.box.x + object.box.width, object.box.y + object.box.height)
                };
                undistortPoints(pts_in, intrinsics, distortion_model, distortion_coeffs, pts_out);

                if(feature.u0 >= pts_out[0].x && feature.u0 <= pts_out[3].x &&
                    feature.v0 >= pts_out[0].y && feature.v0 <= pts_out[3].y) {   
                    isInsideDynamicObject = true;
                    break;
                }
            }
            if(!isInsideDynamicObject){
                feature_ptr_->header.stamp = feature_ptr->header.stamp;
                feature_ptr_->features.push_back(feature);
            }
        }

        feature_pub.publish(feature_ptr_);
        image_pub.publish(msg);

    }
    else{
        feature_pub.publish(feature_ptr);

    }

    if(debug_stereo_pub.getNumSubscribers() > 0)
    {
        Scalar tracked(0, 255, 0);
        Scalar new_feature(0, 255, 255);

        int img_height = cv_ptr1->image.rows;
        int img_width = cv_ptr1->image.cols;

        Mat out_img(img_height, img_width, CV_8UC3);
        if(cv_ptr1->image.channels() == 1) {
            cvtColor(cv_ptr1->image, out_img, CV_GRAY2BGR);
        } else {
            out_img = cv_ptr1->image.clone();
        }
        // Draw each feature point on the image
        for(const auto& feature : feature_ptr->features) {
            cv::Point2f normalized_pt(feature.u0, feature.v0);
            cv::Point2f pt;
            pt.x = (normalized_pt.x * intrinsics[0]) + intrinsics[2];
            pt.y = (normalized_pt.y * intrinsics[1]) + intrinsics[3];
            circle(out_img, pt, 3, tracked, -1);
        
        }

        // cv_bridge::CvImage debug_image(cv_ptr1->header, "rgb8", out_img);
        imshow("debug_image", out_img);
        waitKey(5);
    }

    return true;
    
}


void Semantic::publish(const CameraMeasurementPtr& msg)
{
    feature_pub.publish(msg);
    ROS_INFO("Finished publish.");
}

void Semantic::undistortPoints(
    const std::vector<cv::Point2f>& pts_in,
    const cv::Vec4d& intrinsics,
    const std::string& distortion_model,
    const cv::Vec4d& distortion_coeffs,
    std::vector<cv::Point2f>& pts_out,
    const cv::Matx33d &rectification_matrix,
    const cv::Vec4d &new_intrinsics) {

  if (pts_in.size() == 0) return;

  const cv::Matx33d K(
      intrinsics[0], 0.0, intrinsics[2],
      0.0, intrinsics[1], intrinsics[3],
      0.0, 0.0, 1.0);

  const cv::Matx33d K_new(
      new_intrinsics[0], 0.0, new_intrinsics[2],
      0.0, new_intrinsics[1], new_intrinsics[3],
      0.0, 0.0, 1.0);

  if (distortion_model == "radtan") {
    cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                        rectification_matrix, K_new);
  } else if (distortion_model == "equidistant") {
    cv::fisheye::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                                 rectification_matrix, K_new);
  } else {
    ROS_WARN_ONCE("The model %s is unrecognized, use radtan instead...",
                  distortion_model.c_str());
    cv::undistortPoints(pts_in, pts_out, K, distortion_coeffs,
                        rectification_matrix, K_new);
  }

  return;
}


}

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#if ROS_VERSION_MINIMUM(1,4,5)
    #include <cv_bridge/cv_bridge.h>
#else
    #include <cv_bridge/CvBridge.h>
#endif

#include "vision_msgs/TrainingData.h"

namespace enc = sensor_msgs::image_encodings;

/**
* \class TrainerNode
* \author Stephan Wirth
* \brief node that provides a training gui
* This node displays incoming image messages using OpenCV. The image stream
* can be paused to draw a polygon on top of the still image. Then the image
* together with the polygon can be sent as a object training message to the
* ROS system.
*/
class TrainerNode
{
public:
    enum Mode
    {
        DISPLAY_VIDEO,
        AWAITING_TRAINING_IMAGE,
        SHOWING_TRAINING_IMAGE,
        PAINTING
    };

    TrainerNode() : it_(nh_), current_mode_(DISPLAY_VIDEO)
    {
        image_sub_ = it_.subscribe("image", 1, &TrainerNode::imageCallback, this);
        training_data_pub_ = nh_.advertise<vision_msgs::TrainingData>("training_data", 1);

        cv::namedWindow("Training GUI");
        cv::setMouseCallback("Training GUI", &TrainerNode::staticMouseCallback, this);

        loop_timer_ = nh_.createTimer(ros::Duration(0.05), &TrainerNode::processEvents, this);
    }

    ~TrainerNode()
    {
        cv::destroyWindow("Training GUI");
    }

private:

    void imageCallback(const sensor_msgs::ImageConstPtr& image_msg)
    {
        cv::Mat image;

#if ROS_VERSION_MINIMUM(1,4,5)
        cv_bridge::CvImageConstPtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvShare(image_msg, enc::BGR8);
            image = cv_ptr->image.clone();
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
#else
        try
        {
            IplImage *cv_image = NULL;
            cv_image = bridge_.imgMsgToCv(image_msg, "bgr8");
            image = cv::cvarrToMat(cv_image).clone();
        }
        catch (sensor_msgs::CvBridgeException& e)
        {
            ROS_ERROR("CvBridgeException: %s", e.what());
            return;
        }
#endif
        if (current_mode_ == DISPLAY_VIDEO)
        {
           cv::imshow("Training GUI", image);
        }
        else if (current_mode_ == AWAITING_TRAINING_IMAGE)
        {
            training_image_ = image.clone();
            cv::imshow("Training GUI", training_image_);
            current_mode_ = PAINTING;
            ROS_INFO("Entered painting mode, waiting for user input.");
        }
    }

    static void staticMouseCallback(int event, int x, int y, int flags, void* param)
    {
        // extract this pointer and call function on object
        TrainerNode* node = reinterpret_cast<TrainerNode*>(param);
        node->mouseCallback(event, x, y, flags);
    }

    void mouseCallback(int event, int x, int y, int flags)
    {
        if (current_mode_ == PAINTING)
        {
            current_mouse_position_ = cv::Point(x, y);
            if (event == CV_EVENT_LBUTTONUP)
            {
                ROS_INFO("Adding point (%i,%i) to polygon.", x, y);
                polygon_points_.push_back(current_mouse_position_);
            }
        }
    }

    void processEvents(const ros::TimerEvent&)
    {
        char key = cv::waitKey(3);
        if (key == ' ' && current_mode_ == DISPLAY_VIDEO)
        {
            current_mode_ = AWAITING_TRAINING_IMAGE;
            ROS_INFO("Space pressed, using next arriving image as training image.");
        }
        else if (key == ' ' && current_mode_ == PAINTING)
        {
            // unsubscribe from image topic as we dont need it now
            image_sub_.shutdown();
            if (polygon_points_.size() > 2)
            {
                publishTrainingData(training_image_, polygon_points_, "object1");
            }
            current_mode_ = SHOWING_TRAINING_IMAGE;
            cv::Mat image_with_polygon = training_image_.clone();
            const cv::Point* point_data = polygon_points_.data();
            int num_points = polygon_points_.size();
            bool closed = true;
            cv::polylines(image_with_polygon, &point_data, &num_points, 
                    1, closed, cv::Scalar(0, 255, 0), 2);
            cv::imshow("Training GUI", image_with_polygon);
        }
        else if (key == ' ' && current_mode_ == SHOWING_TRAINING_IMAGE)
        {
            ROS_INFO("Entering display video mode.");
            current_mode_ = DISPLAY_VIDEO;
            polygon_points_.clear();
            // subscribe to image topic again
            image_sub_ = it_.subscribe("image", 1, &TrainerNode::imageCallback, this);
        }
        else if (key < 0)
        {
            // no key was pressed
            if (current_mode_ == PAINTING)
            {
                if (polygon_points_.size() > 0)
                {
                    std::vector<cv::Point> painting_polygon_points = polygon_points_;
                    painting_polygon_points.push_back(current_mouse_position_);
                    cv::Mat image_with_polygon = training_image_.clone();
                    const cv::Point* point_data = painting_polygon_points.data();
                    int num_points = painting_polygon_points.size();
                    bool closed = true;
                    cv::polylines(image_with_polygon, &point_data, &num_points, 
                            1, closed, cv::Scalar(0, 255, 0), 1);
                    cv::imshow("Training GUI", image_with_polygon);
                }
            }
        }
    }

    void publishTrainingData(const cv::Mat& image, 
            const std::vector<cv::Point> polygon_points,
            const std::string& object_name)
    {
        // prepare polygon
        vision_msgs::Polygon2D polygon;
        for (size_t i = 0; i < polygon_points.size(); ++i)
        {
            vision_msgs::Point2D point;
            point.x = polygon_points[i].x;
            point.y = polygon_points[i].y;
            polygon.points.push_back(point);
        }

        // prepare object description
        vision_msgs::ObjectDescription object_description;
        object_description.id = "object1";
        object_description.outline = polygon;

        vision_msgs::TrainingData training_data;

#if ROS_VERSION_MINIMUM(1,4,5)
        // prepare image as message
        cv_bridge::CvImage cv_image;
        cv_image.image = image;
        cv_image.encoding = enc::BGR8;
        try
        {
            cv_image.toImageMsg(training_data.image);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
#else
        try
        {
            // we have to use depricated fromIpltoRosImage() here because
            // we do not need a boost::shared_ptr
            IplImage* ipl_image = new IplImage(image);
            if (!bridge_.fromIpltoRosImage(ipl_image, training_data.image, "bgr8"))
            {
                throw sensor_msgs::CvBridgeException("Conversion to ROS image failed!");
            }
        }
        catch (sensor_msgs::CvBridgeException& e)
        {
            ROS_ERROR("CvBridgeException: %s", e.what());
        }
#endif
        training_data.object_description = object_description;
        training_data_pub_.publish(training_data);

        ROS_INFO("Training message published.");
    }

    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Publisher training_data_pub_;

    cv::Mat training_image_;
    std::vector<cv::Point> polygon_points_;
    Mode current_mode_;

    ros::Timer loop_timer_;
    cv::Point current_mouse_position_;

#if ROS_VERSION_MINIMUM(1,4,5)
#else
    sensor_msgs::CvBridge bridge_;
#endif


};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "traininer_node");
  TrainerNode trainer;
  ros::spin();
  return 0;
}


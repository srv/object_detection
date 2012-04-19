
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cv_bridge/cv_bridge.h>

#include "odat/training_data.h"
#include "odat_ros/conversions.h"
#include "object_detection/shape_processing.h"
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
        PAINTING,
        SELECTING_ORIGIN,
        SELECTING_DIRECTION
    };

    TrainerNode() : nh_(), nh_priv_("~"), it_(nh_), current_mode_(DISPLAY_VIDEO)
    {
        image_sub_ = it_.subscribe("image", 1, &TrainerNode::imageCallback, this);
        training_data_pub_ = nh_priv_.advertise<vision_msgs::TrainingData>("training_data", 1);

        nh_priv_.param("object_id", object_id_, std::string("target"));

        cv::namedWindow("Training GUI");
        cv::setMouseCallback("Training GUI", &TrainerNode::staticMouseCallback, this);

        loop_timer_ = nh_.createTimer(ros::Duration(0.05), &TrainerNode::repaint, this);

        object_pose_.x = 0;
        object_pose_.y = 0;
        object_pose_.theta = 0;

        ROS_INFO("Trainig object '%s'.", object_id_.c_str());
        ROS_INFO("Click left to select a training image.");
    }

    ~TrainerNode()
    {
        cv::destroyWindow("Training GUI");
    }

private:

    void imageCallback(const sensor_msgs::ImageConstPtr& image_msg)
    {
        cv::Mat image;

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
        if (current_mode_ == DISPLAY_VIDEO)
        {
           cv::imshow("Training GUI", image);
           cv::waitKey(5);
        }
        else if (current_mode_ == AWAITING_TRAINING_IMAGE)
        {
            training_image_ = image.clone();
            cv::imshow("Training GUI", training_image_);
            cv::waitKey(5);
            current_mode_ = PAINTING;
            ROS_INFO("Entered painting mode, click left to select an object polygon. Click right when finished.");
        }
    }

    static void staticMouseCallback(int event, int x, int y, int flags, void* param)
    {
        // extract this pointer and call function on object
        TrainerNode* node = reinterpret_cast<TrainerNode*>(param);
        assert(node != NULL);
        node->mouseCallback(event, x, y, flags);
    }

    void mouseCallback(int event, int x, int y, int flags)
    {
        current_mouse_position_ = cv::Point(x, y);
        if (current_mode_ == DISPLAY_VIDEO)
        {
            if (event == CV_EVENT_LBUTTONUP)
            {
                current_mode_ = AWAITING_TRAINING_IMAGE;
                ROS_INFO("Taking next arriving image as training image...");
            }
        } 
        else if (current_mode_ == PAINTING)
        {
            if (event == CV_EVENT_LBUTTONUP)
            {
                ROS_INFO("Adding point (%i,%i) to polygon.", x, y);
                polygon_points_.push_back(current_mouse_position_);
            }
            else if (event == CV_EVENT_RBUTTONUP)
            {
                if (polygon_points_.size() > 2)
                {
                    current_mode_ = SELECTING_ORIGIN;
                    image_sub_.shutdown();
                    object_pose_.x = x;
                    object_pose_.y = y;
                    ROS_INFO("Select the object origin and click.");
                }
                else
                {
                    ROS_INFO("Not enough points in polygon, resetting.");
                    polygon_points_.clear();
                }
            }
        }
        else if (current_mode_ == SELECTING_ORIGIN)
        {
            object_pose_.x = x;
            object_pose_.y = y;
            if (event == CV_EVENT_LBUTTONUP)
            {
              ROS_INFO("Setting point (%i,%i) as origin.", x, y);
              ROS_INFO("Select direction and click. (red = x, green = y)");
              current_mode_ = SELECTING_DIRECTION;
            }
        }
        else if (current_mode_ == SELECTING_DIRECTION)
        {
            object_pose_.theta = atan2(current_mouse_position_.y - object_pose_.y,
                current_mouse_position_.x - object_pose_.x);
            if (event == CV_EVENT_LBUTTONUP)
            {
              current_mode_ = SHOWING_TRAINING_IMAGE;
              publishTrainingData(training_image_, polygon_points_, object_pose_);
            }
        }
        else if (current_mode_ == SHOWING_TRAINING_IMAGE)
        {
            if (event == CV_EVENT_LBUTTONUP)
            {
                current_mode_ = DISPLAY_VIDEO;
                ROS_INFO("Entering display video mode.");
                polygon_points_.clear();
                object_pose_.x = 0;
                object_pose_.y = 0;
                object_pose_.theta = 0;
                // subscribe to image topic again
                image_sub_ = it_.subscribe("image", 1, &TrainerNode::imageCallback, this);
            }
        }
    }

    void repaint(const ros::TimerEvent&)
    {
        if (current_mode_ != DISPLAY_VIDEO && current_mode_ != AWAITING_TRAINING_IMAGE)
        {
            std::vector<cv::Point> painting_polygon_points = polygon_points_;
            if (current_mode_ == PAINTING)
            {
                painting_polygon_points.push_back(current_mouse_position_);
            }
            cv::Mat canvas = training_image_.clone();

            if (painting_polygon_points.size() > 1)
            {
                const cv::Point* point_data = painting_polygon_points.data();
                int num_points = painting_polygon_points.size();
                bool closed = true;
                cv::polylines(canvas, &point_data, &num_points, 
                        1, closed, cv::Scalar(0, 255, 0), 1);
            }
            // coordinate system
            if (object_pose_.x != 0 && object_pose_.y != 0)
            {
              cv::Point origin(object_pose_.x, object_pose_.y);
              double direction = object_pose_.theta;
              cv::Point x_axis(50 * cos(direction), 50 * sin(direction));
              cv::Point y_axis(50 * cos(direction + M_PI_2), 50 * sin(direction + M_PI_2));
              cv::line(canvas, origin, origin + x_axis, cv::Scalar(0, 0, 255), 2);
              cv::line(canvas, origin, origin + y_axis, cv::Scalar(0, 255, 0), 2);
            }
            cv::imshow("Training GUI", canvas);
            cv::waitKey(5);
        }
    }

    void publishTrainingData(const cv::Mat& image, 
            const std::vector<cv::Point> polygon_points,
            const odat::Pose2D& object_pose)
    {
        odat::TrainingData training_data;
        training_data.image = image;
        training_data.mask.roi = object_detection::shape_processing::boundingRect(polygon_points);
        training_data.mask.mask = object_detection::shape_processing::minimalMask(polygon_points);

        training_data.image_pose = object_pose;

        vision_msgs::TrainingData training_data_msg;
        odat_ros::toMsg(training_data, training_data_msg);
        training_data_msg.object_id = object_id_;

        training_data_pub_.publish(training_data_msg);
        ROS_INFO("Training message published.");
        cv::imwrite("training_image_" + object_id_ + ".png", image);
        ROS_INFO("Click left to see incoming images again.");
    }

    ros::NodeHandle nh_;
    ros::NodeHandle nh_priv_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Publisher training_data_pub_;

    cv::Mat training_image_;
    std::vector<cv::Point> polygon_points_;
    odat::Pose2D object_pose_;
    Mode current_mode_;

    ros::Timer loop_timer_;
    cv::Point current_mouse_position_;

    std::string object_id_;

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "trainer_node");
  TrainerNode trainer;
  ros::spin();
  return 0;
}


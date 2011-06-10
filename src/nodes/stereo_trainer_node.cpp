#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "vision_msgs/TrainingData.h"

namespace enc = sensor_msgs::image_encodings;

/**
* \class StereoTrainerNode
* \author Stephan Wirth
* \brief node that provides a training gui
* This node displays incoming image messages using OpenCV. The image stream
* can be paused to draw a polygon on top of the still image. Then the image
* together with the polygon can be sent as a object training message to the
* ROS system.
*/
class StereoTrainerNode
{
public:
    enum Mode
    {
        DISPLAY_VIDEO,
        AWAITING_TRAINING_INPUT,
        SHOWING_TRAINING_IMAGE,
        PAINTING
    };

    StereoTrainerNode() : it_(nh_), current_mode_(DISPLAY_VIDEO)
    {
        image_sub_ = it_.subscribe("image", 1, &StereoTrainerNode::imageCallback, this);
        training_data_pub_ = nh_.advertise<vision_msgs::TrainingData>("training_data", 1);

        // Synchronize inputs. Topic subscriptions happen on demand. 
        exact_sync_.reset(new ExactSync(ExactPolicy(10),
                                        sub_filter_image_, sub_filter_features_));
        exact_sync_->registerCallback(
                boost::bind(&StereoTrainerNode::trainingInputCallback,
                            this, _1, _2));

        cv::namedWindow("Training GUI");
        cv::setMouseCallback("Training GUI", &StereoTrainerNode::staticMouseCallback, this);

        loop_timer_ = nh_.createTimer(ros::Duration(0.05), &StereoTrainerNode::processEvents, this);
    }

    ~StereoTrainerNode()
    {
        cv::destroyWindow("Training GUI");
    }

private:

    void imageCallback(const sensor_msgs::ImageConstPtr& image_msg)
    {
        setCurrentImage(image_msg);
    }

    void setCurrentImage(const sensor_msgs::ImageConstPtr& image_msg)
    {
        cv_bridge::CvImageConstPtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvShare(image_msg, enc::BGR8);
            current_image_ = cv_ptr->image.clone();
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
        }
    }

    void trainingInputCallback(const sensor_msgs::ImageConstPtr& image_msg,
                               const sensor_msgs::PointCloud2ConstPtr& features_msg)
    {
        // unsubscribe from training data
        ROS_INFO("Training input arrived. Unsubscribing from features.");
        sub_filter_image_.unsubscribe();
        sub_filter_features_.unsubscribe();

        setCurrentImage(image_msg);
        cv::imshow("Training GUI", current_image_);
        training_image_msg_ = image_msg;
        training_features_msg_ = features_msg;
        ROS_INFO("Entered painting mode, waiting for user input.");
        current_mode_ = PAINTING;
    }

    static void staticMouseCallback(int event, int x, int y, int flags, void* param)
    {
        // extract this pointer and call function on object
        StereoTrainerNode* node = reinterpret_cast<StereoTrainerNode*>(param);
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
            ROS_INFO("Space pressed, waiting for input training data.");
            current_mode_ = AWAITING_TRAINING_INPUT;
            // unsubscribe from image
            ROS_INFO("Unsubscribing from image.");
            image_sub_.shutdown();

            // subscribe to synchronized image and features
            ROS_INFO("Subscribing to features.");
            // Queue size 1 should be OK; 
            // the one that matters is the synchronizer queue size.
            sub_filter_image_.subscribe(it_, "image", 1);
            sub_filter_features_.subscribe(nh_, "features", 1);
            ROS_INFO("Waiting for training input");
        }
        else if (key == ' ' && current_mode_ == PAINTING)
        {
            if (polygon_points_.size() > 2)
            {
                publishTrainingData(polygon_points_, "object1");
            }
            current_mode_ = SHOWING_TRAINING_IMAGE;
            cv::Mat image_with_polygon = current_image_.clone();
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
            image_sub_ = it_.subscribe("image", 1, &StereoTrainerNode::imageCallback, this);
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
                    cv::Mat image_with_polygon = current_image_.clone();
                    const cv::Point* point_data = painting_polygon_points.data();
                    int num_points = painting_polygon_points.size();
                    bool closed = true;
                    cv::polylines(image_with_polygon, &point_data, &num_points, 
                            1, closed, cv::Scalar(0, 255, 0), 1);
                    cv::imshow("Training GUI", image_with_polygon);
                }
            }
            else if (current_mode_ == DISPLAY_VIDEO && !current_image_.empty())
            {
                cv::imshow("Training GUI", current_image_);
            }
        }
    }

    void publishTrainingData(const std::vector<cv::Point> polygon_points,
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
        object_description.id = object_name;
        object_description.outline = polygon;

        vision_msgs::TrainingData training_data;

        training_data.image = *training_image_msg_;
        training_data.stereo_features = *training_features_msg_;
        training_data.object_description = object_description;
        training_data_pub_.publish(training_data);

        ROS_INFO("Training message published.");
    }

    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Publisher training_data_pub_;

    // for synchronized subscriptions
    image_transport::SubscriberFilter sub_filter_image_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_filter_features_;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image,
        sensor_msgs::PointCloud2> ExactPolicy;
    typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
    boost::shared_ptr<ExactSync> exact_sync_;
 
    cv::Mat current_image_;
    std::vector<cv::Point> polygon_points_;
    Mode current_mode_;

    ros::Timer loop_timer_;
    cv::Point current_mouse_position_;

    sensor_msgs::ImageConstPtr training_image_msg_;
    sensor_msgs::PointCloud2ConstPtr training_features_msg_;

};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "stereo_traininer_node");
  StereoTrainerNode trainer;
  ros::spin();
  return 0;
}


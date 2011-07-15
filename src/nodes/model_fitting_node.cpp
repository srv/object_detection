#include <ros/ros.h>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

#include <tf/transform_broadcaster.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include "descriptor.h"
#include "alignment.h"

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
typedef pcl::PointCloud<object_detection::Descriptor> FeatureCloud;

class ModelFittingNode
{
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    message_filters::Subscriber<PointCloud> point_cloud_sub_;
    message_filters::Subscriber<FeatureCloud> feature_cloud_sub_;
    message_filters::TimeSynchronizer<PointCloud, FeatureCloud> synchronizer_;

    ros::Publisher pose_pub_;
    ros::Publisher model_pub_;
    ros::Publisher inlier_pub_;

    tf::TransformBroadcaster tf_broadcaster_;

    double scale_error_threshold_;
    double ransac_threshold_;

    object_detection::Alignment<PointCloud::PointType, PointCloud::PointType, FeatureCloud::PointType> alignment_;

  public:
    ModelFittingNode() : nh_private_("~"),
    synchronizer_(point_cloud_sub_, feature_cloud_sub_, 10)
    {
        init();
    }

    ~ModelFittingNode()
    {
    }

    void init()
    {
        if(loadModel())
        {
            point_cloud_sub_.subscribe(nh_, "point_cloud", 1);
            feature_cloud_sub_.subscribe(nh_, "feature_cloud", 1);
            synchronizer_.registerCallback(&ModelFittingNode::runFitting, this);
        /*
        pose_pub_ = nh_private_.advertise<geometry_msgs::Pose>("pose", 1);
        model_pub_ = nh_.advertise<PointCloud>("model_points", 1);
        inlier_pub_ = nh_.advertise<PointCloud>("inlier_model_points", 1);

        nh_private_.param<double>("scale_error_threshold", scale_error_threshold_, 0.5);
        nh_private_.param<double>("ransac_threshold", ransac_threshold_, 0.1);

        ROS_INFO("scale error threshold is set to %f", scale_error_threshold_);
        ROS_INFO("ransac threshold is set to %f", ransac_threshold_);
        */
        }
    }

    bool loadModel()
    {
        if (!nh_private_.hasParam("model_points_file"))
        {
            ROS_ERROR("No model points file given! Please specify the model_points_file parameter!");
            return false;
        }

        if (!nh_private_.hasParam("model_features_file"))
        {
            ROS_ERROR("No model features file given! Please specify the model_features_file parameter!");
            return false;
        }

        std::string model_points_file;
        nh_private_.getParam("model_points_file", model_points_file);

        std::string model_features_file;
        nh_private_.getParam("model_features_file", model_features_file);

        // load model
        PointCloud::Ptr model_point_cloud(new PointCloud());
        if (pcl::io::loadPCDFile<PointCloud::PointType>(model_points_file, *model_point_cloud) == -1)
        {
            ROS_ERROR_STREAM("Couldn't read model_points_file '" 
                << model_points_file << "'.");
            return false;
        }
        ROS_INFO_STREAM("Loaded " << model_point_cloud->points.size()
            << " model points from " << model_points_file << ".");
    
        FeatureCloud::Ptr model_feature_cloud(new FeatureCloud());
    
        if (pcl::io::loadPCDFile<FeatureCloud::PointType>(model_features_file, *model_feature_cloud) == -1)
        {
            ROS_ERROR_STREAM("Couldn't read model_features_file '" 
                    << model_features_file << "'.");
            return false;
        }
        ROS_INFO_STREAM("Loaded " << model_feature_cloud->points.size()
            << " model features from " << model_features_file << ".");

        if (model_point_cloud->points.size() !=
                model_feature_cloud->points.size())
        {
            ROS_ERROR("Loaded model points and model features do not have the same size!");
        } 
        alignment_.setInputCloud(model_point_cloud);
        alignment_.setSourceFeatures(model_feature_cloud);
        return true;
    }



    void runFitting(const PointCloud::ConstPtr& point_cloud, const FeatureCloud::ConstPtr& feature_cloud)
    {
        if (point_cloud->points.size() != feature_cloud->points.size())
        {
            ROS_ERROR("Point Cloud and Feature Cloud do not match in size!");
            ROS_ERROR("%zu points and %zu features received.", point_cloud->points.size(), feature_cloud->points.size());
            return;
        }

        alignment_.setInputTarget(point_cloud);
        alignment_.setTargetFeatures(feature_cloud);

        PointCloud::Ptr alignment_output(new PointCloud());
        alignment_.align(*alignment_output);
        Eigen::Matrix4f transformation = alignment_.getFinalTransformation();
        std::cout << "Transformation: " << std::endl;
        std::cout << transformation << std::endl;
    }

        /*
    void publishTransformation(const cv::Mat& transformation, 
            const ros::Time& timestamp, const std::string& camera_frame_id)
    {
        assert(transformation.type() == CV_64F);
        assert(transformation.rows == 3 && transformation.cols == 4);

        double xx = transformation.at<double>(0, 0);
        double xy = transformation.at<double>(0, 1);
        double xz = transformation.at<double>(0, 2);
        double yx = transformation.at<double>(1, 0);
        double yy = transformation.at<double>(1, 1);
        double yz = transformation.at<double>(1, 2);
        double zx = transformation.at<double>(2, 0);
        double zy = transformation.at<double>(2, 1);
        double zz = transformation.at<double>(2, 2);
        btMatrix3x3 rot_mat(xx, xy, xz, yx, yy, yz, zx, zy, zz);

        double tx = transformation.at<double>(0, 3);
        double ty = transformation.at<double>(1, 3);
        double tz = transformation.at<double>(2, 3);
        btVector3 translation(tx, ty, tz);

        tf_broadcaster_.sendTransform(
                tf::StampedTransform(
                    tf::Transform(rot_mat, translation),
                    timestamp, camera_frame_id, "model"));
    }
    */

        /*
    void publishModel(const ros::Time& timestamp)
    {
        pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
        for (size_t i = 0; i < model_points_.size(); ++i)
        {
            pcl::PointXYZRGB point;
            point.x = model_points_[i].x;
            point.y = model_points_[i].y;
            point.z = model_points_[i].z;
            point_cloud.push_back(point);
        }
        point_cloud.header.stamp = timestamp;
        point_cloud.header.frame_id = "model";
        model_pub_.publish(point_cloud);
    }
    */
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "model_fitting");
    ModelFittingNode model_fitting;
    ros::spin();
    return 0;
}


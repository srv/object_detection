#include <ros/ros.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <pcl/ros/conversions.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>

#include <vision_msgs/StereoFeatures.h>

#include <tf/transform_broadcaster.h>

class MovementCalculatorNode
{
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    ros::Subscriber stereo_features_sub_;
    ros::Publisher pose_pub_;
    ros::Publisher model_pub_;
    ros::Publisher inlier_pub_;

    cv::Mat model_features_;
    std::vector<cv::Point3f> model_points_;

    tf::TransformBroadcaster tf_broadcaster_;

    double scale_error_threshold_;
    double ransac_threshold_;

  public:
    MovementCalculatorNode() : nh_private_("~")
    {
        init();
    }

    ~MovementCalculatorNode()
    {
    }

    void init()
    {
        stereo_features_sub_ = nh_.subscribe("stereo_features", 1, &MovementCalculatorNode::stereoFeaturesCb, this);
        pose_pub_ = nh_private_.advertise<geometry_msgs::Pose>("pose", 1);

        model_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("model_points", 1);
        inlier_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("inlier_model_points", 1);

        nh_private_.param<double>("scale_error_threshold", scale_error_threshold_, 0.5);
        nh_private_.param<double>("ransac_threshold", ransac_threshold_, 0.1);

        ROS_INFO("scale error threshold is set to %f", scale_error_threshold_);
        ROS_INFO("ransac threshold is set to %f", ransac_threshold_);
    }

    void stereoFeaturesCb(const vision_msgs::StereoFeaturesConstPtr& features_msg)
    {
        // return if there are no features
        if (features_msg->features.size() == 0) 
        {
            return;
        }

        pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
        pcl::fromROSMsg(features_msg->world_points, point_cloud);
        size_t num_features = features_msg->features.size();
        assert(num_features == point_cloud.size());

        size_t descriptor_size = features_msg->features[0].descriptor.size();

        cv::Mat features_mat(num_features, descriptor_size, CV_32F);
        std::vector<cv::Point3f> points(num_features);
        
        pcl::PointCloud<pcl::PointXYZRGB>::const_iterator point_iter = 
            point_cloud.begin();

        for (size_t i = 0; i < num_features; ++i)
        {
            std::copy(features_msg->features[i].descriptor.begin(),
                      features_msg->features[i].descriptor.end(),
                      features_mat.ptr<float>(i));

            const pcl::PointXYZRGB point = *point_iter;
            points[i].x = point.x;
            points[i].y = point.y;
            points[i].z = point.z;
            point_iter++;
        }
 
        if (!model_features_.empty())
        {
            cv::Mat transformation;
            /*bool success = matchFeaturePoints(features_mat, points, 
                    model_features_, model_points_,
                    transformation);
                    */
            std::vector<cv::DMatch> inlier_matches;
            bool success = matchFeaturePoints(model_features_, model_points_,
                    features_mat, points, transformation, 
                    inlier_matches);
            if (success)
            {
                std::cout << transformation.col(3) << std::endl;
                publishTransformation(transformation, 
                        features_msg->header.stamp, 
                        features_msg->header.frame_id);
                std::vector<cv::Point3f> inlier_matched_model_points(inlier_matches.size());
                std::vector<cv::Point3f> inlier_matched_world_points(inlier_matches.size());

                /*
                cv::Mat transformed_world_points_mat;
                cv::transform(cv::Mat(points), transformed_world_points_mat, 
                        transformation.inv());
                std::vector<cv::Point3f> transformed_world_points = 
                    cv::Mat_<cv::Point3f>(transformed_world_points_mat);
                    */

                for (size_t i = 0; i < inlier_matches.size(); ++i)
                {
                    inlier_matched_model_points[i] = model_points_[inlier_matches[i].queryIdx];
                    inlier_matched_world_points[i] = points[inlier_matches[i].trainIdx];
                }
                pcl::PointCloud<pcl::PointXYZRGB> point_cloud;
                for (size_t i = 0; i < inlier_matched_model_points.size(); ++i)
                {
                    pcl::PointXYZRGB point;
                    point.x = inlier_matched_model_points[i].x;
                    point.y = inlier_matched_model_points[i].y;
                    point.z = inlier_matched_model_points[i].z;
                    point_cloud.push_back(point);
                }
                point_cloud.header.stamp = features_msg->header.stamp;
                point_cloud.header.frame_id = "model";
                inlier_pub_.publish(point_cloud);
             }
            publishModel(features_msg->header.stamp);
        }
        else
        {
            model_features_ = features_mat;
            cv::Point3f centroid(0, 0, 0);
            for (size_t i = 0; i < points.size(); ++i)
            {
               centroid += points[i]; 
            }
            centroid.x /= points.size();
            centroid.y /= points.size();
            centroid.z /= points.size();
            for (size_t i = 0; i < points.size(); ++i)
            {
                points[i] = points[i] - centroid;
            }
            model_points_ = points;
        }
    }

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

    
    /**
    * Calculates the transformation that has to be made to
    * transform points1 to points2 by first matching the
    * corresponding features and then calculating the
    * 6D transformation.
    */
    bool matchFeaturePoints(const cv::Mat& features1, 
            const std::vector<cv::Point3f> points1,
            const cv::Mat& features2,
            const std::vector<cv::Point3f> points2,
            cv::Mat& transformation,
            std::vector<cv::DMatch>& inlier_matches)
    {
        assert((unsigned int)features1.rows == points1.size());
        assert((unsigned int)features2.rows == points2.size());
        inlier_matches.clear();

        // 3d matching
        cv::Ptr<cv::DescriptorMatcher> descriptor_matcher = cv::DescriptorMatcher::create("BruteForce");
        std::vector<std::vector<cv::DMatch> > matches;
        cv::Mat query_features = features1;
        cv::Mat training_features = features2;
        descriptor_matcher->knnMatch(query_features, training_features, matches, 2);

        std::vector<cv::Point3f> matched_points1;
        std::vector<cv::Point3f> matched_points2;

        cv::Mat canvas(600, 800, CV_8UC3);

        std::vector<cv::DMatch> good_matches;

        for (size_t i = 0; i < matches.size(); ++i)
        {
            if (matches[i].size() == 2)
            {
                const cv::DMatch& match1 = matches[i][0];
                const cv::DMatch& match2 = matches[i][1];
                if (match1.distance / match2.distance < 0.8)
                {
                    matched_points1.push_back(points1[match1.queryIdx]);
                    matched_points2.push_back(points2[match1.trainIdx]);

                    good_matches.push_back(match1);

                    cv::Point2f paint_point1;
                    paint_point1.x = points1[match1.queryIdx].x;
                    paint_point1.y = points1[match1.queryIdx].y;
                    cv::Point2f paint_point2;
                    paint_point2.x = points2[match1.trainIdx].x;
                    paint_point2.y = points2[match1.trainIdx].y;
                    paint_point1 *= 300;
                    paint_point1 += cv::Point2f(400, 300);
                    paint_point2 *= 300;
                    paint_point2 += cv::Point2f(400, 300);
                    cv::line(canvas, paint_point1, paint_point2, cv::Scalar(0, 255, 0));
                }
            }
        }

        if (matched_points1.size() > 3)
        {
            std::vector<uchar> outliers;
            double confidence = 0.999;
            cv::estimateAffine3D(cv::Mat(matched_points1), 
                    cv::Mat(matched_points2), 
                    transformation, outliers, ransac_threshold_, confidence);

            int num_outliers = 0;
            int num_inliers = 0;
            for (size_t i = 0; i < outliers.size(); ++i)
            {
                if (outliers[i] == 0) 
                {
                    num_outliers++; 

                    cv::Point2f paint_point1;
                    paint_point1.x = matched_points1[i].x;
                    paint_point1.y = matched_points1[i].y;
                    cv::Point2f paint_point2;
                    paint_point2.x = matched_points2[i].x;
                    paint_point2.y = matched_points2[i].y;
                    paint_point1 *= 300;
                    paint_point1 += cv::Point2f(400, 300);
                    paint_point2 *= 300;
                    paint_point2 += cv::Point2f(400, 300);
                    cv::line(canvas, paint_point1, paint_point2, cv::Scalar(0, 0, 255));
                }
                else 
                {
                    num_inliers++;
                    inlier_matches.push_back(good_matches[i]);
                }
            }
            std::cout << matched_points1.size() << "\tmatching points. "
                      << "\t" << num_outliers << " outliers, " 
                      << "\t" << num_inliers << " inliers." << std::endl;
        }
        else
        {
            return false;
        }
        cv::imshow("matches", canvas);
        cv::waitKey(3);

        // sanity check
        // singular values should be 1 in a rigid transformation
        // qr is a matrix header to the rotation/scale/shear part
        cv::Mat qr = transformation(cv::Range::all(), cv::Range(0, 3));
        cv::SVD svd(qr);

        cv::Mat error_mat = 1.0 - svd.w;
        double scale_error = cv::norm(error_mat);

        if (scale_error < scale_error_threshold_)
        {
            // calculate scale-free rotation
            cv::Mat rotation = svd.u * svd.vt;
            rotation.copyTo(qr);
            return true;
        }
        else
        {
            return false;
        }
    }

    void calculateTransformationCv(const std::vector<cv::Point3f>& points1,
            const std::vector<cv::Point3f>& points2, cv::Mat& transformation)
    {
    }
 
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "movement_calculator");
    MovementCalculatorNode calculator;
    ros::spin();
    return 0;
}


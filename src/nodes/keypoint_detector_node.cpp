#include <fstream>
#include <boost/algorithm/string.hpp>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <std_msgs/Float32.h>

#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>


namespace enc = sensor_msgs::image_encodings;
using namespace cv;
using namespace std;

using std_msgs::Header;

#define DRAW_RICH_KEYPOINTS_MODE     0
#define DRAW_OUTLIERS_MODE           0

static const std::string WINDOW_NAME = "Keypoint Matching";

class KeypointDetectorNode
{

  public:
    enum KeypointMatcherFilter
    { 
        INVALID_FILTER = -1,
        NONE_FILTER = 0, 
        CROSS_CHECK_FILTER = 1
    };


    KeypointDetectorNode()
    : it_(nh_), ransac_reprojection_threshold_(-1), eval_(false)
    {
        image_sub_ = it_.subscribe("image", 1, &KeypointDetectorNode::imageCb, this);
        cv::namedWindow(WINDOW_NAME, 1);
        inliers_pub_ = nh_.advertise<std_msgs::Float32>("inliers", 1000);
    }

    void setRansacReprojectionThreshold(double threshold)
    {
        ransac_reprojection_threshold_ = threshold;
    }

    void setFeatureDetector(Ptr<FeatureDetector>& detector)
    {
        feature_detector_ = detector;
    }

    void setDescriptorExtractor(Ptr<DescriptorExtractor>& descriptor_extractor)
    {
        descriptor_extractor_ = descriptor_extractor;
    }

    void setDescriptorMatcher(Ptr<DescriptorMatcher>& descriptor_matcher)
    {
        descriptor_matcher_ = descriptor_matcher;
    }

    void setMatcherFilterType(KeypointMatcherFilter filter)
    {
        matcher_filter_type_ = filter;
    }

    void initObject(const cv::Mat& object_image)
    {
        object_image_ = object_image;
        cout << endl << "Extracting model keypoints..." << flush;
        feature_detector_->detect( object_image_, object_keypoints_ );
        cout << "done. " << object_keypoints_.size() << " points." << endl;
        cout << "Computing model descriptors..." << flush;
        descriptor_extractor_->compute( object_image_, object_keypoints_, object_descriptors_);
        cout << "done." << endl;
    }

  private:
	
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    ros::Publisher inliers_pub_;

    double ransac_reprojection_threshold_;
    Ptr<FeatureDetector> feature_detector_;
    Ptr<DescriptorExtractor> descriptor_extractor_;
    Ptr<DescriptorMatcher> descriptor_matcher_;
    KeypointMatcherFilter matcher_filter_type_;

    cv::Mat object_image_;
    std::vector<cv::KeyPoint> object_keypoints_;
    cv::Mat object_descriptors_;

    bool eval_;

    void imageCb(const sensor_msgs::ImageConstPtr& image_msg)
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
        doIteration( object_image_, image, 
                     object_keypoints_, object_descriptors_,
                    feature_detector_, descriptor_extractor_,
                    descriptor_matcher_, matcher_filter_type_, eval_,
                    ransac_reprojection_threshold_);
    }

    void simpleMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                        const Mat& descriptors1, const Mat& descriptors2,
                        vector<DMatch>& matches12 )
    {
        descriptorMatcher->match( descriptors1, descriptors2, matches12 );
    }

    void crossCheckMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
                            const Mat& descriptors1, const Mat& descriptors2,
                            vector<DMatch>& filteredMatches12, int knn=1 )
    {
        filteredMatches12.clear();
        vector<vector<DMatch> > matches12, matches21;
        descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
        descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
        for( size_t m = 0; m < matches12.size(); m++ )
        {
            bool findCrossCheck = false;
            for( size_t fk = 0; fk < matches12[m].size(); fk++ )
            {
                DMatch forward = matches12[m][fk];

                for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
                {
                    DMatch backward = matches21[forward.trainIdx][bk];
                    if( backward.trainIdx == forward.queryIdx )
                    {
                        filteredMatches12.push_back(forward);
                        findCrossCheck = true;
                        break;
                    }
                }
                if( findCrossCheck ) break;
            }
        }
    }

    void doIteration( const Mat& img1, Mat& img2, 
                    vector<KeyPoint>& keypoints1, const Mat& descriptors1,
                    Ptr<FeatureDetector>& detector, Ptr<DescriptorExtractor>& descriptorExtractor,
                    Ptr<DescriptorMatcher>& descriptorMatcher, int matcherFilter, bool eval,
                    double ransacReprojThreshold)
    {
        assert( !img1.empty() );
        Mat H12;
        assert( !img2.empty() );

        vector<KeyPoint> keypoints2;
        detector->detect( img2, keypoints2 );
        cout << keypoints2.size() << " points" << endl;

        if( !H12.empty() && eval )
        {
            cout << "< Evaluate feature detector..." << endl;
            float repeatability;
            int correspCount;
            evaluateFeatureDetector( img1, img2, H12, &keypoints1, &keypoints2, repeatability, correspCount );
            cout << "repeatability = " << repeatability << endl;
            cout << "correspCount = " << correspCount << endl;
            cout << ">" << endl;
        }

        Mat descriptors2;
        descriptorExtractor->compute( img2, keypoints2, descriptors2 );

        vector<DMatch> filteredMatches;
        switch( matcherFilter )
        {
        case CROSS_CHECK_FILTER :
            crossCheckMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches, 1 );
            break;
        default :
            simpleMatching( descriptorMatcher, descriptors1, descriptors2, filteredMatches );
        }

        if( !H12.empty() && eval )
        {
            cout << "< Evaluate descriptor match..." << endl;
            vector<Point2f> curve;
            Ptr<GenericDescriptorMatcher> gdm = new VectorDescriptorMatcher( descriptorExtractor, descriptorMatcher );
            evaluateGenericDescriptorMatcher( img1, img2, H12, keypoints1, keypoints2, 0, 0, curve, gdm );
            for( float l_p = 0; l_p < 1 - FLT_EPSILON; l_p+=0.1f )
                cout << "1-precision = " << l_p << "; recall = " << getRecall( curve, l_p ) << endl;
            cout << ">" << endl;
        }

        vector<int> queryIdxs( filteredMatches.size() ), trainIdxs( filteredMatches.size() );
        for( size_t i = 0; i < filteredMatches.size(); i++ )
        {
            queryIdxs[i] = filteredMatches[i].queryIdx;
            trainIdxs[i] = filteredMatches[i].trainIdx;
        }

        if( ransacReprojThreshold >= 0 )
        {
            vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, queryIdxs);
            vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, trainIdxs);
            H12 = findHomography( Mat(points1), Mat(points2), CV_RANSAC, ransacReprojThreshold );
        }

        Mat drawImg;
        int num_inliers = 0;
        if( !H12.empty() ) // filter outliers
        {
            vector<char> matchesMask( filteredMatches.size(), 0 );
            vector<Point2f> points1; KeyPoint::convert(keypoints1, points1, queryIdxs);
            vector<Point2f> points2; KeyPoint::convert(keypoints2, points2, trainIdxs);
            Mat points1t; perspectiveTransform(Mat(points1), points1t, H12);
            for( size_t i1 = 0; i1 < points1.size(); i1++ )
            {
                if( norm(points2[i1] - points1t.at<Point2f>((int)i1,0)) < 4 ) // inlier
                {
                    matchesMask[i1] = 1;
                    num_inliers++;
                }
            }
            // draw inliers
            drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255), matchesMask
#if DRAW_RICH_KEYPOINTS_MODE
                        , DrawMatchesFlags::DRAW_RICH_KEYPOINTS
#endif
                    );

#if DRAW_OUTLIERS_MODE
            // draw outliers
            for( size_t i1 = 0; i1 < matchesMask.size(); i1++ )
                matchesMask[i1] = !matchesMask[i1];
            cout << "draw2" << endl;
            drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg, CV_RGB(0, 0, 255), CV_RGB(255, 0, 0), matchesMask,
                        DrawMatchesFlags::DRAW_OVER_OUTIMG | DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
#endif
            std::cout << num_inliers * 100.0f / filteredMatches.size() << " % inliers." << std::endl;
            std_msgs::Float32 inliers_msg;
            inliers_msg.data = num_inliers * 100.0f / filteredMatches.size();
            inliers_pub_.publish(inliers_msg);
        }
        else
        {
            cout << "draw3" << endl;
            drawMatches( img1, keypoints1, img2, keypoints2, filteredMatches, drawImg );
        }

        imshow( WINDOW_NAME, drawImg );
        cvWaitKey(3);
    }
};


KeypointDetectorNode::KeypointMatcherFilter getMatcherFilterType( const string& str )
{
    if( str == "NoneFilter" )
        return KeypointDetectorNode::NONE_FILTER;
    if( str == "CrossCheckFilter" )
        return KeypointDetectorNode::CROSS_CHECK_FILTER;
    CV_Error(CV_StsBadArg, "Invalid filter name");
    return KeypointDetectorNode::INVALID_FILTER;
}

void help(char** argv)
{
     cout << "\nThis ROS node demonstrates keypoint finding and matching between 2 images using features2d framework.\n"
     << "\n"
     << "If ransacReprojThreshold>=0 then homography matrix is calculated\n"
     << "Example:\n"
     << argv[0] << " [detectorType] [descriptorType] [matcherType] [matcherFilterType] [object-image] [ransacReprojThreshold]\n"
     << "\n"
     << "Matches are filtered using homography matrix if ransacReprojThreshold>=0\n"
     << "Example:\n"
     << argv[0] << " SURF SURF BruteForce CrossCheckFilter box.jpg 3\n"
     << "\n"
     << "Possible detectorType values: see in documentation on createFeatureDetector().\n"
     << "Possible descriptorType values: see in documentation on createDescriptorExtractor().\n"
     << "Possible matcherType values: see in documentation on createDescriptorMatcher().\n"
     << "Possible matcherFilterType values: NoneFilter, CrossCheckFilter." << endl;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "keypoint_detector");

    if( argc != 7 )
    {
    	help(argv);
        return -1;
    }
    
    double ransacReprojThreshold = atof(argv[6]);

    cout << "< Creating detector, descriptor extractor and descriptor matcher ..." << endl;
    Ptr<FeatureDetector> detector = FeatureDetector::create( argv[1] );
    if (detector.empty())
    {
        std::cerr << "Cannot create feature detector with name '" << argv[1] << "'." << std::endl;
        return -1;
    }

    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create( argv[2] );
    if (descriptorExtractor.empty())
    {
        std::cerr << "Cannot create descriptor extractor with name '" << argv[2] << "'." << std::endl;
        return -1;
    }

    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create( argv[3] );
    if (descriptorMatcher.empty())
    {
        std::cerr << "Cannot create descriptor matcher with name '" << argv[3] << "'." << std::endl;
        return -1;
    }

    KeypointDetectorNode node;
    node.setRansacReprojectionThreshold(ransacReprojThreshold);
    node.setFeatureDetector(detector);
    node.setDescriptorExtractor(descriptorExtractor);
    node.setDescriptorMatcher(descriptorMatcher);
    node.setMatcherFilterType(getMatcherFilterType(argv[4]));
		
    cout << "< Reading the image..." << endl;
    Mat object_img = imread( argv[5] );
    cout << ">" << endl;
    if( object_img.empty() )
    {
        cout << "Can not read input image" << endl;
        return -1;
    }
    node.initObject(object_img);


    ros::spin();
    return 0;
}

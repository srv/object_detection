#ifndef ALIGNMENT_H_
#define ALIGNMENT_H_

// PCL includes
#include <pcl/registration/registration.h>
#include <pcl/common/eigen.h>
#include <pcl/features/feature.h>

#include <boost/make_shared.hpp>

// adapted from ia_ransac.h from pcl package

namespace object_detection
{
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /** \brief Alignment aligns two feature/point sets using ransac 
    */
  template <typename PointSource, typename PointTarget, typename FeatureT>
  class Alignment : public pcl::Registration<PointSource, PointTarget>
  {
    using pcl::Registration<PointSource, PointTarget>::reg_name_;
    using pcl::Registration<PointSource, PointTarget>::getClassName;
    using pcl::Registration<PointSource, PointTarget>::input_;
    using pcl::Registration<PointSource, PointTarget>::indices_;
    using pcl::Registration<PointSource, PointTarget>::target_;
    using pcl::Registration<PointSource, PointTarget>::final_transformation_;
    using pcl::Registration<PointSource, PointTarget>::transformation_;
    using pcl::Registration<PointSource, PointTarget>::corr_dist_threshold_;
    using pcl::Registration<PointSource, PointTarget>::inlier_threshold_;
    using pcl::Registration<PointSource, PointTarget>::min_number_correspondences_;
    using pcl::Registration<PointSource, PointTarget>::max_iterations_;
    using pcl::Registration<PointSource, PointTarget>::tree_;

    typedef typename pcl::Registration<PointSource, PointTarget>::PointCloudSource PointCloudSource;
    typedef typename PointCloudSource::Ptr PointCloudSourcePtr;
    typedef typename PointCloudSource::ConstPtr PointCloudSourceConstPtr;

    typedef typename pcl::Registration<PointSource, PointTarget>::PointCloudTarget PointCloudTarget;

    typedef pcl::PointIndices::Ptr PointIndicesPtr;
    typedef pcl::PointIndices::ConstPtr PointIndicesConstPtr;

    typedef pcl::PointCloud<FeatureT> FeatureCloud;
    typedef typename FeatureCloud::Ptr FeatureCloudPtr;
    typedef typename FeatureCloud::ConstPtr FeatureCloudConstPtr;

    typedef typename pcl::KdTreeFLANN<FeatureT>::Ptr FeatureKdTreePtr; 

    public:
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /** \brief Constructor. */
      Alignment () : nr_samples_(3), feature_matching_threshold_(0.8)
      {
        reg_name_ = "Alignment";
        feature_tree_ = boost::make_shared<pcl::KdTreeFLANN<FeatureT> > ();
      };

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /** \brief Provide a boost shared pointer to the source point cloud's feature descriptors
        * \param features the source point cloud's features
        */
      void setSourceFeatures (const FeatureCloudConstPtr &features);

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /** \brief Get a pointer to the source point cloud's features */
      inline FeatureCloudConstPtr const getSourceFeatures () { return (input_features_); }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /** \brief Provide a boost shared pointer to the target point cloud's feature descriptors
        * \param features the target point cloud's features
        */
      void setTargetFeatures (const FeatureCloudConstPtr &features);

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /** \brief Get a pointer to the target point cloud's features */
      inline FeatureCloudConstPtr const getTargetFeatures () { return (target_features_); }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /** \brief Set the threshold for correspondence selection/rejection
        * As described in Lowe 2003, "Distinctive Image Features from Scale-Invariant Keypoints", to find a feature
        * match the first 2 nearest neighbors of a feature are retrieved. If the ratio of the distance to the first
        * to the distance of the second is below this threshold, the match is assumed as valid.
        * \param threshold the threshold
        */
      void setFeatureMatchingThreshold (float threshold) { feature_matching_threshold_ = threshold; }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /** \brief Get the current threshold for feature correspondence rejection */
      float getFeatureMatchingThreshold () { return (feature_matching_threshold_); }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /** \brief Set the number of samples to use during each iteration (should be 3 always?)
        * \param nr_samples the number of samples to use during each iteration
        */
      void setNumberOfSamples (int nr_samples) { nr_samples_ = nr_samples; }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /** \brief Set the number of samples to use during each iteration, as set by the user */
      int getNumberOfSamples () { return (nr_samples_); }

    private:
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /** \brief Choose a random index between 0 and n-1
        * \param n the number of possible indices to choose from
        */
      inline int getRandomIndex (int n) { return (n * (rand () / (RAND_MAX + 1.0))); };
      
      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /** \brief Selects  nr_samples_ correspondences while making sure that their pairwise distances are greater 
        * than a user-defined minimum distance (min_sample_distance_).
        * \param source_indices the resulting indices in the source data
        * \param target_indices the resulting indices in the target data
        */
       void selectCorrespondences (const std::vector<int> &input_matched_indices, 
               const std::vector<int> &target_matched_indices,
               std::vector<int> &source_indices, std::vector<int> &target_indices);

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /** \brief For the given \a input_feature, find its next 2 neighbors in the target feature cloud 
        * and return the matching index if the ratio of the distances of the 2 neighbors is below the feature matching
        * threshold.
        * \param input_feature feature descriptors
        * \param ratio_threshold threshold for the matching
        * \param corresponding_index the resulting index in the target cloud, if no correspondence is found, the
        *        value remains untouched
        * \return true if a similar feature was found, false otherwise
        */
      bool findSimilarFeature (const FeatureT &input_feature, float ratio_threshold, int &corresponding_index);

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
       /** \brief finds all matchings between input and target features 
        */
       void findAllMatchings(std::vector<int> &input_matched_indices, std::vector<int> & target_matched_indices);

    protected:

      inline void 
      computeSampleDistanceThreshold (const PointCloudSource &cloud)
      {
        // Compute the principal directions via PCA
        Eigen::Vector4f xyz_centroid;
        compute3DCentroid (cloud, xyz_centroid);
        EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
        computeCovarianceMatrixNormalized (cloud, xyz_centroid, covariance_matrix);
        EIGEN_ALIGN16 Eigen::Vector3f eigen_values;
        EIGEN_ALIGN16 Eigen::Matrix3f eigen_vectors;
        pcl::eigen33 (covariance_matrix, eigen_vectors, eigen_values);

        // Compute the distance threshold for sample selection
        sample_dist_thresh_ = eigen_values.array ().sqrt ().sum () / 3.0;
        //sample_dist_thresh_ *= sample_dist_thresh_;
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /** \brief Rigid transformation computation method.
        * \param output the transformed input point cloud dataset using the rigid transformation found
        */
      virtual void computeTransformation (PointCloudSource &output);

      
      /** \brief The source point cloud's feature descriptors. */
      FeatureCloudConstPtr input_features_;

      /** \brief The target point cloud's feature descriptors. */
      FeatureCloudConstPtr target_features_;  

      /** \brief The number of samples to use during each iteration. */
      int nr_samples_;

      /** \brief The minimum distances between samples. */
      float sample_dist_thresh_;

      /** \brief threshold for feature correspondence acceptance/rejection */
      float feature_matching_threshold_;
     
      /** \brief The KdTree used to compare feature descriptors. */
      FeatureKdTreePtr feature_tree_;               

  };
}

#include "alignment.hpp"

#endif  //#ifndef ALIGNMENT_H_

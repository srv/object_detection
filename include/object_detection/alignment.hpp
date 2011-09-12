
namespace od = object_detection;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget, typename FeatureT> void
od::Alignment<PointSource, PointTarget, FeatureT>::setSourceFeatures (
      const FeatureCloudConstPtr &features)
{
  if (features == NULL || features->points.empty ())
  {
    ROS_ERROR ("[%s::setSourceFeatures] Invalid or empty point cloud dataset given!", getClassName ().c_str ());
    return;
  }
  input_features_ = features;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget, typename FeatureT> void
od::Alignment<PointSource, PointTarget, FeatureT>::setTargetFeatures (
      const FeatureCloudConstPtr &features)
{
  if (features == NULL || features->points.empty ())
  {
    ROS_ERROR ("[%s::setTargetFeatures] Invalid or empty point cloud dataset given!", getClassName ().c_str ());
    return;
  }
  target_features_ = features;
  feature_tree_->setInputCloud (target_features_);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget, typename FeatureT> void
od::Alignment<PointSource, PointTarget, FeatureT>::selectCorrespondences (
      const std::vector<int>& input_matched_indices, const std::vector<int> &target_matched_indices,
      std::vector<int> &source_indices, std::vector<int> &target_indices)
{
  if (nr_samples_ > (int) input_->points.size ())
  {
    ROS_ERROR ("[%s::selectCorrespondences] The number of samples (%d) must not be greater than the number of points (%d)!",
               getClassName ().c_str (), nr_samples_, (int) input_->points.size ());
    return;
  }


  // Iteratively draw random samples until nr_samples is reached
  int iterations_without_a_sample = 0;
  int max_iterations_without_a_sample = 3 * input_->points.size ();
  source_indices.clear();
  target_indices.clear();
  while ((int) source_indices.size() < nr_samples_)
  {
    // Choose a sample at random
    int sample_matched_index = getRandomIndex (input_matched_indices.size ());
    int sample_index = input_matched_indices[sample_matched_index];

    // Check to see if the sample is 1) unique and 2) far away from the other samples
    
    bool valid_sample = true;
    for (size_t i = 0; i < source_indices.size (); ++i)
    {
      float distance_between_samples = euclideanDistance (input_->points[sample_index], input_->points[source_indices[i]]);

      if (sample_index == source_indices[i] || distance_between_samples < sample_dist_thresh_)
      {
          valid_sample = false;
          break;
      }
    }

    // If the sample is valid, add it to the output
    if (valid_sample)
    {
      source_indices.push_back (sample_index);
      target_indices.push_back (target_matched_indices[sample_matched_index]);
      iterations_without_a_sample = 0;
    }
    else
    {
      ++iterations_without_a_sample;
    }

    // If no valid samples can be found, relax the inter-sample distance requirements
    if (iterations_without_a_sample >= max_iterations_without_a_sample)
    {
      ROS_WARN ("[%s::selectCorrespondences] No valid sample found after %d iterations. Relaxing sample_dist_thresh_ to %f", 
                getClassName ().c_str (), iterations_without_a_sample, 0.5*sample_dist_thresh_);
      sample_dist_thresh_ *= 0.5;
      iterations_without_a_sample = 0;
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget, typename FeatureT> bool
od::Alignment<PointSource, PointTarget, FeatureT>::findSimilarFeature (
      const FeatureT &input_feature, float ratio_threshold, int &corresponding_index)
{
  const int k = 2;
  std::vector<int> nn_indices (k);
  std::vector<float> nn_distances (k);

  feature_tree_->nearestKSearch (input_feature, k, nn_indices, nn_distances);
  float squared_distance_ratio = nn_distances[0] / nn_distances[1];
  if (squared_distance_ratio < ratio_threshold * ratio_threshold)
  {
      corresponding_index = nn_indices[0];
      return true;
  }
  else
  {
      return false;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget, typename FeatureT> void
od::Alignment<PointSource, PointTarget, FeatureT>::findAllMatchings(
        std::vector<int> &input_matched_indices, std::vector<int> &target_matched_indices)
{
    input_matched_indices.clear();
    target_matched_indices.clear();
    for (size_t i = 0; i < input_features_->points.size(); ++i)
    {
        int target_index;
        if (findSimilarFeature(input_features_->points[i], feature_matching_threshold_, target_index))
        {
            input_matched_indices.push_back(i);
            target_matched_indices.push_back(target_index);
        }
    }
    ROS_INFO ("[%s::findAllMatchings] Found %zu matchings, %zu features did not match.",
            getClassName ().c_str (),
            input_matched_indices.size(),
            input_features_->points.size() - input_matched_indices.size());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget, typename FeatureT> void
od::Alignment<PointSource, PointTarget, FeatureT>::computeTransformation (PointCloudSource &output)
{
  if (!input_features_)
  {
    ROS_ERROR ("[%s::computeTransformation] No source features were given! Call setSourceFeatures before aligning.", 
               getClassName ().c_str ());
    return;
  }
  if (!target_features_)
  {
    ROS_ERROR ("[%s::computeTransformation] No target features were given! Call setTargetFeatures before aligning", 
               getClassName ().c_str ());
    return;
  }

  std::vector<int> sample_indices;
  std::vector<int> corresponding_indices;
  PointCloudSource input_transformed;
  float lowest_error (0);

  final_transformation_ = Eigen::Matrix4f::Identity ();

  std::vector<int> input_matched_indices;
  std::vector<int> target_matched_indices;

  findAllMatchings(input_matched_indices, target_matched_indices);
  if (input_matched_indices.size() < 8)
  {
    ROS_ERROR ("[%s::computeTransformation] Too few matchings (%zu)!", 
                getClassName ().c_str (), input_matched_indices.size());
    return;
  }

  computeSampleDistanceThreshold(*input_);
  ROS_INFO ("[%s::computeTransformation] Computed sample distance threshold of %f", 
             getClassName ().c_str (), sample_dist_thresh_);

  unsigned int min_num_inliers = 0.5 * input_matched_indices.size();
  int i_iter;
  for (i_iter = 0; i_iter < max_iterations_; ++i_iter)
  {
    // select correspondences
    selectCorrespondences(input_matched_indices, target_matched_indices,
            sample_indices, corresponding_indices);

    // Estimate the transform from the samples to their corresponding points
    estimateRigidTransformationSVD (*input_, sample_indices, *target_, corresponding_indices, transformation_);

    // Tranform the data and compute the error
    // TODO transform only matched indices?
    transformPointCloud (*input_, input_transformed, transformation_);

    // compute error
    float error = 0.0;
    unsigned int num_inliers = 0;
    for (size_t i = 0; i < input_matched_indices.size(); ++i)
    {
        const PointSource& transformed_point = input_transformed.points[input_matched_indices[i]];
        const PointTarget& corresponding_point = target_->points[target_matched_indices[i]];
        float distance = euclideanDistance(transformed_point, corresponding_point);
        if (distance < inlier_threshold_)
        {
            error += distance / inlier_threshold_;
            num_inliers++;
        }
        else
        {
            error += 1.0;
        }
    }

    // If the new error is lower, update the final transformation
    if ((i_iter == 0 || error < lowest_error))
    {
      lowest_error = error;
      final_transformation_ = transformation_;
    }
    if (num_inliers >= min_num_inliers)
    {
        break;
    }
  }

  // final step: compute using all inliers
  std::vector<int> inlier_indices;
  std::vector<int> inlier_corresponding_indices;
  for (size_t i = 0; i < input_matched_indices.size(); ++i)
    {
        const PointSource& transformed_point = input_transformed.points[input_matched_indices[i]];
        const PointTarget& corresponding_point = target_->points[target_matched_indices[i]];
        float distance = euclideanDistance(transformed_point, corresponding_point);
        if (distance < inlier_threshold_)
        {
            inlier_indices.push_back(input_matched_indices[i]);
            inlier_corresponding_indices.push_back(target_matched_indices[i]);
        }
  }
  estimateRigidTransformationSVD (*input_, inlier_indices, *target_, inlier_corresponding_indices, final_transformation_);
  ROS_INFO ("[%s::computeTransformation] Computed transformation with %zu inliers after %d iterations.", 
             getClassName ().c_str (), inlier_indices.size(), i_iter);

  // Apply the final transformation
  transformPointCloud (*input_, output, final_transformation_);

  // compute some statistics
  int num_matched_outliers = 0;
  int num_matched_inliers = 0; 
  int num_unmatched_outliers = 0; 
  int num_unmatched_inliers = 0;

  for (size_t i = 0; i < input_->points.size(); ++i)
  {
      bool is_inlier = false;
      bool has_matched = false;
      int k = 1;
      std::vector<int> nn_indices(k);
      std::vector<float> nn_distances(k);
      // find nearest point in target
      tree_->nearestKSearch (input_->points[i], k, nn_indices, nn_distances);
      if (nn_distances[0] < inlier_threshold_ * inlier_threshold_)
      {
          is_inlier = true;
      }
      if (std::find(input_matched_indices.begin(), input_matched_indices.end(), i)
              != input_matched_indices.end())
      {
          has_matched = true;
      }
      if (is_inlier)
      {
          if (has_matched)
          {
              num_matched_inliers++;
          }
          else
          {
              num_unmatched_inliers++;
          }
      }
      else
      {
          if (has_matched)
          {
              num_matched_outliers++;
          }
          else
          {
              num_unmatched_outliers++;
          }
      }
  }
  ROS_INFO ("[%s::computeTransformation] Statistics: inlier threshold: %f, %d matched inliers, %d unmatched inliers, %d outliers match, %d unmatched outliers.", 
             getClassName ().c_str (), inlier_threshold_, num_matched_inliers, num_unmatched_inliers, num_matched_outliers, num_unmatched_outliers);

}


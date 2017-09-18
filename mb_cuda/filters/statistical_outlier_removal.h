#ifndef MBCUDA_STATISTICAL_OUTLIER_REMOVAL_H_
#define MBCUDA_STATISTICAL_OUTLIER_REMOVAL_H_

#define FLANN_USE_CUDA
#include "mb_cuda/common/point_types.h"
#include "mb_cuda/common/thrust.h"
#include <flann/flann.hpp>

namespace mb_cuda {

  typedef mb_cuda::PointXYZRGB pointT;
  typedef thrust::device_vector<pointT> deviceCloudT;
  typedef thrust::host_vector<pointT> hostCloudT;

  /**
   * @brief statistical_outlier_removal
   * @param [in] input_points
   * @param [in] k
   * @param [in] std_mul
   * @param [out] inliers_indices
   * @param [out] inliers_num
   */
  void statistical_outlier_removal(deviceCloudT input_points, int k, float std_mul, thrust::device_vector<int> inliers_indices, int& inliers_num);
}

#endif

#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>


#include "mb_cuda/common/point_types.h"
#include "mb_cuda/io/pcl_thrust.h"
#include "mb_cuda/io/host_device.h"
#include "mb_cuda/filters/pass_through.h"

#include <boost/timer.hpp>
#include <boost/shared_ptr.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include "grabber/kinect2grabber.h"

typedef pcl::PointXYZRGB pointT;
typedef pcl::PointCloud<pointT> pointCloudT;
typedef pointCloudT::Ptr pointCloudPtr;

int main()
{
  pointCloudPtr cloud(new pointCloudT);
  std::cout << "milkbox demo using cuda!" << std::endl;
  std::cout << "getting cloud..." << std::endl;
  //kinect grab
//  sensor::Kinect2Grabber kinect2_grabber(sensor::CUDA,false);

  //initialize the kinect2 sensor
//  kinect2_grabber.start();
//  kinect2_grabber.getPointCloud(cloud);

  //read the pcd file
  std::string fname="//home//wangy//dev//3dvision_ws//projects//build//milkbox_cuda//scene_1.pcd";
  pcl::io::loadPCDFile(fname,*cloud);

  //convert the pcl pointcloud to host_vector type
  thrust::host_vector<mb_cuda::PointXYZRGB> host_cloud;
  mb_cuda::pcl_to_thrust(cloud,host_cloud);
  std::cout<<"host cloud vector size is: "<<host_cloud.size()<<std::endl;

  thrust::device_vector<mb_cuda::PointXYZRGB> device_cloud;
  mb_cuda::host_to_device(host_cloud,device_cloud);
  std::cout<<"device cloud vector size is: "<<device_cloud.size()<<std::endl;

  //filter by passthrough
  thrust::device_vector<mb_cuda::PointXYZRGB> d_filtered_cloud;
  boost::timer _timer;
  mb_cuda::pass_through_filter(device_cloud,d_filtered_cloud,'z',0,0.8);
  std::cerr<<"thrust passthrough elapsed time: "<<_timer.elapsed()*1000<<" ms"<<std::endl;
  std::cerr<<"after filtered, the cloud size is: "<<d_filtered_cloud.size()<<std::endl;

  thrust::host_vector<mb_cuda::PointXYZRGB> h_filtered_cloud;
  mb_cuda::device_to_host(d_filtered_cloud,h_filtered_cloud);
  mb_cuda::thrust_to_pcl(h_filtered_cloud,cloud);

//  pointCloudPtr filtered(new pointCloudT);
//  pcl::PassThrough<pcl::PointXYZRGB> pass;
//  pass.setInputCloud(cloud);
//  pass.setFilterFieldName("z");
//  pass.setFilterLimits(0,0.8);
//  _timer.restart();
//  pass.filter(*filtered);
//  std::cerr<<"pcl passthrough elapsed time: "<<_timer.elapsed()*1000<<" ms"<<std::endl;
//  std::cerr<<"after filtered, the cloud size is: "<<filtered->points.size()<<std::endl;

  //view
  pcl::visualization::PCLVisualizer viewer("point cloud from kinect2");
  viewer.addPointCloud(cloud,"scene_cloud");
  while(!viewer.wasStopped()){
      //kinect2_grabber.getPointCloud(cloud);
      viewer.updatePointCloud(cloud,"scene_cloud");
      viewer.spinOnce();
    }
  return 0;
}

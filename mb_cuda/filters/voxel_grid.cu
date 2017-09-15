#include "mb_cuda/filters/voxel_grid.h"
#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace mb_cuda {

  /**
   * @brief The minmax_pair struct
   */
  struct minmax_tuple
  {
    float3 min_val;
    float3 max_val;
  };

  /**
   * @brief The minmax_unary_op struct
   */
  struct minmax_unary_op:
      public thrust::unary_function<pointT, minmax_tuple>
  {
    __host__ __device__
    minmax_tuple operator () (const pointT& pt) const
    {
      minmax_tuple minmax;
      minmax.min_val.x=pt.x;
      minmax.min_val.y=pt.y;
      minmax.min_val.z=pt.z;

      minmax.max_val.x=pt.x;
      minmax.max_val.y=pt.y;
      minmax.max_val.z=pt.z;
      return minmax;
    }
  };

  /**
   * @brief The minmax_binary_op struct
   */
  struct minmax_binary_op:
      public thrust::binary_function<minmax_tuple,minmax_tuple,minmax_tuple>
  {
    __host__ __device__
    minmax_tuple operator()(const minmax_tuple& p1, const minmax_tuple& p2) const
    {
      minmax_tuple result;
      result.min_val.x=p1.min_val.x>p2.min_val.x ? p2.min_val.x : p1.min_val.x;
      result.min_val.y=p1.min_val.y>p2.min_val.y ? p2.min_val.y : p1.min_val.y;
      result.min_val.z=p1.min_val.z>p2.min_val.z ? p2.min_val.z : p1.min_val.z;

      result.max_val.x=p1.max_val.x<p2.max_val.x ? p2.max_val.x : p1.max_val.x;
      result.max_val.y=p1.max_val.y<p2.max_val.y ? p2.max_val.y : p1.max_val.y;
      result.max_val.z=p1.max_val.z<p2.max_val.z ? p2.max_val.z : p1.max_val.z;

      return result;
    }
  };

  /**
   * @brief getMinMax3D
   * @param input
   * @param minPt
   * @param maxPt
   */
  void getMinMax3D(const deviceCloudT& input, pointT& minPt, pointT& maxPt)
  {
    //initialize default values
    if(input.size()<=0)
      return;
    float minV=std::numeric_limits<float>::min();
    float maxV=std::numeric_limits<float>::max();
    minPt.x=minV;
    minPt.y=minV;
    minPt.z=minV;
    maxPt.x=maxV;
    maxPt.y=maxV;
    maxPt.z=maxV;

    minmax_unary_op unary_op;
    minmax_binary_op binary_op;

    //initialize reduction with the first value
    minmax_tuple initpair=unary_op(input[0]);

    //compute minimum and maximum values
    minmax_tuple result=thrust::transform_reduce(
          input.begin(),input.end(),unary_op,initpair,binary_op);

    minPt.x=result.min_val.x;
    minPt.y=result.min_val.y;
    minPt.z=result.min_val.z;
    maxPt.x=result.max_val.x;
    maxPt.y=result.max_val.y;
    maxPt.z=result.max_val.z;
  }

  /**
   * @brief The index_pair struct records the voxel index and point index of the cloud
   */
  struct index_pair
  {
    int voxel_index;
    int point_index;
  };

  /**
   * @brief The index_indices_pair struct
   * index stors the voxel index
   * indices store the indices of points belonging to the voxel
   */
  struct index_indices_pair
  {
    int voxel_index;
    thrust::device_ptr<int> pt_indices;
//    thrust::device_vector<int> pt_indices;
  };

  /**
   * @brief The computeVoxelIndex_functor struct
   */
  struct computeVoxelIndex_functor
  {
    float leafsize;
//    int min_b_0,min_b_1,min_b_2;
//    int divb_mul_0,divb_mul_1,divb_mul_2;
    thrust::device_ptr<int> min_b_;
    thrust::device_ptr<int> divb_mul_;

    __host__ __device__
    index_pair operator () (const pointT& pt,const int& pt_idx) const
    {
      int ijk0=static_cast<int>(std::floor((pt.x)/leafsize))-min_b_[0];
      int ijk1=static_cast<int>(std::floor((pt.y)/leafsize))-min_b_[1];
      int ijk2=static_cast<int>(std::floor((pt.z)/leafsize))-min_b_[2];
      int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];

      index_pair ip;
      ip.point_index=pt_idx;
      ip.voxel_index=idx;
      return ip;
    }
  };


  /**
   * @brief The compareIndexPair_functor struct
   */
  struct compareIndexPair_functor
  {
    __host__ __device__
    bool operator() (const index_pair& ip1, const index_pair& ip2)const
    {
      return (ip1.voxel_index<ip2.voxel_index);
    }
  };

  /**
   * @brief The equalToZero_funcor struct
   */
  struct findEqual_funcor:public thrust::binary_function<index_pair,index_pair,int>
  {
    __host__ __device__
    int operator() (const index_pair& idx1, const index_pair& idx2) const
    {
      if(idx1.voxel_index==idx2.voxel_index)
        return -1;
      else
        return idx1.voxel_index;
    }
  };

  /**
   * @brief The isEqualMinusOne struct
   */
  struct isEqualMinusOne
  {
    __host__ __device__
    bool operator() (int idx)
    {
      return idx==-1;
    }
  };

  struct map_voxel_indices_functor :public thrust::binary_function<int,int,int>
  {
    __host__ __device__
    int operator() (const int& voxel_idx, const int & sequence_idx)const
    {
      if(voxel_idx==-1)
        return -1;
      else
        return sequence_idx;
    }
  };

  __global__ void setouput(float* point_output)
  {
    int idx=threadIdx.x;
    point_output[idx]=idx;
  }
  /**
   * @brief compute_centroid_kernel
   * @param voxel_indices
   * @param len
   * @param full_indices
   * @param pointNum
   * @param point_input
   * @param point_output
   */
//  __global__ void compute_centroid_kernel(int* voxel_indices,
//                                          int len,
//                                          thrust::device_ptr<index_pair> full_indices,
//                                          int pointNum,
//                                          float* point_input,
//                                          char* color_input,
//                                          float* point_output,
//                                          char* color_output
//                                          )
//  {
//    int idx=blockIdx.x*blockDim.x+threadIdx.x;
//    if(idx>len)
//      return;

//    int index=static_cast<int>(voxel_indices[idx]);

//    if(index>=pointNum)
//      return;

//    float cx=0,cy=0,cz=0;
//    char cr=0,cg=0,cb=0;
//    int pt_idx=static_cast<int>((index_pair(full_indices[index])).point_index);

//    if(pt_idx>=pointNum)
//      return;

//    cx=static_cast<float>(point_input[3*pt_idx+0]);
//    cy=static_cast<float>(point_input[3*pt_idx+1]);
//    cz=static_cast<float>(point_input[3*pt_idx+2]);
//    cr=static_cast<char>(color_input[3*pt_idx+0]);
//    cg=static_cast<char>(color_input[3*pt_idx+1]);
//    cb=static_cast<char>(color_input[3*pt_idx+2]);

//    int i = index + 1;
//    while (i < pointNum &&
//           (index_pair(full_indices[i])).voxel_index == (index_pair(full_indices[index])).voxel_index)
//    {
//      int p_idx=static_cast<int>((index_pair(full_indices[i])).point_index);
//      if(p_idx>=pointNum)
//        continue;
//      cx=cx+static_cast<float>(point_input[3*p_idx+0]);
//      cy=cy+static_cast<float>(point_input[3*p_idx+1]);
//      cz=cz+static_cast<float>(point_input[3*p_idx+2]);

//      ++i;
//    }

//    int dist=i-index;
//    if(dist!=0){
//        cx=cx/static_cast<float>(dist);
//        cy=cy/static_cast<float>(dist);
//        cz=cz/static_cast<float>(dist);

//      }

//    point_output[idx*3+0]=cx;
//    point_output[idx*3+1]=cy;
//    point_output[idx*3+2]=cz;
//    color_output[idx*3+0]=cr;
//    color_output[idx*3+1]=cg;
//    color_output[idx*3+2]=cb;

//  }

  __global__ void compute_centroid_kernel(thrust::device_ptr<int> voxel_indices,
                                          int len,
                                          thrust::device_ptr<index_pair> full_indices,
                                          int pointNum,
                                          thrust::device_ptr<pointT> point_input,
                                          thrust::device_ptr<pointT> point_output
                                          )
  {
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>len)
      return;

    int index=static_cast<int>(voxel_indices[idx]);

    if(index>=pointNum)
      return;

    int pt_idx=static_cast<int>((index_pair(full_indices[index])).point_index);

    if(pt_idx>=pointNum)
      return;

    pointT centroid=point_input[pt_idx];

    int i = index + 1;
    while (i < pointNum &&
           (index_pair(full_indices[i])).voxel_index == (index_pair(full_indices[index])).voxel_index)
    {
      int p_idx=static_cast<int>((index_pair(full_indices[i])).point_index);
      if(p_idx>=pointNum)
        continue;
      pointT pt=point_input[p_idx];
      centroid.x+=pt.x;
      centroid.y+=pt.y;
      centroid.z+=pt.z;

      ++i;
    }

    int dist=i-index;
    if(dist!=0){
        centroid.x /= static_cast<float>(dist);
        centroid.y /= static_cast<float>(dist);
        centroid.z /= static_cast<float>(dist);
      }

    point_output[idx]=centroid;

//    __syncthreads();
  }

  /**
   * @brief voxel_grid_filter
   * @param input
   * @param output
   * @param leafsize
   */
  void voxel_grid_filter(const deviceCloudT& input,deviceCloudT& output,const float leafsize)
  {
    //obtain the minimum and maximum x,y,z coordinates
    pointT min_pt=input[0],max_pt=input[0];
    getMinMax3D(input,min_pt,max_pt);

    //check the leafsize's feasibility
    int64_t dx=static_cast<int64_t>((max_pt.x-min_pt.x)/leafsize);
    int64_t dy=static_cast<int64_t>((max_pt.y-min_pt.y)/leafsize);
    int64_t dz=static_cast<int64_t>((max_pt.z-min_pt.z)/leafsize);
    if( (dx*dy*dz) > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) )
    {
       std::cerr<<"ERROR: the leafsize is too small, so that the voxel size will overflow"<<std::endl;
       return;
    }

    //compute the minimum and maximum bounding box values
    thrust::device_ptr<int> min_b_=thrust::device_malloc<int>(3);
    thrust::device_ptr<int> max_b_=thrust::device_malloc<int>(3);
    thrust::device_ptr<int> div_b_=thrust::device_malloc<int>(3);
    thrust::device_ptr<int> divb_mul_=thrust::device_malloc<int>(3);

    min_b_[0]=static_cast<int>(std::floor(min_pt.x/leafsize));
    min_b_[1]=static_cast<int>(std::floor(min_pt.y/leafsize));
    min_b_[2]=static_cast<int>(std::floor(min_pt.z/leafsize));
    max_b_[0]=static_cast<int>(std::floor(max_pt.x/leafsize));
    max_b_[1]=static_cast<int>(std::floor(max_pt.y/leafsize));
    max_b_[2]=static_cast<int>(std::floor(max_pt.z/leafsize));

//    std::cout<<"min_pt: "<<min_pt.x<<" "<<min_pt.y<<" "<<min_pt.z<<std::endl;
//    std::cout<<"min_b_: "<<min_b_[0]<<" "<<min_b_[1]<<" "<<min_b_[2]<<std::endl;

    //compute the number of divisions needed along all axis
    div_b_[0]=max_b_[0]-min_b_[0]+1;
    div_b_[1]=max_b_[1]-min_b_[1]+1;
    div_b_[2]=max_b_[2]-min_b_[2]+1;

    divb_mul_[0]=1;
    divb_mul_[1]=div_b_[0];
    divb_mul_[2]=div_b_[0]*div_b_[1];


//    std::cout<<"divb_mul_: "<<divb_mul_[0]<<" "<<divb_mul_[1]<<" "<<divb_mul_[2]<<std::endl;

    //index_vector stores the point index and voxel index of a point
    thrust::device_vector<index_pair> index_vector(input.size());
    //generate an aux vector that stors the indices of point cloud
    thrust::device_vector<int> point_index_vec(input.size());
    thrust::sequence(point_index_vec.begin(),point_index_vec.end());

    //compute the indices of voxel grids
    computeVoxelIndex_functor cvi_functor;
    cvi_functor.leafsize=leafsize;
    cvi_functor.min_b_=min_b_;
    cvi_functor.divb_mul_=divb_mul_;
//    cvi_functor.min_b_0=min_b_[0];
//    cvi_functor.min_b_1=min_b_[1];
//    cvi_functor.min_b_2=min_b_[2];
//    cvi_functor.divb_mul_0=divb_mul_[0];
//    cvi_functor.divb_mul_1=divb_mul_[1];
//    cvi_functor.divb_mul_2=divb_mul_[2];
    thrust::transform(input.begin(),
                      input.end(),
                      point_index_vec.begin(),
                      index_vector.begin(),
                      cvi_functor);

    thrust::device_free(min_b_);
    thrust::device_free(max_b_);
    thrust::device_free(div_b_);
    thrust::device_free(divb_mul_);

    //sort the index_vector
    thrust::sort(index_vector.begin(),index_vector.end(),compareIndexPair_functor());

    //count ouput cells
    //we need to skip all the same, adjacenet idx values
    thrust::device_vector<index_pair> temp_index_vec(index_vector.size()+1);
    thrust::copy(index_vector.begin(),index_vector.end(),temp_index_vec.begin()+1);
    index_pair ip;
    ip.voxel_index=-1;
    ip.point_index=-1;
    temp_index_vec[0]=ip;
    thrust::device_vector<int> nonequal_voxel_indices(index_vector.size());
    thrust::transform(index_vector.begin(),index_vector.end(),temp_index_vec.begin(),nonequal_voxel_indices.begin(),findEqual_funcor());

    thrust::device_vector<int> voxel_indices(index_vector.size());
    thrust::transform(nonequal_voxel_indices.begin(),nonequal_voxel_indices.end(),point_index_vec.begin(),voxel_indices.begin(),map_voxel_indices_functor());

    thrust::device_vector<int>::iterator it=thrust::remove_if(thrust::device,voxel_indices.begin(),voxel_indices.end(),isEqualMinusOne());
    voxel_indices.resize(it-voxel_indices.begin());

    //input paras
    int len=voxel_indices.size();
    int pt_num=input.size();


//    int* voxel_indices_array_ptr;
//    cudaMalloc(&voxel_indices_array_ptr,len*sizeof(int));
//    int* h_voxel_indices_array_ptr=(int*)malloc(len*sizeof(int));
//    for(int i=0;i<len;++i){
//        h_voxel_indices_array_ptr[i]=voxel_indices[i];
//      }
//    cudaMemcpy(voxel_indices_array_ptr,h_voxel_indices_array_ptr,len*sizeof(int),cudaMemcpyHostToDevice);
//    free(h_voxel_indices_array_ptr);

    thrust::device_ptr<int> voxel_indices_array_ptr=thrust::device_malloc<int>(len);
    thrust::copy(voxel_indices.begin(),voxel_indices.end(),voxel_indices_array_ptr);

    thrust::device_ptr<index_pair> full_indices=thrust::device_malloc<index_pair>(pt_num);
    thrust::copy(thrust::device,index_vector.begin(),index_vector.end(),full_indices);

//    float* point_input;
//    cudaMalloc(&point_input,3*pt_num*sizeof(float));
//    float* h_point_input=(float*)malloc(3*pt_num*sizeof(float));
//    char* color_input;
//    cudaMalloc(&color_input,3*pt_num*sizeof(char));
//    char* h_color_input=(char*)malloc(3*pt_num*sizeof(char));
//    for(int i=0;i<pt_num;++i){
//        h_point_input[3*i+0]=(pointT(input[i])).x;
//        h_point_input[3*i+1]=(pointT(input[i])).y;
//        h_point_input[3*i+2]=(pointT(input[i])).z;
//        h_color_input[3*i+0]=(pointT(input[i])).r;
//        h_color_input[3*i+1]=(pointT(input[i])).g;
//        h_color_input[3*i+2]=(pointT(input[i])).b;
//      }
//    cudaMemcpy(point_input,h_point_input,3*pt_num*sizeof(float),cudaMemcpyHostToDevice);
//    cudaMemcpy(color_input,h_color_input,3*pt_num*sizeof(char),cudaMemcpyHostToDevice);
//    free(h_point_input);
//    free(h_color_input);

    thrust::device_ptr<pointT> point_input=thrust::device_malloc<pointT>(pt_num);
    thrust::copy(input.begin(),input.end(),point_input);

    thrust::device_ptr<pointT> point_output=thrust::device_malloc<pointT>(len);

//    float* point_output;
//    cudaMalloc(&point_output,3*len*sizeof(float));
//    char* color_output;
//    cudaMalloc(&color_output,3*len*sizeof(char));


//    //kernel function
    int BLOCK_SIZE=16;
    dim3 threadsPerBlock(16,16);
    int numBlocks= len/(BLOCK_SIZE*BLOCK_SIZE) + ((len % (BLOCK_SIZE*BLOCK_SIZE))==0 ? 0 : 1);
    std::cout<<"num blocks: "<<numBlocks<<std::endl;
//    compute_centroid_kernel <<< numBlocks , BLOCK_SIZE*BLOCK_SIZE >>> (voxel_indices_array_ptr,
//                                                           len,
//                                                           full_indices,
//                                                           pt_num,
//                                                           point_input,
//                                                           color_input,
//                                                           point_output,
//                                                           color_output
//                                                           );

    compute_centroid_kernel <<<numBlocks,BLOCK_SIZE*BLOCK_SIZE>>> (voxel_indices_array_ptr,len,full_indices,pt_num,point_input,point_output);

    output.clear();
    output.resize(len);
    thrust::copy_n(thrust::device,point_output,len,output.begin());

    thrust::free(thrust::device,voxel_indices_array_ptr);
    thrust::free(thrust::device,full_indices);
    thrust::free(thrust::device,point_input);
    thrust::free(thrust::device,point_output);


////    cudaDeviceSynchronize();
//    float* h_point_cloud=(float*)malloc(3*len*sizeof(float));
//    cudaMemcpy(h_point_cloud,point_output,3*len*sizeof(float),cudaMemcpyDeviceToHost);
//    char* h_color_cloud=(char*)malloc(3*len*sizeof(char));
//    cudaMemcpy(h_color_cloud,color_output,3*len*sizeof(char),cudaMemcpyDeviceToHost);

//    output.resize(len);
//    for(int i=0;i<len;++i){
//        pointT pt;
//        pt.x=static_cast<float>(h_point_cloud[i*3+0]);
//        pt.y=static_cast<float>(h_point_cloud[i*3+1]);
//        pt.z=static_cast<float>(h_point_cloud[i*3+2]);
//        pt.r=static_cast<char>(h_color_cloud[i*3+0]);
//        pt.g=static_cast<char>(h_color_cloud[i*3+1]);
//        pt.b=static_cast<char>(h_color_cloud[i*3+2]);
//        output[i]=pt;
//      }

//  //    std::cout<<"\r\noutput: ";
//  //    for(int i=0;i<len;++i){
//  //        pointT pt=output[i];
//  //        std::cout<<pt.x<<" "<<pt.y<<" "<<pt.z<<std::endl;
//  //      }
//  //    std::cout<<std::endl;

//    thrust::free(thrust::device,full_indices);
//    cudaFree(voxel_indices_array_ptr);
//    cudaFree(point_input);
//    cudaFree(point_output);
//    free(h_point_cloud);
//    free(h_color_cloud);


    //using CPU to compute
//    size_t total = 0;
//    size_t index = 0;
//    thrust::host_vector<pointT> centroid_list;
//    thrust::host_vector<pointT> host_input=input;
//    thrust::host_vector<index_pair> host_index_vector=index_vector;
//    while (index < host_index_vector.size ())
//    {
//        size_t i = index + 1;
//        pointT centroid;
//        int pt_idx=(index_pair(host_index_vector[index])).point_index;
//        pointT pt=(pointT(host_input[pt_idx]));
//        centroid.x=pt.x;
//        centroid.y=pt.y;
//        centroid.z=pt.z;
//        centroid.r=pt.r;
//        centroid.g=pt.g;
//        centroid.b=pt.b;

//        while (i < host_index_vector.size () &&
//               (index_pair(host_index_vector[i])).voxel_index == (index_pair(host_index_vector[index])).voxel_index)
//        {
//          int pt_idx=(index_pair(host_index_vector[i])).point_index;
//          centroid+=(pointT(host_input[pt_idx]));
//          ++i;
//        }

//        int dist=i-index;
//        centroid.x/=static_cast<float>(dist);
//        centroid.y/=static_cast<float>(dist);
//        centroid.z/=static_cast<float>(dist);
//        centroid_list.push_back(centroid);

//        ++total;
//        index = i;
//    }

//    output.clear();
//    output.resize(total);
//    thrust::copy(centroid_list.begin(),centroid_list.end(),output.begin());

  }


}

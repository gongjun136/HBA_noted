#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

#include <mutex>
#include <assert.h>
#include <ros/ros.h>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseArray.h>
#include <tf/transform_broadcaster.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>

#include "ba.hpp"
#include "hba.hpp"
#include "tools.hpp"
#include "mypcl.hpp"

using namespace std;
using namespace Eigen;

// pcd文件名中前缀0的数量(例如，如果pcd文件以 00000.pcd开头, 请填写 5)。默认是 0。
int pcd_name_fill_num = 0;

/**
 * @brief 主要目的是将输入的点云数据（已经经过旋转和平移变换）分割到具有固定大小的体素中，并更新或创建对应的八叉树结构。
 *
 * @param feat_map 体素地图，用于存储体素位置和对应的八叉树根节点
 * @param feat_pt 输入的点云数据
 * @param q 旋转四元数，用于点云的旋转
 * @param t 平移向量，用于点云的平移
 * @param fnum 用于标识当前的帧数或编号
 * @param voxel_size 体素的大小，决定了体素化的粒度
 * @param window_size 窗口大小，可能与时间窗口或其他窗口概念相关
 * @param eigen_ratio 特征比率，可能与特征值或其他数学概念有关
 */
void cut_voxel(unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> &feat_map,
               pcl::PointCloud<PointType> &feat_pt,
               Eigen::Quaterniond q,
               Eigen::Vector3d t, int fnum,
               double voxel_size,
               int window_size,
               float eigen_ratio)
{
  // 临时存储体素中的位置
  float loc_xyz[3];
  // 遍历输入的点云数据
  for (PointType &p_c : feat_pt.points)
  {
    // 使用输入的旋转和平移将点云转换到新的坐标系中
    Eigen::Vector3d pvec_orig(p_c.x, p_c.y, p_c.z);
    Eigen::Vector3d pvec_tran = q * pvec_orig + t;
    // 体素化处理，将转换后的点云数据映射到具有特定尺寸的体素中
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pvec_tran[j] / voxel_size;
      if (loc_xyz[j] < 0)
        loc_xyz[j] -= 1.0;
    }

    // 根据计算的体素位置更新体素地图
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end())
    {
      // 如果该体素位置已经存在于地图中，更新其数据
      iter->second->vec_orig[fnum].push_back(pvec_orig);
      iter->second->vec_tran[fnum].push_back(pvec_tran);

      iter->second->sig_orig[fnum].push(pvec_orig);
      iter->second->sig_tran[fnum].push(pvec_tran);
    }
    else
    {
      // 如果该体素位置不在地图中，创建一个新的八叉树根节点并加入地图
      OCTO_TREE_ROOT *ot = new OCTO_TREE_ROOT(window_size, eigen_ratio);
      ot->vec_orig[fnum].push_back(pvec_orig);
      ot->vec_tran[fnum].push_back(pvec_tran);
      ot->sig_orig[fnum].push(pvec_orig);
      ot->sig_tran[fnum].push(pvec_tran);

      ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
      ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
      ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      ot->layer = 0;
      feat_map[position] = ot;
    }
  }
}

/**
 * @brief 并行处理函数，用于进行一系列的点云处理和优化操作。
 *
 * @param layer 当前层
 * @param thread_id 当前线程的ID
 * @param next_layer 下一层
 */
void parallel_comp(LAYER &layer, int thread_id, LAYER &next_layer)
{
  // 获取当前层的part_length和层数
  int &part_length = layer.part_length;
  int &layer_num = layer.layer_num;

  // 根据线程ID选择一个窗口进行处理
  for (int i = thread_id * part_length; i < (thread_id + 1) * part_length; i++)
  {
    // 初始化点云数组
    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(WIN_SIZE);
    raw_pc.resize(WIN_SIZE);

    // 初始化当前和上一次的残差值
    double residual_cur = 0, residual_pre = 0;
    // 从层中取出pose并存储到缓冲区
    vector<IMUST> x_buf(WIN_SIZE);
    for (int j = 0; j < WIN_SIZE; j++)
    {
      x_buf[j].R = layer.pose_vec[i * GAP + j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i * GAP + j].t;
    }

    // 如果不是第一层，则直接使用原始点云数据
    if (layer_num != 1)
      for (int j = i * GAP; j < i * GAP + WIN_SIZE; j++)
        src_pc[j - i * GAP] = (*layer.pcds[j]).makeShared();

    // 计算当前线程的内存成本
    size_t mem_cost = 0;
    for (int loop = 0; loop < layer.max_iter; loop++)
    {
      // 如果是第一层，则从文件中加载点云数据
      if (layer_num == 1)
      {
        for (int j = i * GAP; j < i * GAP + WIN_SIZE; j++)
        {
          // 是否第一次迭代
          if (loop == 0)
          {
            // 第一次迭代，从文件加载点云数据，存储到raw_pc
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            raw_pc[j - i * GAP] = pc;
          }
          // 将raw_pc中对应的窗口的点云数据放入src_pc中
          src_pc[j - i * GAP] = (*raw_pc[j - i * GAP]).makeShared();
        }
      }

      // 创建体素地图
      unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

      // 遍历窗口，执行体素化操作
      for (size_t j = 0; j < WIN_SIZE; j++)
      {

        if (layer.downsample_size > 0)
        {
          downsample_voxel(*src_pc[j], layer.downsample_size);
        }
        cut_voxel(surf_map, *src_pc[j], Eigen::Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio);
      }
      /// 遍历体素地图并执行体素细分操作
      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();

      // 遍历体素地图，并获取所有的平面体素voxhess
      VOX_HESS voxhess(WIN_SIZE);
      for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->tras_opt(voxhess);

      // 创建优化器并执行优化
      VOX_OPTIMIZER opt_lsv(WIN_SIZE);
      // 使用VOX_OPTIMIZER中的remove_outlier函数去除异常的体素
      // x_buf 是 IMU 状态的缓冲区，voxhess 是体素的Hessian信息，layer.reject_ratio 是需要排除的异常体素的比率
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      // // 创建一个包含6个元素的向量，用于存储Hessian向量信息
      PLV(6) hess_vec;
      // 使用阻尼迭代优化方法（LM优化）进行体素的优化操作,根据最小特征值作为体素的残差优化x_buf的位姿
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
      // 清理体素地图
      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      {
        delete iter->second;
      }

      // 判断是否需要中断迭代
      if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1)
      {
        if (layer.mem_costs[thread_id] < mem_cost)
          layer.mem_costs[thread_id] = mem_cost;

        for (int j = 0; j < WIN_SIZE * (WIN_SIZE - 1) / 2; j++)
          layer.hessians[i * (WIN_SIZE - 1) * WIN_SIZE / 2 + j] = hess_vec[j];

        break;
      }
      residual_pre = residual_cur;
    }

    // 合并点云数据并保存为下一层的关键帧点云
    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);// 创建关键帧点云容器
    // 遍历窗口内的点云并进行变换
    for (size_t j = 0; j < WIN_SIZE; j++)
    {
      // a. 计算变换矩阵
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));
      // b. 变换点云
      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      // c. 合并点云pc_keyframe和pc_oneframe
      pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
    }
    // 窗口内的所有点云最终作为下一层的关键帧点云
    downsample_voxel(*pc_keyframe, 0.05);
    next_layer.pcds[i] = pc_keyframe;
  }
}


void parallel_tail(LAYER &layer, int thread_id, LAYER &next_layer)
{
  int &part_length = layer.part_length;
  int &layer_num = layer.layer_num;
  int &left_gap_num = layer.left_gap_num;

  double load_t = 0, undis_t = 0, dsp_t = 0, cut_t = 0, recut_t = 0, total_t = 0,
         tran_t = 0, sol_t = 0, save_t = 0;

  if (layer.gap_num - (layer.thread_num - 1) * part_length + 1 != left_gap_num)
    printf("THIS IS WRONG!\n");

  for (uint i = thread_id * part_length; i < thread_id * part_length + left_gap_num; i++)
  {
    printf("parallel computing %d\n", i);
    double t0, t1;
    double t_begin = ros::Time::now().toSec();

    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(WIN_SIZE);
    raw_pc.resize(WIN_SIZE);

    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(WIN_SIZE);
    for (int j = 0; j < WIN_SIZE; j++)
    {
      x_buf[j].R = layer.pose_vec[i * GAP + j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i * GAP + j].t;
    }

    if (layer_num != 1)
    {
      t0 = ros::Time::now().toSec();
      for (int j = i * GAP; j < i * GAP + WIN_SIZE; j++)
        src_pc[j - i * GAP] = (*layer.pcds[j]).makeShared();
      load_t += ros::Time::now().toSec() - t0;
    }

    size_t mem_cost = 0;
    for (int loop = 0; loop < layer.max_iter; loop++)
    {
      if (layer_num == 1)
      {
        t0 = ros::Time::now().toSec();
        for (int j = i * GAP; j < i * GAP + WIN_SIZE; j++)
        {
          if (loop == 0)
          {
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            raw_pc[j - i * GAP] = pc;
          }
          src_pc[j - i * GAP] = (*raw_pc[j - i * GAP]).makeShared();
        }
        load_t += ros::Time::now().toSec() - t0;
      }

      unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

      for (size_t j = 0; j < WIN_SIZE; j++)
      {
        t0 = ros::Time::now().toSec();
        if (layer.downsample_size > 0)
          downsample_voxel(*src_pc[j], layer.downsample_size);
        dsp_t += ros::Time::now().toSec() - t0;

        t0 = ros::Time::now().toSec();
        cut_voxel(surf_map, *src_pc[j], Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, WIN_SIZE, layer.eigen_ratio);
        cut_t += ros::Time::now().toSec() - t0;
      }

      t0 = ros::Time::now().toSec();
      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();
      recut_t += ros::Time::now().toSec() - t0;

      t0 = ros::Time::now().toSec();
      VOX_HESS voxhess(WIN_SIZE);
      for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->tras_opt(voxhess);
      tran_t += ros::Time::now().toSec() - t0;

      VOX_OPTIMIZER opt_lsv(WIN_SIZE);
      t0 = ros::Time::now().toSec();
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6)
      hess_vec;
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
      sol_t += ros::Time::now().toSec() - t0;

      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        delete iter->second;

      if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1)
      {
        if (layer.mem_costs[thread_id] < mem_cost)
          layer.mem_costs[thread_id] = mem_cost;

        if (i < thread_id * part_length + left_gap_num)
          for (int j = 0; j < WIN_SIZE * (WIN_SIZE - 1) / 2; j++)
            layer.hessians[i * (WIN_SIZE - 1) * WIN_SIZE / 2 + j] = hess_vec[j];

        break;
      }
      residual_pre = residual_cur;
    }

    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    for (size_t j = 0; j < WIN_SIZE; j++)
    {
      t1 = ros::Time::now().toSec();
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
      save_t += ros::Time::now().toSec() - t1;
    }
    t0 = ros::Time::now().toSec();
    downsample_voxel(*pc_keyframe, 0.05);
    dsp_t += ros::Time::now().toSec() - t0;

    t0 = ros::Time::now().toSec();
    next_layer.pcds[i] = pc_keyframe;
    save_t += ros::Time::now().toSec() - t0;

    total_t += ros::Time::now().toSec() - t_begin;
  }
  if (layer.tail > 0)
  {
    int i = thread_id * part_length + left_gap_num;

    vector<pcl::PointCloud<PointType>::Ptr> src_pc, raw_pc;
    src_pc.resize(layer.last_win_size);
    raw_pc.resize(layer.last_win_size);

    double residual_cur = 0, residual_pre = 0;
    vector<IMUST> x_buf(layer.last_win_size);
    for (int j = 0; j < layer.last_win_size; j++)
    {
      x_buf[j].R = layer.pose_vec[i * GAP + j].q.toRotationMatrix();
      x_buf[j].p = layer.pose_vec[i * GAP + j].t;
    }

    if (layer_num != 1)
    {
      for (int j = i * GAP; j < i * GAP + layer.last_win_size; j++)
        src_pc[j - i * GAP] = (*layer.pcds[j]).makeShared();
    }

    size_t mem_cost = 0;
    for (int loop = 0; loop < layer.max_iter; loop++)
    {
      if (layer_num == 1)
        for (int j = i * GAP; j < i * GAP + layer.last_win_size; j++)
        {
          if (loop == 0)
          {
            pcl::PointCloud<PointType>::Ptr pc(new pcl::PointCloud<PointType>);
            mypcl::loadPCD(layer.data_path, pcd_name_fill_num, pc, j, "pcd/");
            raw_pc[j - i * GAP] = pc;
          }
          src_pc[j - i * GAP] = (*raw_pc[j - i * GAP]).makeShared();
        }

      unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

      for (size_t j = 0; j < layer.last_win_size; j++)
      {
        if (layer.downsample_size > 0)
          downsample_voxel(*src_pc[j], layer.downsample_size);
        cut_voxel(surf_map, *src_pc[j], Quaterniond(x_buf[j].R), x_buf[j].p,
                  j, layer.voxel_size, layer.last_win_size, layer.eigen_ratio);
      }
      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        iter->second->recut();

      VOX_HESS voxhess(layer.last_win_size);
      for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
        iter->second->tras_opt(voxhess);

      VOX_OPTIMIZER opt_lsv(layer.last_win_size);
      opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
      PLV(6)
      hess_vec;
      opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);

      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
        delete iter->second;

      if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1)
      {
        if (layer.mem_costs[thread_id] < mem_cost)
          layer.mem_costs[thread_id] = mem_cost;

        for (int j = 0; j < layer.last_win_size * (layer.last_win_size - 1) / 2; j++)
          layer.hessians[i * (WIN_SIZE - 1) * WIN_SIZE / 2 + j] = hess_vec[j];

        break;
      }
      residual_pre = residual_cur;
    }

    pcl::PointCloud<PointType>::Ptr pc_keyframe(new pcl::PointCloud<PointType>);
    for (size_t j = 0; j < layer.last_win_size; j++)
    {
      Eigen::Quaterniond q_tmp;
      Eigen::Vector3d t_tmp;
      assign_qt(q_tmp, t_tmp, Quaterniond(x_buf[0].R.inverse() * x_buf[j].R),
                x_buf[0].R.inverse() * (x_buf[j].p - x_buf[0].p));

      pcl::PointCloud<PointType>::Ptr pc_oneframe(new pcl::PointCloud<PointType>);
      mypcl::transform_pointcloud(*src_pc[j], *pc_oneframe, t_tmp, q_tmp);
      pc_keyframe = mypcl::append_cloud(pc_keyframe, *pc_oneframe);
    }
    downsample_voxel(*pc_keyframe, 0.05);
    next_layer.pcds[i] = pc_keyframe;
  }
  printf("total time: %.2fs\n", total_t);
  printf("load pcd %.2fs %.2f%% | undistort pcd %.2fs %.2f%% | "
         "downsample %.2fs %.2f%% | cut voxel %.2fs %.2f%% | recut %.2fs %.2f%% | trans %.2fs %.2f%% | solve %.2fs %.2f%% | "
         "save pcd %.2fs %.2f%%\n",
         load_t, load_t / total_t * 100, undis_t, undis_t / total_t * 100,
         dsp_t, dsp_t / total_t * 100, cut_t, cut_t / total_t * 100, recut_t, recut_t / total_t * 100, tran_t, tran_t / total_t * 100,
         sol_t, sol_t / total_t * 100, save_t, save_t / total_t * 100);
}

void global_ba(LAYER &layer)
{
  int window_size = layer.pose_vec.size();
  vector<IMUST> x_buf(window_size);
  for (int i = 0; i < window_size; i++)
  {
    x_buf[i].R = layer.pose_vec[i].q.toRotationMatrix();
    x_buf[i].p = layer.pose_vec[i].t;
  }

  vector<pcl::PointCloud<PointType>::Ptr> src_pc;
  src_pc.resize(window_size);
  for (int i = 0; i < window_size; i++)
    src_pc[i] = (*layer.pcds[i]).makeShared();

  double residual_cur = 0, residual_pre = 0;
  size_t mem_cost = 0, max_mem = 0;
  double dsp_t = 0, cut_t = 0, recut_t = 0, tran_t = 0, sol_t = 0, t0;
  for (int loop = 0; loop < layer.max_iter; loop++)
  {
    std::cout << "---------------------" << std::endl;
    std::cout << "Iteration " << loop << std::endl;

    unordered_map<VOXEL_LOC, OCTO_TREE_ROOT *> surf_map;

    for (int i = 0; i < window_size; i++)
    {
      t0 = ros::Time::now().toSec();
      if (layer.downsample_size > 0)
        downsample_voxel(*src_pc[i], layer.downsample_size);
      dsp_t += ros::Time::now().toSec() - t0;
      t0 = ros::Time::now().toSec();
      cut_voxel(surf_map, *src_pc[i], Quaterniond(x_buf[i].R), x_buf[i].p, i,
                layer.voxel_size, window_size, layer.eigen_ratio * 2);
      cut_t += ros::Time::now().toSec() - t0;
    }
    t0 = ros::Time::now().toSec();
    for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->recut();
    recut_t += ros::Time::now().toSec() - t0;

    t0 = ros::Time::now().toSec();
    VOX_HESS voxhess(window_size);
    for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->tras_opt(voxhess);
    tran_t += ros::Time::now().toSec() - t0;

    t0 = ros::Time::now().toSec();
    VOX_OPTIMIZER opt_lsv(window_size);
    opt_lsv.remove_outlier(x_buf, voxhess, layer.reject_ratio);
    PLV(6)
    hess_vec;
    opt_lsv.damping_iter(x_buf, voxhess, residual_cur, hess_vec, mem_cost);
    sol_t += ros::Time::now().toSec() - t0;

    for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      delete iter->second;

    cout << "Residual absolute: " << abs(residual_pre - residual_cur) << " | "
         << "percentage: " << abs(residual_pre - residual_cur) / abs(residual_cur) << endl;

    if (loop > 0 && abs(residual_pre - residual_cur) / abs(residual_cur) < 0.05 || loop == layer.max_iter - 1)
    {
      if (max_mem < mem_cost)
        max_mem = mem_cost;
#ifdef FULL_HESS
      for (int i = 0; i < window_size * (window_size - 1) / 2; i++)
        layer.hessians[i] = hess_vec[i];
#else
      for (int i = 0; i < window_size - 1; i++)
      {
        Matrix6d hess = Hess_cur.block(6 * i, 6 * i + 6, 6, 6);
        for (int row = 0; row < 6; row++)
          for (int col = 0; col < 6; col++)
            hessFile << hess(row, col) << ((row * col == 25) ? "" : " ");
        if (i < window_size - 2)
          hessFile << "\n";
      }
#endif
      break;
    }
    residual_pre = residual_cur;
  }
  for (int i = 0; i < window_size; i++)
  {
    layer.pose_vec[i].q = Quaterniond(x_buf[i].R);
    layer.pose_vec[i].t = x_buf[i].p;
  }
  printf("Downsample: %f, Cut: %f, Recut: %f, Tras: %f, Sol: %f\n", dsp_t, cut_t, recut_t, tran_t, sol_t);
}

/**
 * @brief 创建并管理多线程，以实现并行计算任务
 *
 * @param layer 当前层
 * @param next_layer 下一层
 */
void distribute_thread(LAYER &layer, LAYER &next_layer)
{
  // 获取当前层的线程数
  int &thread_num = layer.thread_num;
  // 记录当前时间
  double t0 = ros::Time::now().toSec();

  // 遍历线程
  for (int i = 0; i < thread_num; i++)
  {
    // 判断是否为最后一个线程
    if (i < thread_num - 1)
      // 如果不是最后一个线程，使用parallel_comp函数作为线程任务
      layer.mthreads[i] = new thread(parallel_comp, ref(layer), i, ref(next_layer));
    else
      // 如果是最后一个线程，使用parallel_tail函数作为线程任务
      layer.mthreads[i] = new thread(parallel_tail, ref(layer), i, ref(next_layer));
    // 可以在此打印线程分配所花费的时间
    // printf("Thread distribution time: %f\n", ros::Time::now().toSec()-t0);
  }

  // 重新记录当前时间
  t0 = ros::Time::now().toSec();

  // 等待所有线程完成任务
  for (int i = 0; i < thread_num; i++)
  {
    layer.mthreads[i]->join();
    // 释放线程资源
    delete layer.mthreads[i];
  }
  // 可以在此打印线程完成任务所花费的时间
  // printf("Thread join time: %f\n", ros::Time::now().toSec()-t0);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "hba");
  ros::NodeHandle nh("~");

  google::InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "/home/gj/catkin_ws_HBA/src/HBA/log";
  // 声明变量来存储参数。
  int total_layer_num, thread_num;
  string data_path;
  // 从ROS参数服务器检索参数
  nh.getParam("total_layer_num", total_layer_num);     // 获取层数总计
  nh.getParam("pcd_name_fill_num", pcd_name_fill_num); // 获取pcd文件名的填充数字
  nh.getParam("data_path", data_path);                 // 获取数据的路径。
  nh.getParam("thread_num", thread_num);               // 获取要使用的线程数。
  // 使用给定的参数初始化层次化的bundle adjustment (HBA)。
  HBA hba(total_layer_num, data_path, thread_num);
  // 遍历除最后一层的所有层。
  for (int i = 0; i < total_layer_num - 1; i++)
  {
    std::cout << "---------------------" << std::endl;
    // 为当前层和下一层在多个线程之间分配计算任务。
    distribute_thread(hba.layers[i], hba.layers[i + 1]);
    // 更新下一层的状态。
    hba.update_next_layer_state(i);
  }
  // 在最后一层执行全局bundle adjustment。
  global_ba(hba.layers[total_layer_num - 1]);
  // 优化位姿图。
  hba.pose_graph_optimization();
  // 迭代完成
  printf("iteration complete\n");
  google::ShutdownGoogleLogging();
}
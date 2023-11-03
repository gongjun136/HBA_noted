#ifndef HBA_HPP
#define HBA_HPP
#include <glog/logging.h>
#include <thread>
#include <fstream>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "mypcl.hpp"
#include "tools.hpp"
#include "ba.hpp"

class LAYER
{
public:
  int pose_size,  //pose的数量
      layer_num, // 当前层数
      max_iter,  // 最大迭代次数，默认值10
      // part_length表示的是每个线程应该处理的间隔数量。
      // 在多线程处理中，为了平衡每个线程的工作量，整个pose序列被分成多个部分，每个部分由一个线程处理。
      // part_length是通过(gap_num + 1) / thread_num计算得到的，表示每个线程应该处理的间隔数量。
      part_length,
      // left_size表示的是最后一个线程处理的pose数量。它取决于left_gap_num和tail。
      // 如果tail为0，表示最后一个窗口是完整的，那么left_size就是left_gap_num乘以WIN_SIZE。否则，它还需要加上GAP和tail。
      left_size,
      // left_h_size表示的是最后一个线程处理的Hessian矩阵的大小。它与left_size类似，但考虑的是Hessian矩阵的大小，而不是pose的数量。
      left_h_size,
      // j_upper表示的是最后一个线程处理的最后一个间隔的索引。它与left_gap_num和tail有关。
      // 如果tail为0，那么j_upper就是gap_num - (thread_num - 1) * part_length + 1。否则，它需要加1。
      j_upper,
      // 表示的是在将所有的pose分成多个窗口后，最后一个窗口后剩余的pose数量
      // 它是通过取pose_size与WIN_SIZE的模得到的。如果tail为0，说明最后一个窗口是完整的；如果不为0，说明最后一个窗口的大小可能小于WIN_SIZE。
      tail,
      thread_num,
      // gap_num表示的是在所有pose中，两个连续窗口之间的间隔数量。这个间隔是固定的，由GAP定义。
      // gap_num是通过(pose_size - WIN_SIZE) / GAP计算得到的，表示整个pose序列可以被划分成多少个这样的间隔。
      gap_num,
      // last_win_size表示的是最后一个窗口的大小。
      // 由于最后一个窗口可能不完整（即小于WIN_SIZE），所以需要单独计算其大小。它是通过pose_size - GAP * (gap_num + 1)得到的。
      last_win_size,
      // 表示的是最后一个线程需要处理的间隔数量
      left_gap_num;
  double downsample_size, voxel_size, eigen_ratio, reject_ratio;

  std::string data_path;          // 数据路径
  vector<mypcl::pose> pose_vec;   // pose_vec是一个pose数组，用于保存所有的pose信息。
  std::vector<thread *> mthreads; // mthreads用于保存多线程操作中的线程对象
  std::vector<double> mem_costs;  // mem_costs用于保存每个线程的内存消耗

  std::vector<VEC(6)> hessians;
  std::vector<pcl::PointCloud<PointType>::Ptr> pcds; // pcds是一个点云数组，用于保存当前层的关键帧点云数据。

  LAYER()
  {
    pose_size = 0;            //
    layer_num = 1;            //
    max_iter = 10;            // LM优化的最大迭代次数
    downsample_size = 0.1;    // 体素化前的体素下采样大小
    voxel_size = 4.0;         // 窗口内所有帧划分体素的初始大小
    eigen_ratio = 0.1;        // 平面检测的最大阈值:最小特征值/最大特征值
    reject_ratio = 0.05;      // 排除的异常体素的比率
    pose_vec.clear();
    mthreads.clear();
    pcds.clear();
    hessians.clear();
    mem_costs.clear();
  }

  /**
   * @brief 为存储不同数据结构分配内存空间，并初始化某些数据结构的大小。
   * 特别重点是根据不同的层来计算并设置`hessian_size`，这是后续操作中非常关键的参数。
   *
   * @param total_layer_num_ 总的层数
   */
  void init_storage(int total_layer_num_)
  {
    // 根据线程数调整mthreads的大小。mthreads用于保存多线程操作中的线程对象。
    mthreads.resize(thread_num);
    // 根据线程数调整mem_costs的大小。mem_costs用于保存每个线程的内存消耗。
    mem_costs.resize(thread_num);

    // 根据pose的数量调整pcds的大小。pcds是一个点云数组，用于保存不同pose的点云数据。
    pcds.resize(pose_size);
    // 根据pose的数量调整pose_vec的大小。pose_vec是一个pose数组，用于保存所有的pose信息。
    pose_vec.resize(pose_size);

#ifdef FULL_HESS
    // 依据当前层的数量与总层数比较，来计算hessian矩阵的大小。
    if (layer_num < total_layer_num_)
    {
      // 如果当前层数小于总层数，那么我们需要按照特定的公式来计算hessian_size。
      int hessian_size = (thread_num - 1) * (WIN_SIZE - 1) * WIN_SIZE / 2 * part_length;
      hessian_size += (WIN_SIZE - 1) * WIN_SIZE / 2 * left_gap_num;
      if (tail > 0)
        hessian_size += (last_win_size - 1) * last_win_size / 2;
      hessians.resize(hessian_size); // hessians保存Hessian矩阵的信息。
      printf("hessian_size: %d\n", hessian_size);
    }
    else
    {
      // 如果当前层数不小于总层数，那么我们简单地计算hessian_size。
      int hessian_size = pose_size * (pose_size - 1) / 2;
      hessians.resize(hessian_size);
      printf("hessian_size: %d\n", hessian_size);
    }
#endif
    // 初始化mem_costs的所有元素为0。
    for (int i = 0; i < thread_num; i++)
      mem_costs.push_back(0);
  }

  /**
   * @brief 初始化层的参数。根据层的数量以及pose的数量，来计算并初始化一些关键参数，如`gap_num`和`part_length`(每个线程的gap数)。
   *
   * @param pose_size_ 默认为0，当前层的pose的数量。对于第一层，其值是总的pose数，而对于其他层，是传入的参数。
   */
  void init_parameter(int pose_size_ = 0)
  {
    // 如果是第一层，则pose_size为pose_vec的大小，即总的pose数。
    // 否则，pose_size为传入的pose_size_。
    if (layer_num == 1)
      pose_size = pose_vec.size();
    else
      pose_size = pose_size_;

    // 根据pose_size计算一些关键参数，例如tail, gap_num, last_win_size等。
    tail = (pose_size - WIN_SIZE) % GAP;
    gap_num = (pose_size - WIN_SIZE) / GAP;
    last_win_size = pose_size - GAP * (gap_num + 1);
    part_length = ceil((gap_num + 1) / double(thread_num));

    // 在多线程处理中，为了确保每个线程的工作量大致相同并避免某些线程处理的数据量过大或过小，可能需要对每个线程处理的数据量进行调整。
    // 这就是part_length调整的目的

    // 初始的part_length是通过整体的gap_num与线程数量thread_num的比值来计算的。这意味着每个线程应该处理大致相同的间隔数量。
    // 但是，由于gap_num可能不能被thread_num整除，所以可能会有一些剩余的间隔，这些间隔需要被分配给某些线程处理。
    // 这个初步计算可能会导致某些线程处理的数据量过大或过小。
    if (gap_num - (thread_num - 1) * part_length < 0)
      part_length = floor((gap_num + 1) / double(thread_num));
    // 如果part_length为0，这意味着间隔数量太少，不能均匀分配给所有线程。在这种情况下，我们需要减少线程数量。
    // gap_num - (thread_num - 1) * part_length，我们可以计算出最后一个线程需要处理的间隔数量
    // 如果最后一个线程处理的间隔数量与part_length的比值超过2，这意味着最后一个线程的工作量过大。为了修正这个问题，我们再次减少线程数量。
    while (part_length == 0 || (gap_num - (thread_num - 1) * part_length + 1) / double(part_length) > 2)
    {
      thread_num -= 1;
      part_length = ceil((gap_num + 1) / double(thread_num));
      if (gap_num - (thread_num - 1) * part_length < 0)
        part_length = floor((gap_num + 1) / double(thread_num));
    }
    // left_gap_num表示的是最后一个线程需要处理的间隔数量。由于前面的线程都处理part_length数量的间隔，所以最后一个线程可能需要处理的间隔数量与其他线程不同。
    // 它是通过gap_num - (thread_num - 1) * part_length计算得到的。
    left_gap_num = gap_num - (thread_num - 1) * part_length + 1;
    // 根据tail的值，进一步调整left_size, left_h_size和j_upper的值。
    if (tail == 0)
    {
      left_size = (gap_num - (thread_num - 1) * part_length + 1) * WIN_SIZE;
      left_h_size = (gap_num - (thread_num - 1) * part_length) * GAP + WIN_SIZE - 1;
      j_upper = gap_num - (thread_num - 1) * part_length + 1;
    }
    else
    {
      left_size = (gap_num - (thread_num - 1) * part_length + 1) * WIN_SIZE + GAP + tail;
      left_h_size = (gap_num - (thread_num - 1) * part_length + 1) * GAP + last_win_size - 1;
      j_upper = gap_num - (thread_num - 1) * part_length + 2;
    }
    // 打印出关键的参数值，帮助开发者进行调试。
    LOG(INFO) << "init parameter:\n";
    LOG(INFO) << "layer_num " << layer_num << " | thread_num " << thread_num << " | pose_size " << pose_size
              << " | max_iter " << max_iter << " | part_length " << part_length << " | gap_num " << gap_num
              << " | last_win_size " << last_win_size << " | left_gap_num " << left_gap_num << " | tail " << tail
              << " | left_size " << left_size << " | left_h_size " << left_h_size << " | j_upper " << j_upper
              << " | downsample_size " << downsample_size << " | voxel_size " << voxel_size << " | eigen_ratio "
              << eigen_ratio << " | reject_ratio " << reject_ratio;
  }
};

/**
 * @brief HBA类
 *
 */
class HBA
{
public:
  int thread_num, total_layer_num;
  std::vector<LAYER> layers; // 层数组
  std::string data_path;

  /**
   * @brief HBA构造函数的目的是为了初始化层次化bundle adjustment（HBA）的参数
   * 对于第一层，它直接从给定的数据路径读取pose数据并初始化相应的参数。
   * 对于后续的层，它基于前一层的数据和参数进行初始化。
   *
   * @param total_layer_num_ 总层数
   * @param data_path_ 数据路径
   * @param thread_num_ 线程数
   */
  HBA(int total_layer_num_, std::string data_path_, int thread_num_)
  {
    // 初始化参数
    total_layer_num = total_layer_num_;
    thread_num = thread_num_;
    data_path = data_path_;

    // 根据总层数初始化层的数组
    layers.resize(total_layer_num);
    for (int i = 0; i < total_layer_num; i++)
    {
      layers[i].layer_num = i + 1;       // 设置当前层的编号
      layers[i].thread_num = thread_num; // 设置当前层的线程数
    }
    // 读取第一层的pose数据并初始化相关参数
    layers[0].data_path = data_path;
    layers[0].pose_vec = mypcl::read_pose(data_path + "pose.json");
    layers[0].init_parameter();
    layers[0].init_storage(total_layer_num);
    // 根据第一层的参数，初始化其他层的参数
    for (int i = 1; i < total_layer_num; i++)
    {
      int pose_size_ = (layers[i - 1].thread_num - 1) * layers[i - 1].part_length;
      pose_size_ += layers[i - 1].tail == 0 ? layers[i - 1].left_gap_num : (layers[i - 1].left_gap_num + 1);
      layers[i].init_parameter(pose_size_);
      layers[i].init_storage(total_layer_num);
      layers[i].data_path = layers[i - 1].data_path + "process1/"; // 设置数据路径
    }
    printf("HBA init done!\n"); // 完成HBA的初始化
  }

  void update_next_layer_state(int cur_layer_num)
  {
    for (int i = 0; i < layers[cur_layer_num].thread_num; i++)
      if (i < layers[cur_layer_num].thread_num - 1)
        for (int j = 0; j < layers[cur_layer_num].part_length; j++)
        {
          int index = (i * layers[cur_layer_num].part_length + j) * GAP;
          layers[cur_layer_num + 1].pose_vec[i * layers[cur_layer_num].part_length + j] = layers[cur_layer_num].pose_vec[index];
        }
      else
        for (int j = 0; j < layers[cur_layer_num].j_upper; j++)
        {
          int index = (i * layers[cur_layer_num].part_length + j) * GAP;
          layers[cur_layer_num + 1].pose_vec[i * layers[cur_layer_num].part_length + j] = layers[cur_layer_num].pose_vec[index];
        }
  }

  void pose_graph_optimization()
  {
    std::vector<mypcl::pose> upper_pose, init_pose;
    upper_pose = layers[total_layer_num - 1].pose_vec;
    init_pose = layers[0].pose_vec;
    std::vector<VEC(6)> upper_cov, init_cov;
    upper_cov = layers[total_layer_num - 1].hessians;
    init_cov = layers[0].hessians;

    int cnt = 0;
    gtsam::Values initial;
    gtsam::NonlinearFactorGraph graph;
    gtsam::Vector Vector6(6);
    Vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8;
    gtsam::noiseModel::Diagonal::shared_ptr priorModel = gtsam::noiseModel::Diagonal::Variances(Vector6);
    initial.insert(0, gtsam::Pose3(gtsam::Rot3(init_pose[0].q.toRotationMatrix()), gtsam::Point3(init_pose[0].t)));
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(gtsam::Rot3(init_pose[0].q.toRotationMatrix()), gtsam::Point3(init_pose[0].t)), priorModel));

    for (uint i = 0; i < init_pose.size(); i++)
    {
      if (i > 0)
        initial.insert(i, gtsam::Pose3(gtsam::Rot3(init_pose[i].q.toRotationMatrix()), gtsam::Point3(init_pose[i].t)));

      if (i % GAP == 0)
        for (int j = 0; j < WIN_SIZE - 1; j++)
          for (int k = j + 1; k < WIN_SIZE; k++)
          {
            if (i + j + 1 >= init_pose.size() || i + k >= init_pose.size())
              break;

            cnt++;
            if (init_cov[cnt - 1].norm() < 1e-20)
              continue;

            Eigen::Vector3d t_ab = init_pose[i + j].t;
            Eigen::Matrix3d R_ab = init_pose[i + j].q.toRotationMatrix();
            t_ab = R_ab.transpose() * (init_pose[i + k].t - t_ab);
            R_ab = R_ab.transpose() * init_pose[i + k].q.toRotationMatrix();
            gtsam::Rot3 R_sam(R_ab);
            gtsam::Point3 t_sam(t_ab);

            Vector6 << fabs(1.0 / init_cov[cnt - 1](0)), fabs(1.0 / init_cov[cnt - 1](1)), fabs(1.0 / init_cov[cnt - 1](2)),
                fabs(1.0 / init_cov[cnt - 1](3)), fabs(1.0 / init_cov[cnt - 1](4)), fabs(1.0 / init_cov[cnt - 1](5));
            gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
            gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(i + j, i + k, gtsam::Pose3(R_sam, t_sam),
                                                                                             odometryNoise));
            graph.push_back(factor);
          }
    }

    int pose_size = upper_pose.size();
    cnt = 0;
    for (int i = 0; i < pose_size - 1; i++)
      for (int j = i + 1; j < pose_size; j++)
      {
        cnt++;
        if (upper_cov[cnt - 1].norm() < 1e-20)
          continue;

        Eigen::Vector3d t_ab = upper_pose[i].t;
        Eigen::Matrix3d R_ab = upper_pose[i].q.toRotationMatrix();
        t_ab = R_ab.transpose() * (upper_pose[j].t - t_ab);
        R_ab = R_ab.transpose() * upper_pose[j].q.toRotationMatrix();
        gtsam::Rot3 R_sam(R_ab);
        gtsam::Point3 t_sam(t_ab);

        Vector6 << fabs(1.0 / upper_cov[cnt - 1](0)), fabs(1.0 / upper_cov[cnt - 1](1)), fabs(1.0 / upper_cov[cnt - 1](2)),
            fabs(1.0 / upper_cov[cnt - 1](3)), fabs(1.0 / upper_cov[cnt - 1](4)), fabs(1.0 / upper_cov[cnt - 1](5));
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);
        gtsam::NonlinearFactor::shared_ptr factor(new gtsam::BetweenFactor<gtsam::Pose3>(i * pow(GAP, total_layer_num - 1),
                                                                                         j * pow(GAP, total_layer_num - 1), gtsam::Pose3(R_sam, t_sam), odometryNoise));
        graph.push_back(factor);
      }

    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    gtsam::ISAM2 isam(parameters);
    isam.update(graph, initial);
    isam.update();

    gtsam::Values results = isam.calculateEstimate();

    cout << "vertex size " << results.size() << endl;

    for (uint i = 0; i < results.size(); i++)
    {
      gtsam::Pose3 pose = results.at(i).cast<gtsam::Pose3>();
      assign_qt(init_pose[i].q, init_pose[i].t, Eigen::Quaterniond(pose.rotation().matrix()), pose.translation());
    }
    mypcl::write_pose(init_pose, data_path);
    printf("pgo complete\n");
  }
};

#endif
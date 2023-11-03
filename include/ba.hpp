#ifndef BA_HPP
#define BA_HPP

#include <thread>
#include <fstream>
#include <iomanip>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "tools.hpp"

#define WIN_SIZE 10 // 局部BA使用的窗口大小
#define GAP 5       // 每两个相邻窗口开始之间的步幅大小
#define AVG_THR
#define FULL_HESS
// #define ENABLE_RVIZ
// #define ENABLE_FILTER

const double one_three = (1.0 / 3.0);

int layer_limit = 2;
int MIN_PT = 15; // 体素最小点数
int thd_num = 16;

// 存储和操作体素地图中的信息，并执行一些基于体素特征的优化存储和操作体素地图中的信息，并执行一些基于体素特征的优化
class VOX_HESS
{
public:
  // 存储了指向VOX_FACTOR向量的指针。每一个VOX_FACTOR向量都代表一个体素在一个给定的窗口内的统计信息
  vector<const vector<VOX_FACTOR> *> plvec_voxels;
  vector<PLV(3)> origin_points;
  int win_size;

  VOX_HESS(int _win_size = WIN_SIZE) : win_size(_win_size) { origin_points.resize(win_size); }

  ~VOX_HESS()
  {
    vector<const vector<VOX_FACTOR> *>().swap(plvec_voxels);
  }

  void get_center(const PLV(3) & vec_orig, PLV(3) & origin_points_)
  {
    size_t pt_size = vec_orig.size();
    for (size_t i = 0; i < pt_size; i++)
      origin_points_.emplace_back(vec_orig[i]);
    return;
  }

  void push_voxel(const vector<VOX_FACTOR> *sig_orig, const vector<PLV(3)> *vec_orig)
  {
    int process_size = 0;
    // 计算在给定窗口内有多少帧的数据是有效的（即其点的数量不为0）
    for (int i = 0; i < win_size; i++)
      if ((*sig_orig)[i].N != 0)
        process_size++;

#ifdef ENABLE_FILTER
    // 如果当前体素在给定的窗口内只有少于1帧的有效数据，则直接返回
    if (process_size < 1)
      return;
    // 对于有有效数据的每一帧，从原始点云数据vec_orig中提取中心点，并保存到origin_points中
    for (int i = 0; i < win_size; i++)
      if ((*sig_orig)[i].N != 0)
        get_center((*vec_orig)[i], origin_points[i]);
#endif
    // 如果当前体素在给定的时间窗口内只有少于2帧的有效数据，则不进一步处理
    if (process_size < 2)
      return;
    // 将当前体素的原始统计信息sig_orig添加到plvec_voxels中
    plvec_voxels.push_back(sig_orig);
  }

  Eigen::Matrix<double, 6, 1> lam_f(Eigen::Vector3d *u, int m, int n)
  {
    Eigen::Matrix<double, 6, 1> jac;
    jac[0] = u[m][0] * u[n][0];
    jac[1] = u[m][0] * u[n][1] + u[m][1] * u[n][0];
    jac[2] = u[m][0] * u[n][2] + u[m][2] * u[n][0];
    jac[3] = u[m][1] * u[n][1];
    jac[4] = u[m][1] * u[n][2] + u[m][2] * u[n][1];
    jac[5] = u[m][2] * u[n][2];
    return jac;
  }

  /**
   * @brief 并行地计算与体素相关的Hessian矩阵、雅可比向量和残差。这是为了在大规模优化问题中提高效率。
   * 
   * @param xs 包含位姿等状态信息的向量
   * @param head 用于指定该线程应处理的体素范围
   * @param end 用于指定该线程应处理的体素范围
   * @param Hess 输出的Hessian矩阵
   * @param JacT 输出的雅可比向量
   * @param residual 输出的残差
   */
  void acc_evaluate2(const vector<IMUST> &xs, int head, int end,
                     Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    // 初始化
    Hess.setZero();
    JacT.setZero();
    residual = 0;
    vector<VOX_FACTOR> sig_tran(win_size);
    const int kk = 0;

    PLV(3)
    viRiTuk(win_size);
    PLM(3)
    viRiTukukT(win_size);

    vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT;

    // 遍历head到end范围内的每一个体素
    for (int a = head; a < end; a++)
    {
      const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[a];

      VOX_FACTOR sig;
      for (int i = 0; i < win_size; i++)
        if (sig_orig[i].N != 0)
        {
          sig_tran[i].transform(sig_orig[i], xs[i]);
          sig += sig_tran[i];
        }

      const Eigen::Vector3d &vBar = sig.v / sig.N;
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P / sig.N - vBar * vBar.transpose());
      const Eigen::Vector3d &lmbd = saes.eigenvalues();
      const Eigen::Matrix3d &U = saes.eigenvectors();
      int NN = sig.N;

      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};

      const Eigen::Vector3d &uk = u[kk];
      Eigen::Matrix3d ukukT = uk * uk.transpose();
      umumT.setZero();
      for (int i = 0; i < 3; i++)
        if (i != kk)
          umumT += 2.0 / (lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      for (int i = 0; i < win_size; i++)
        if (sig_orig[i].N != 0)
        {
          Eigen::Matrix3d Pi = sig_orig[i].P;
          Eigen::Vector3d vi = sig_orig[i].v;
          Eigen::Matrix3d Ri = xs[i].R;
          double ni = sig_orig[i].N;

          Eigen::Matrix3d vihat;
          vihat << SKEW_SYM_MATRX(vi);
          Eigen::Vector3d RiTuk = Ri.transpose() * uk;
          Eigen::Matrix3d RiTukhat;
          RiTukhat << SKEW_SYM_MATRX(RiTuk);

          Eigen::Vector3d PiRiTuk = Pi * RiTuk;
          viRiTuk[i] = vihat * RiTuk;
          viRiTukukT[i] = viRiTuk[i] * uk.transpose();

          Eigen::Vector3d ti_v = xs[i].p - vBar;
          double ukTti_v = uk.dot(ti_v);

          Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
          Eigen::Vector3d combo2 = Ri * vi + ni * ti_v;
          Auk[i].block<3, 3>(0, 0) = (Ri * Pi + ti_v * vi.transpose()) * RiTukhat - Ri * combo1;
          Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
          Auk[i] /= NN;

          const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
          JacT.block<6, 1>(6 * i, 0) += jjt;

          const Eigen::Matrix3d &HRt = 2.0 / NN * (1.0 - ni / NN) * viRiTukukT[i];
          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
          Hb.block<3, 3>(0, 0) +=
              2.0 / NN * (combo1 - RiTukhat * Pi) * RiTukhat - 2.0 / NN / NN * viRiTuk[i] * viRiTuk[i].transpose() - 0.5 * hat(jjt.block<3, 1>(0, 0));
          Hb.block<3, 3>(0, 3) += HRt;
          Hb.block<3, 3>(3, 0) += HRt.transpose();
          Hb.block<3, 3>(3, 3) += 2.0 / NN * (ni - ni * ni / NN) * ukukT;

          Hess.block<6, 6>(6 * i, 6 * i) += Hb;
        }

      for (int i = 0; i < win_size - 1; i++)
        if (sig_orig[i].N != 0)
        {
          double ni = sig_orig[i].N;
          for (int j = i + 1; j < win_size; j++)
            if (sig_orig[j].N != 0)
            {
              double nj = sig_orig[j].N;
              Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
              Hb.block<3, 3>(0, 0) += -2.0 / NN / NN * viRiTuk[i] * viRiTuk[j].transpose();
              Hb.block<3, 3>(0, 3) += -2.0 * nj / NN / NN * viRiTukukT[i];
              Hb.block<3, 3>(3, 0) += -2.0 * ni / NN / NN * viRiTukukT[j].transpose();
              Hb.block<3, 3>(3, 3) += -2.0 * ni * nj / NN / NN * ukukT;

              Hess.block<6, 6>(6 * i, 6 * j) += Hb;
            }
        }

      residual += lmbd[kk];
    }

    for (int i = 1; i < win_size; i++)
      for (int j = 0; j < i; j++)
        Hess.block<6, 6>(6 * i, 6 * j) = Hess.block<6, 6>(6 * j, 6 * i).transpose();
  }

  void evaluate_only_residual(const vector<IMUST> &xs, double &residual)
  {
    residual = 0;
    vector<VOX_FACTOR> sig_tran(win_size);
    int kk = 0; // The kk-th lambda value

    int gps_size = plvec_voxels.size();

    for (int a = 0; a < gps_size; a++)
    {
      const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[a];
      VOX_FACTOR sig;

      for (int i = 0; i < win_size; i++)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }

      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      residual += lmbd[kk];
    }
  }
  /**
   * @brief 主要目的是计算残差，并将其返回。该残差是基于体素的协方差的最小特征值
   * 对于平面体素，最小的特征值是一个很好的残差度量，用于表示点在平面的法线方向上的分布
   *
   * @param xs 窗口内的状态
   * @return std::vector<double> 计算每个体素的残差
   */
  std::vector<double> evaluate_residual(const vector<IMUST> &xs)
  {
    /* for outlier removal usage */
    std::vector<double> residuals;
    // 存储经过转换后的体素的统计信息，其中win_size是窗口大小
    vector<VOX_FACTOR> sig_tran(win_size);
    // 表示要考虑的特定的特征值索引。在后续计算中使用。
    int kk = 0; // The kk-th lambda value
    int gps_size = plvec_voxels.size();

    // 遍历plvec_voxels中的每一个体素
    for (int a = 0; a < gps_size; a++)
    {
      // 获取该体素的原始统计数据sig_orig
      const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[a];
      VOX_FACTOR sig;
      // 使用传入的状态xs转换sig_orig，并将结果存储在sig_tran中。
      // 将所有转换后的统计数据sig_tran累加到sig中
      for (int i = 0; i < win_size; i++)
      {
        sig_tran[i].transform(sig_orig[i], xs[i]);
        sig += sig_tran[i];
      }

      // 计算均值向量vBar。
      Eigen::Vector3d vBar = sig.v / sig.N;
      // 计算协方差矩阵cmt
      Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();

      // 对协方差矩阵进行特征值分解，并获取其特征值lmbd
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();
      // 将最小特征值添加到residuals中
      residuals.push_back(lmbd[kk]);
    }

    return residuals;
  }

  void remove_residual(const vector<IMUST> &xs, double threshold, double reject_num)
  {
    vector<VOX_FACTOR> sig_tran(win_size);    // 存储经过变换后的体素统计信息
    int kk = 0; // The kk-th lambda value     // 表示要考虑的特征值的索引，这里是最小的特征值
    int rej_cnt = 0;                          // 已经删除的体素的计数器
    size_t i = 0;                             // 当前正在考虑的plvec_voxels中的体素的索引
    // 遍历体素
    for (; i < plvec_voxels.size();)
    {
      // 首先提取其原始统计信息（sig_orig）
      const vector<VOX_FACTOR> &sig_orig = *plvec_voxels[i];
      VOX_FACTOR sig;

      // 对于每一个状态xs[j]，计算变换后的体素统计信息并累加到sig中
      for (int j = 0; j < win_size; j++)
      {
        sig_tran[j].transform(sig_orig[j], xs[j]);
        sig += sig_tran[j];
      }

      // 计算体素的中心和协方差矩阵
      Eigen::Vector3d vBar = sig.v / sig.N;
      Eigen::Matrix3d cmt = sig.P / sig.N - vBar * vBar.transpose();
      // 使用Eigen::SelfAdjointEigenSolver来计算协方差矩阵的特征值。这些特征值被存储在lmbd中
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cmt);
      Eigen::Vector3d lmbd = saes.eigenvalues();

      // 如果最小的特征值（lmbd[kk]）大于或等于给定的阈值，则删除这个体素
      if (lmbd[kk] >= threshold)
      {
        plvec_voxels.erase(plvec_voxels.begin() + i);
        rej_cnt++;
        continue;
      }
      i++;
      // 如果已删除的体素数量rej_cnt达到了指定的reject_num，则终止循环
      if (rej_cnt == reject_num)
        break;
    }
  }
};

int BINGO_CNT = 0;
enum OCTO_STATE
{
  UNKNOWN,  // 未知
  MID_NODE, // 最小节点（体素内点数限制）
  PLANE     // 平面体素
};
/**
 * @brief 八叉树节点
 *
 */
class OCTO_TREE_NODE
{
public:
  OCTO_STATE octo_state;
  int layer, win_size;                   // layer 节点在八叉树中的深度，从根开始计数，win_size 窗口大小
  vector<PLV(3)> vec_orig, vec_tran;     // vec_orig 存储的是原始的点云数据，vec_tran 存储的是经过旋转和平移变换后的点云数据。下标都为帧数
  vector<VOX_FACTOR> sig_orig, sig_tran; // sig_orig 存储原始点云数据的统计信息，sig_tran 存储经过旋转和平移变换后的点云数据的统计信息。下标都为帧数

  OCTO_TREE_NODE *leaves[8];
  float voxel_center[3]; // 体素的中心位置
  float quater_length;   // 体素大小的四分之一，很可能是为了在八叉树中表示子体素的大小或者表示体素中心到边界的距离
  float eigen_thr;       // 平面特征值比最大阈值:最小和最大特征值的比率

  Eigen::Vector3d center, direct, value_vector;
  double eigen_ratio;

#ifdef ENABLE_RVIZ
  ros::NodeHandle nh;
  ros::Publisher pub_residual = nh.advertise<sensor_msgs::PointCloud2>("/residual", 1000);
  ros::Publisher pub_direct = nh.advertise<visualization_msgs::MarkerArray>("/direct", 1000);
#endif
  /**
   * @brief 八叉树节点构造函数
   *
   * @param _win_size
   * @param _eigen_thr
   */
  OCTO_TREE_NODE(int _win_size = WIN_SIZE, float _eigen_thr = 1.0 / 10) : win_size(_win_size), eigen_thr(_eigen_thr)
  {
    octo_state = UNKNOWN;
    layer = 0;
    vec_orig.resize(win_size);
    vec_tran.resize(win_size);
    sig_orig.resize(win_size);
    sig_tran.resize(win_size);
    // 初始化子节点为空
    for (int i = 0; i < 8; i++)
      leaves[i] = nullptr;
  }

  virtual ~OCTO_TREE_NODE()
  {
    for (int i = 0; i < 8; i++)
      if (leaves[i] != nullptr)
        delete leaves[i];
  }

  /**
   * @brief 判断点集是否构成平面特征
   *
   * @return true 是
   * @return false 不是平面
   */
  bool judge_eigen()
  {
    // 初始化协方差
    VOX_FACTOR covMat;
    // 遍历所有的窗口来累积协方差
    for (int i = 0; i < win_size; i++)
      if (sig_tran[i].N > 0)
        covMat += sig_tran[i];
    // 计算特征值和特征向量
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat.cov());
    value_vector = saes.eigenvalues();
    // 计算体素中的数据的中心
    center = covMat.v / covMat.N;
    // 获取最小的特征向量（对应于数据分布的最小变化方向）
    direct = saes.eigenvectors().col(0);
    // 计算最小和最大特征值的比率来判断数据分布是否近似于一个平面
    eigen_ratio = saes.eigenvalues()[0] / saes.eigenvalues()[2]; // [0] is the smallest
    // 如果比率大于给定的阈值，则数据不构成平面，返回false
    if (eigen_ratio > eigen_thr)
      // 不是平面特征
      return 0;
    // 进一步计算和判断
    // 基于体素中的数据的中心和最小特征值进一步计算
    double eva0 = saes.eigenvalues()[0];
    double sqr_eva0 = sqrt(eva0);
    Eigen::Vector3d center_turb = center + 5 * sqr_eva0 * direct;
    vector<VOX_FACTOR> covMats(8);
    // 对于窗口内的每一个点，判断它在体素中的位置，并计算子体素的协方差
    for (int i = 0; i < win_size; i++)
    {
      for (Eigen::Vector3d ap : vec_tran[i])
      {
        int xyz[3] = {0, 0, 0};
        for (int k = 0; k < 3; k++)
          if (ap(k) > center_turb[k])
            xyz[k] = 1;

        Eigen::Vector3d pvec(ap(0), ap(1), ap(2));

        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        covMats[leafnum].push(pvec);
      }
    }

    // 为了判断子体素中的数据是否构成一个平面，设置阈值
    double ratios[2] = {1.0 / (3.0 * 3.0), 2.0 * 2.0};
    int num_all = 0, num_qua = 0;
    for (int i = 0; i < 8; i++)
      if (covMats[i].N > 10)
      {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMats[i].cov());
        double child_eva0 = (saes.eigenvalues()[0]);
        if (child_eva0 > ratios[0] * eva0 && child_eva0 < ratios[1] * eva0)
          num_qua++;
        num_all++;
      }

    // 计算满足条件的子体素的比例
    double prop = 1.0 * num_qua / num_all;

    // 如果这个比例小于0.5，则数据不构成平面，返回false，否则返回true
    if (prop < 0.5)
      return 0;
    return 1;
  }

  /**
   * @brief 函数是在 OCTO_TREE_NODE 类中的，它的目的是根据当前体素内的点云数据进行细分，然后将这些数据分配到合适的子体素（子节点）中
   *
   * @param ci 当前窗口的点云索引
   */
  void cut_func(int ci)
  {
    // 获取指定帧 ci 的原始和转换后的点云数据
    PLV(3) &pvec_orig = vec_orig[ci];
    PLV(3) &pvec_tran = vec_tran[ci];

    // 遍历指定帧的所有点
    uint a_size = pvec_tran.size();
    for (uint j = 0; j < a_size; j++)
    {
      // 使用三维数组 xyz，函数确定每个点相对于当前体素中心的位置
      int xyz[3] = {0, 0, 0};
      // 如果点的某一坐标大于体素中心的对应坐标，那么该坐标的值设置为1，否则保持为0
      for (uint k = 0; k < 3; k++)
        if (pvec_tran[j][k] > voxel_center[k])
          xyz[k] = 1;
      // 计算出点所在的子体素编号。这实际上是一个简单的三维到一维的映射方法，用于确定点应该放入哪个子节点。
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
      // 如果对应的子节点（子体素）是空的（即 leaves[leafnum] == nullptr），函数将为其分配内存并初始化它
      if (leaves[leafnum] == nullptr)
      {
        // 每个体素可以进一步细分为八个等大小的子体素。当你有一个立方体并想要将其细分时，可以想象将其沿每个轴（X、Y和Z轴）的中心分割成两部分
        // 子节点的中心坐标是基于当前体素的中心坐标和 quater_length（四分之一的体素大小）计算得出的
        leaves[leafnum] = new OCTO_TREE_NODE(win_size, eigen_thr);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
        leaves[leafnum]->quater_length = quater_length / 2.0;
        leaves[leafnum]->layer = layer + 1;
      }
      // 函数将当前遍历到的点添加到对应的子节点的原始和转换后的点云数据中
      leaves[leafnum]->vec_orig[ci].push_back(pvec_orig[j]);
      leaves[leafnum]->vec_tran[ci].push_back(pvec_tran[j]);

      leaves[leafnum]->sig_orig[ci].push(pvec_orig[j]);
      leaves[leafnum]->sig_tran[ci].push(pvec_tran[j]);
    }

    // 最后，函数使用 swap() 方法清空当前体素的原始和转换后的点云数据，释放内存
    PLV(3)
    ().swap(pvec_orig);
    PLV(3)
    ().swap(pvec_tran);
  }

  // 体素再细分
  void recut()
  {
    // 首先检查体素的状态是否为 UNKNOWN
    if (octo_state == UNKNOWN)
    {
      // 如果是，则进入再细分的逻辑
      int point_size = 0;
      for (int i = 0; i < win_size; i++)
        point_size += sig_orig[i].N;

      // 函数计算体素内的总点数。如果点数小于某个阈值 MIN_PT，则将此体素标记为 MID_NODE，表示它是一个最小节点，不需要进一步细分
      if (point_size < MIN_PT)
      {
        octo_state = MID_NODE;
        vector<PLV(3)>().swap(vec_orig);
        vector<PLV(3)>().swap(vec_tran);
        vector<VOX_FACTOR>().swap(sig_orig);
        vector<VOX_FACTOR>().swap(sig_tran);
        // 清除与该体素相关的所有数据并返回
        return;
      }

      // 使用 judge_eigen() 函数来判断体素是否包含平面特征
      if (judge_eigen())
      {
        // 如果是，则将此体素标记为 PLANE，并清除与该体素相关的所有数据。
        // 这意味着此体素已被识别为一个平面特征，不需要进一步的细分。
        octo_state = PLANE;
#ifndef ENABLE_FILTER
#ifndef ENABLE_RVIZ
        vector<PLV(3)>().swap(vec_orig);
        vector<PLV(3)>().swap(vec_tran);
#endif
#endif
        return;
      }
      else
      {
        // 如果不是平面
        // 如果当前体素的层数达到了设定的最大层数，则将当前体素设置为最小节点
        if (layer == layer_limit)
        {
          // 着体素已经在树的最大深度上，所以不需要进一步的细分
          octo_state = MID_NODE;
          vector<PLV(3)>().swap(vec_orig);
          vector<PLV(3)>().swap(vec_tran);
          vector<VOX_FACTOR>().swap(sig_orig);
          vector<VOX_FACTOR>().swap(sig_tran);
          return;
        }
        // 如果体素既不是平面特征也没有达到最大层级，函数将对当前体素进行再细分
        // 首先清除与该体素相关的向量数据
        vector<VOX_FACTOR>().swap(sig_orig);
        vector<VOX_FACTOR>().swap(sig_tran);
        // 对然后调用 cut_func(i) 对每一帧的数据进行再细分
        for (int i = 0; i < win_size; i++)
          cut_func(i);
      }
    }
    // 函数遍历当前体素的所有子节点（在 leaves 数组中）。如果某个子节点不为空，它将对该子节点调用 recut() 函数
    // 这样整个过程就递归地进行了。
    for (int i = 0; i < 8; i++)
      if (leaves[i] != nullptr)
        leaves[i]->recut();
    // 当前体素的状态不是 UNKNOWN。这意味着该体素已经被分类为 PLANE、MID_NODE ，不需要进一步细分，此时所有的子体素为nullptr就退出递归
  }

  /**
   * @brief 获取所有的平面体素
   *
   * @param[out] vox_opt 所有的平面体素
   */
  void tras_opt(VOX_HESS &vox_opt)
  {
    // 判断是否为平面体素
    if (octo_state == PLANE)
      // 如果当前体素是平面体素，那么将其原始点云和点云统计信息插入到VOX_HESS对象中
      vox_opt.push_voxel(&sig_orig, &vec_orig);
    else
      // 如果不是平面体素，遍历子节点递归到体素为PLANE或者子体素全为nullptr停止
      for (int i = 0; i < 8; i++)
        if (leaves[i] != nullptr)
          leaves[i]->tras_opt(vox_opt);
  }

  void tras_display(int layer = 0)
  {
    float ref = 255.0 * rand() / (RAND_MAX + 1.0f);
    pcl::PointXYZINormal ap;
    ap.intensity = ref;

    if (octo_state == PLANE)
    {
      // std::vector<unsigned int> colors;
      // colors.push_back(static_cast<unsigned int>(rand() % 256));
      // colors.push_back(static_cast<unsigned int>(rand() % 256));
      // colors.push_back(static_cast<unsigned int>(rand() % 256));
      pcl::PointCloud<pcl::PointXYZINormal> color_cloud;

      for (int i = 0; i < win_size; i++)
      {
        for (size_t j = 0; j < vec_tran[i].size(); j++)
        {
          Eigen::Vector3d &pvec = vec_tran[i][j];
          ap.x = pvec.x();
          ap.y = pvec.y();
          ap.z = pvec.z();
          // ap.b = colors[0];
          // ap.g = colors[1];
          // ap.r = colors[2];
          ap.normal_x = sqrt(value_vector[1] / value_vector[0]);
          ap.normal_y = sqrt(value_vector[2] / value_vector[0]);
          ap.normal_z = sqrt(value_vector[0]);
          // ap.curvature = total;
          color_cloud.push_back(ap);
        }
      }

#ifdef ENABLE_RVIZ
      sensor_msgs::PointCloud2 dbg_msg;
      pcl::toROSMsg(color_cloud, dbg_msg);
      dbg_msg.header.frame_id = "camera_init";
      pub_residual.publish(dbg_msg);

      visualization_msgs::Marker marker;
      visualization_msgs::MarkerArray marker_array;
      marker.header.frame_id = "camera_init";
      marker.header.stamp = ros::Time::now();
      marker.ns = "basic_shapes";
      marker.id = BINGO_CNT;
      BINGO_CNT++;
      marker.action = visualization_msgs::Marker::ADD;
      marker.type = visualization_msgs::Marker::ARROW;
      marker.color.a = 1;
      marker.color.r = layer == 0 ? 1 : 0;
      marker.color.g = layer == 1 ? 1 : 0;
      marker.color.b = layer == 2 ? 1 : 0;
      marker.scale.x = 0.01;
      marker.scale.y = 0.05;
      marker.scale.z = 0.05;
      marker.lifetime = ros::Duration();
      geometry_msgs::Point apoint;
      apoint.x = center(0);
      apoint.y = center(1);
      apoint.z = center(2);
      marker.points.push_back(apoint);
      apoint.x += 0.2 * direct(0);
      apoint.y += 0.2 * direct(1);
      apoint.z += 0.2 * direct(2);
      marker.points.push_back(apoint);
      marker_array.markers.push_back(marker);
      pub_direct.publish(marker_array);
#endif
    }
    else
    {
      if (layer == layer_limit)
        return;
      layer++;
      for (int i = 0; i < 8; i++)
        if (leaves[i] != nullptr)
          leaves[i]->tras_display(layer);
    }
  }
};

// 八叉树根结点，每个节点,存储了某一个体素位置对应的所有帧的点云的信息,以及体素在细分后的所有子体素的点云信息.
class OCTO_TREE_ROOT : public OCTO_TREE_NODE
{
public:
  OCTO_TREE_ROOT(int _winsize, float _eigen_thr) : OCTO_TREE_NODE(_winsize, _eigen_thr) {}
};

// 体素优化器
class VOX_OPTIMIZER
{
public:
  int win_size, jac_leng, imu_leng;
  VOX_OPTIMIZER(int _win_size = WIN_SIZE) : win_size(_win_size)
  {
    jac_leng = DVEL * win_size;
    imu_leng = DIM * win_size;
  }

  /**
   * @brief 函数的目的是并行地计算与体素相关的Hessian矩阵、雅可比向量和残差
   * 
   * @param x_stats 包含体素位姿等状态信息的向量
   * @param voxhess 包含了与体素相关的所有信息和方法
   * @param x_ab 是一个表示状态变化或相对位姿的向量
   * @param Hess 输出的Hessian矩阵
   * @param JacT 输出的雅可比向量
   * @return double 
   */
  double divide_thread(vector<IMUST> &x_stats, VOX_HESS &voxhess, vector<IMUST> &x_ab,
                       Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    // 将Hessian矩阵,雅可比向量,残差设置为零
    double residual = 0;
    Hess.setZero();
    JacT.setZero();
    // 根据thd_num（可能是全局定义的线程数）初始化hessians和jacobins向量来存储每个线程的局部Hessian和雅可比值。
    PLM(-1) hessians(thd_num);
    PLV(-1) jacobins(thd_num);

    // 计算每个线程应处理的体素数量。这是通过将体素总数g_size除以线程数tthd_num来完成的
    for (int i = 0; i < thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    vector<double> resis(tthd_num, 0);
    // 如果体素的数量小于线程的数量，那么只使用一个线程
    int g_size = voxhess.plvec_voxels.size();
    if (g_size < tthd_num)
      tthd_num = 1;

    // 为每个线程创建一个新的线程对象并开始执行
    vector<thread *> mthreads(tthd_num);
    double part = 1.0 * g_size / tthd_num;
    // 每个线程调用VOX_HESS::acc_evaluate2函数来计算其分配的体素的Hessian、雅可比和残差。这是通过给每个线程分配一个体素范围来实现的
    for (int i = 0; i < tthd_num; i++)
      mthreads[i] = new thread(&VOX_HESS::acc_evaluate2, &voxhess, x_stats, part * i, part * (i + 1),
                               ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    // 将每个线程计算的Hessian和雅可比加到总的Hessian矩阵和雅可比向量中
    for (int i = 0; i < tthd_num; i++)
    {
      mthreads[i]->join();
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }
// 如果定义了AVG_THR，则返回平均残差（即总残差除以体素数g_size）
// 否则，返回总残差
#ifdef AVG_THR
    return residual / g_size;
#else
    return residual;
#endif
  }

  double only_residual(vector<IMUST> &x_stats, VOX_HESS &voxhess, vector<IMUST> &x_ab, bool is_avg = false)
  {
    double residual2 = 0;
    voxhess.evaluate_only_residual(x_stats, residual2);
    if (is_avg)
      return residual2 / voxhess.plvec_voxels.size();
    return residual2;
  }

  /**
   * @brief 去除平面体素的
   *
   * @param x_stats 窗口内的状态
   * @param voxhess
   * @param ratio
   */
  void remove_outlier(vector<IMUST> &x_stats, VOX_HESS &voxhess, double ratio)
  {
    // 1. 计算所有体素的残差值
    std::vector<double> residuals = voxhess.evaluate_residual(x_stats);
    // 2. 对残差值进行排序（升序）
    std::sort(residuals.begin(), residuals.end()); // sort in ascending order
    // 3. 根据给定的比率(ratio)，确定残差值的阈值
    // 例如，如果ratio是0.1（或10%），并且我们有1000个体素，那么reject_num将是100，意味着我们打算删除残差最大的100个体素
    int reject_num = std::floor(ratio * voxhess.plvec_voxels.size());
    // 计算阈值的方法是查找排序后的residuals向量中的特定索引位置的值。这个位置是由std::floor((1 - ratio) * voxhess.plvec_voxels.size()) - 1确定的。
    // 继续上面的例子，这将查找在位置900-1=899的值（因为是0-based索引）。因此，该值将是第900大的残差，意味着有100个体素的残差大于此值
    double threshold = residuals[std::floor((1 - ratio) * voxhess.plvec_voxels.size()) - 1];
    // std::cout << "vox_num before " << voxhess.plvec_voxels.size();
    // std::cout << ", reject threshold " << std::setprecision(3) << threshold << ", rejected " << reject_num;
    // 4. 删除超过阈值的体素（即异常体素）
    voxhess.remove_residual(x_stats, threshold, reject_num);
    // std::cout << ", vox_num after " << voxhess.plvec_voxels.size() << std::endl;
  }

  // 执行阻尼迭代优化过程，LM优化
  void damping_iter(vector<IMUST> &x_stats, VOX_HESS &voxhess, double &residual,
                    PLV(6) & hess_vec, size_t &mem_cost)
  {
    // 初始化参数
    double u = 0.01, v = 2;       // u和v是LM算法的参数，其中u是阻尼系数，v是更新因子
    // D，Hess，HessuD，JacT，dxi和new_dxi都是用于存储和计算LM迭代中的中间变量的矩阵和向量
    Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng),
        HessuD(jac_leng, jac_leng);
    Eigen::VectorXd JacT(jac_leng), dxi(jac_leng), new_dxi(jac_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp;
    // 使用传入的状态x_stats来计算相对位姿并存储在x_ab中
    vector<IMUST> x_ab(win_size);
    x_ab[0] = x_stats[0];
    for (int i = 1; i < win_size; i++)
    {
      x_ab[i].p = x_stats[i - 1].R.transpose() * (x_stats[i].p - x_stats[i - 1].p);
      x_ab[i].R = x_stats[i - 1].R.transpose() * x_stats[i].R;
    }

    // 迭代过程，计算Hessian矩阵和雅可比，记录计算时间
    double hesstime = 0;
    double solvtime = 0;
    size_t max_mem = 0;
    double loop_num = 0;
    for (int i = 0; i < 10; i++)
    {
      // 如果is_calc_hess为true
      if (is_calc_hess)
      {
        // 则调用divide_thread函数来计算Hessian矩阵和雅可比矩阵,以及残差
        double tm = ros::Time::now().toSec();
        residual1 = divide_thread(x_stats, voxhess, x_ab, Hess, JacT);
        hesstime += ros::Time::now().toSec() - tm;
      }

      double tm = ros::Time::now().toSec();
      // 使用当前的u值更新D的对角线，并计算HessuD
      D.diagonal() = Hess.diagonal();
      HessuD = Hess + u * D;
      double t1 = ros::Time::now().toSec();
      // 将HessuD矩阵转换为稀疏矩阵，并使用稀疏线性求解器求解线性方程组来获取增量dxi
      Eigen::SparseMatrix<double> A1_sparse(jac_leng, jac_leng);
      std::vector<Eigen::Triplet<double>> tripletlist;
      for (int a = 0; a < jac_leng; a++)
        for (int b = 0; b < jac_leng; b++)
          if (HessuD(a, b) != 0)
          {
            tripletlist.push_back(Eigen::Triplet<double>(a, b, HessuD(a, b)));
            // A1_sparse.insert(a, b) = HessuD(a, b);
          }
      A1_sparse.setFromTriplets(tripletlist.begin(), tripletlist.end());
      A1_sparse.makeCompressed();
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> Solver_sparse;
      Solver_sparse.compute(A1_sparse);
      size_t temp_mem = check_mem();
      if (temp_mem > max_mem)
        max_mem = temp_mem;
      dxi = Solver_sparse.solve(-JacT);
      temp_mem = check_mem();
      if (temp_mem > max_mem)
        max_mem = temp_mem;
      solvtime += ros::Time::now().toSec() - tm;
      // new_dxi = Solver_sparse.solve(-JacT);
      // printf("new solve time cost %f\n",ros::Time::now().toSec() - t1);
      // relative_err = ((Hess + u*D)*dxi + JacT).norm()/JacT.norm();
      // absolute_err = ((Hess + u*D)*dxi + JacT).norm();
      // std::cout<<"relative error "<<relative_err<<std::endl;
      // std::cout<<"absolute error "<<absolute_err<<std::endl;
      // std::cout<<"delta x\n"<<(new_dxi-dxi).transpose()/dxi.norm()<<std::endl;

      // 使用dxi更新x_stats以得到新的状态x_stats_temp
      x_stats_temp = x_stats;
      for (int j = 0; j < win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DVEL * j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DVEL * j + 3, 0);
      }

      double q1 = 0.5 * dxi.dot(u * D * dxi - JacT);
#ifdef AVG_THR
      // 计算新的残差residual2
      residual2 = only_residual(x_stats_temp, voxhess, x_ab, true);
      q1 /= voxhess.plvec_voxels.size();
#else
      residual2 = only_residual(x_stats_temp, voxhess, x_ab);
#endif
      residual = residual2;
      q = (residual1 - residual2);
      // printf("iter%d: (%lf %lf) u: %lf v: %lf q: %lf %lf %lf\n",
      //        i, residual1, residual2, u, v, q/q1, q1, q);
      loop_num = i + 1;
      // if(hesstime/loop_num > 1) printf("Avg. Hessian time: %lf ", hesstime/loop_num);
      // if(solvtime/loop_num > 1) printf("Avg. solve time: %lf\n", solvtime/loop_num);
      // if(double(max_mem/1048576.0) > 2.0) printf("Max mem: %lf\n", double(max_mem/1048576.0));
      // 使用当前和新的残差来更新u和v
      if (q > 0)
      {
        x_stats = x_stats_temp;
        q = q / q1;
        v = 2;
        q = 1 - pow(2 * q - 1, 3);
        u *= (q < one_three ? one_three : q);
        is_calc_hess = true;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;
      }
#ifdef AVG_THR
      // 根据残差的变化确定下一次迭代是否需要重新计算Hessian
      // 如果残差的变化小于一个给定的阈值或迭代次数达到10次，则终止迭代
      if ((fabs(residual1 - residual2) / residual1) < 0.05 || i == 9)
      {
        if (mem_cost < max_mem)
          mem_cost = max_mem;
        for (int j = 0; j < win_size - 1; j++)
          for (int k = j + 1; k < win_size; k++)
            hess_vec.push_back(Hess.block<DVEL, DVEL>(DVEL * j, DVEL * k).diagonal().segment<DVEL>(0));
        break;
      }
#else
      if (fabs(residual1 - residual2) < 1e-9)
        break;
#endif
    }
  }

  size_t check_mem()
  {
    FILE *file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != nullptr)
    {
      if (strncmp(line, "VmRSS:", 6) == 0)
      {
        int len = strlen(line);

        const char *p = line;
        for (; std::isdigit(*p) == false; ++p)
        {
        }

        line[len - 3] = 0;
        result = atoi(p);

        break;
      }
    }
    fclose(file);

    return result;
  }
};

#endif
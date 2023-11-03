#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <Eigen/Core>
#include <unordered_map>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <math.h>

#define HASH_P 116101
#define MAX_N 10000000019
#define SMALL_EPS 1e-10
#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0
#define PLM(a) vector<Eigen::Matrix<double, a, a>, Eigen::aligned_allocator<Eigen::Matrix<double, a, a>>>
// 大小为 a×1 的 Eigen 列向量的数组
#define PLV(a) vector<Eigen::Matrix<double, a, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, a, 1>>>
#define VEC(a) Eigen::Matrix<double, a, 1>

#define G_m_s2 9.81
#define DIMU 18
#define DIM 15
#define DNOI 12
#define NMATCH 5
#define DVEL 6

typedef pcl::PointXYZ PointType;
// typedef pcl::PointXYZI PointType;
// typedef pcl::PointXYZINormal PointType;
using namespace std;

Eigen::Matrix3d I33(Eigen::Matrix3d::Identity());
Eigen::Matrix<double, DIMU, DIMU> I_imu(Eigen::Matrix<double, DIMU, DIMU>::Identity());

//体素的位置
class VOXEL_LOC
{
public:
  int64_t x, y, z; //巍峨uzgu

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOC &other) const
  {
    return (x == other.x && y == other.y && z == other.z);
  }
};

namespace std
{
  template <>
  struct hash<VOXEL_LOC>
  {
    size_t operator()(const VOXEL_LOC &s) const
    {
      using std::hash;
      using std::size_t;
      // return (((hash<int64_t>()(s.z)*HASH_P)%MAX_N + hash<int64_t>()(s.y))*HASH_P)%MAX_N + hash<int64_t>()(s.x);
      long long index_x, index_y, index_z;
      double cub_len = 0.125;
      index_x = int(round(floor((s.x) / cub_len + SMALL_EPS)));
      index_y = int(round(floor((s.y) / cub_len + SMALL_EPS)));
      index_z = int(round(floor((s.z) / cub_len + SMALL_EPS)));
      return (((((index_z * HASH_P) % MAX_N + index_y) * HASH_P) % MAX_N) + index_x) % MAX_N;
    }
  };
}

double matrixAbsSum(Eigen::MatrixXd mat)
{
  double sum = 0.0;
  for (int i = 0; i < mat.rows(); i++)
    for (int j = 0; j < mat.cols(); j++)
      sum += fabs(mat(i, j));
  return sum;
}

double sigmoid_w(double r)
{
  return 1.0 / (1 + exp(-r));
}

Eigen::Matrix3d Exp(const Eigen::Vector3d &ang)
{
  double ang_norm = ang.norm();
  Eigen::Matrix3d Eye3 = Eigen::Matrix3d::Identity();
  if (ang_norm > 0.0000001)
  {
    Eigen::Vector3d r_axis = ang / ang_norm;
    Eigen::Matrix3d K;
    K << SKEW_SYM_MATRX(r_axis);
    /// Roderigous Tranformation
    return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
  }
  else
  {
    return Eye3;
  }
}

Eigen::Matrix3d Exp(const Eigen::Vector3d &ang_vel, const double &dt)
{
  double ang_vel_norm = ang_vel.norm();
  Eigen::Matrix3d Eye3 = Eigen::Matrix3d::Identity();

  if (ang_vel_norm > 0.0000001)
  {
    Eigen::Vector3d r_axis = ang_vel / ang_vel_norm;
    Eigen::Matrix3d K;

    K << SKEW_SYM_MATRX(r_axis);
    double r_ang = ang_vel_norm * dt;

    /// Roderigous Tranformation
    return Eye3 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
  }
  else
  {
    return Eye3;
  }
}

Eigen::Vector3d Log(const Eigen::Matrix3d &R)
{
  double theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
  Eigen::Vector3d K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

Eigen::Matrix3d hat(const Eigen::Vector3d &v)
{
  Eigen::Matrix3d Omega;
  Omega << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  return Omega;
}

Eigen::Matrix3d jr(Eigen::Vector3d vec)
{
  double ang = vec.norm();

  if (ang < 1e-9)
  {
    return I33;
  }
  else
  {
    vec /= ang;
    double ra = sin(ang) / ang;
    return ra * I33 + (1 - ra) * vec * vec.transpose() - (1 - cos(ang)) / ang * hat(vec);
  }
}

Eigen::Matrix3d jr_inv(const Eigen::Matrix3d &rotR)
{
  Eigen::AngleAxisd rot_vec(rotR);
  Eigen::Vector3d axi = rot_vec.axis();
  double ang = rot_vec.angle();

  if (ang < 1e-9)
  {
    return I33;
  }
  else
  {
    double ctt = ang / 2 / tan(ang / 2);
    return ctt * I33 + (1 - ctt) * axi * axi.transpose() + ang / 2 * hat(axi);
  }
}

/**
 * @brief IMU的状态
 * 
 */
struct IMUST
{
  double t;                 // 可能表示IMU的时间戳
  Eigen::Matrix3d R;        // 3x3的旋转矩阵，表示IMU的旋转状态。
  Eigen::Vector3d p;        // 3D位置向量，表示IMU的位置
  Eigen::Vector3d v;        // 3D速度向量，表示IMU的速度
  Eigen::Vector3d bg;       // 陀螺仪的偏置
  Eigen::Vector3d ba;       // 加速度计的偏置
  Eigen::Vector3d g;        // 重力向量，其中 G_m_s2 是重力加速度的常数

  // 默认构造函数，用于初始化所有成员变量
  IMUST()
  {
    setZero();
  }

  // 带参数的构造函数，用于直接初始化成员变量
  IMUST(double _t, const Eigen::Matrix3d &_R, const Eigen::Vector3d &_p, const Eigen::Vector3d &_v,
        const Eigen::Vector3d &_bg, const Eigen::Vector3d &_ba,
        const Eigen::Vector3d &_g = Eigen::Vector3d(0, 0, -G_m_s2)) : t(_t), R(_R), p(_p), v(_v), bg(_bg), ba(_ba), g(_g) {}

  // 重载的加法赋值操作符，用于更新当前IMU状态
  IMUST &operator+=(const Eigen::Matrix<double, DIMU, 1> &ist)
  {
    this->R = this->R * Exp(ist.block<3, 1>(0, 0));
    this->p += ist.block<3, 1>(3, 0);
    this->v += ist.block<3, 1>(6, 0);
    this->bg += ist.block<3, 1>(9, 0);
    this->ba += ist.block<3, 1>(12, 0);
    this->g += ist.block<3, 1>(15, 0);
    return *this;
  }

  // 重载的减法操作符，用于计算两个IMU状态之间的差异
  Eigen::Matrix<double, DIMU, 1> operator-(const IMUST &b)
  {
    Eigen::Matrix<double, DIMU, 1> a;
    a.block<3, 1>(0, 0) = Log(b.R.transpose() * this->R);
    a.block<3, 1>(3, 0) = this->p - b.p;
    a.block<3, 1>(6, 0) = this->v - b.v;
    a.block<3, 1>(9, 0) = this->bg - b.bg;
    a.block<3, 1>(12, 0) = this->ba - b.ba;
    a.block<3, 1>(15, 0) = this->g - b.g;
    return a;
  }

  // 重载的赋值操作符
  IMUST &operator=(const IMUST &b)
  {
    this->R = b.R;
    this->p = b.p;
    this->v = b.v;
    this->bg = b.bg;
    this->ba = b.ba;
    this->g = b.g;
    this->t = b.t;
    return *this;
  }

  // 将所有成员变量设置为其初始值
  void setZero()
  {
    t = 0;
    R.setIdentity();
    p.setZero();
    v.setZero();
    bg.setZero();
    ba.setZero();
    g << 0, 0, -G_m_s2;
  }
};

void assign_qt(Eigen::Quaterniond &q, Eigen::Vector3d &t,
               const Eigen::Quaterniond &q_, const Eigen::Vector3d &t_)
{
  q.w() = q_.w();
  q.x() = q_.x();
  q.y() = q_.y();
  q.z() = q_.z();
  t(0) = t_(0);
  t(1) = t_(1);
  t(2) = t_(2);
}

// 点结构
struct M_POINT
{
  float xyz[3];
  int count = 0;
};

/**
 * @brief 对点云数据进行体素下采样
 * 
 * @param pc 输入和输出的点云数据。
 * @param voxel_size 体素的大小
 */
void downsample_voxel(pcl::PointCloud<PointType> &pc, double voxel_size)
{
  // 如果体素大小过小，直接返回，不进行下采样
  if (voxel_size < 0.01)
    return;

  // 使用unordered_map存储每个体素中的点
  std::unordered_map<VOXEL_LOC, M_POINT> feature_map; //体素地图
  size_t pt_size = pc.size();
  // 遍历点云中的每一个点
  for (size_t i = 0; i < pt_size; i++)
  {
    PointType &pt_trans = pc[i];
    float loc_xyz[3];
    // 计算点在每个维度上的体素位置
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pt_trans.data[j] / voxel_size;
      if (loc_xyz[j] < 0)
        loc_xyz[j] -= 1.0;
    }

    // 使用VOXEL_LOC类型存储体素位置，并检查该位置是否已经在feature_map中
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feature_map.find(position);
    // 如果该体素位置已经在feature_map中，累加该位置的坐标并增加点的计数
    if (iter != feature_map.end())
    {
      iter->second.xyz[0] += pt_trans.x;
      iter->second.xyz[1] += pt_trans.y;
      iter->second.xyz[2] += pt_trans.z;
      iter->second.count++;
    }
    else
    {
      // 如果该体素位置不在feature_map中，添加新的位置并初始化点的坐标和计数
      M_POINT anp;
      anp.xyz[0] = pt_trans.x;
      anp.xyz[1] = pt_trans.y;
      anp.xyz[2] = pt_trans.z;
      anp.count = 1;
      feature_map[position] = anp;
    }
  }

  // 从feature_map中提取下采样后的点，计算每个体素的平均位置作为新的点云位置
  pt_size = feature_map.size();
  pc.clear();
  pc.resize(pt_size);
  size_t i = 0;
  for (auto iter = feature_map.begin(); iter != feature_map.end(); ++iter)
  {
    pc[i].x = iter->second.xyz[0] / iter->second.count;
    pc[i].y = iter->second.xyz[1] / iter->second.count;
    pc[i].z = iter->second.xyz[2] / iter->second.count;
    i++;
  }
}

void pl_transform(pcl::PointCloud<PointType> &pl1, const Eigen::Matrix3d &rr, const Eigen::Vector3d &tt)
{
  for (PointType &ap : pl1.points)
  {
    Eigen::Vector3d pvec(ap.x, ap.y, ap.z);
    pvec = rr * pvec + tt;
    ap.x = pvec[0];
    ap.y = pvec[1];
    ap.z = pvec[2];
  }
}

void plvec_trans(PLV(3) & porig, PLV(3) & ptran, IMUST &stat)
{
  uint asize = porig.size();
  ptran.resize(asize);
  for (uint i = 0; i < asize; i++)
    ptran[i] = stat.R * porig[i] + stat.p;
}

// bool time_compare(PointType &x, PointType &y) {return (x.curvature < y.curvature);}

// 主要用于表示体素中点集的统计信息
class VOX_FACTOR
{
public:
  Eigen::Matrix3d P;    // 累积了体素中点与其转置的乘积
  Eigen::Vector3d v;    // 向量累积了体素中的点
  int N;                // 计数器，记录已添加到该体素中的点的数量

  // 构造函数，用于初始化 P, v, 和 N
  VOX_FACTOR()
  {
    P.setZero();
    v.setZero();
    N = 0;
  }

  // 重置 P, v, 和 N 为其初始状态
  void clear()
  {
    P.setZero();
    v.setZero();
    N = 0;
  }

  // 向体素中添加一个点，并更新 P, v, 和 N
  void push(const Eigen::Vector3d &vec)
  {
    N++;
    P += vec * vec.transpose();
    v += vec;
  }

  // 计算并返回体素中点集的协方差矩阵
  Eigen::Matrix3d cov()
  {
    Eigen::Vector3d center = v / N;
    return P / N - center * center.transpose();
  }
  // 重载的加法赋值操作符，用于将两个 VOX_FACTOR 对象相加并更新当前对象
  VOX_FACTOR &operator+=(const VOX_FACTOR &sigv)
  {
    this->P += sigv.P;
    this->v += sigv.v;
    this->N += sigv.N;

    return *this;
  }

  // 使用给定的状态 stat 转换 VOX_FACTOR
  void transform(const VOX_FACTOR &sigv, const IMUST &stat)
  {
    N = sigv.N;
    v = stat.R * sigv.v + N * stat.p;
    Eigen::Matrix3d rp = stat.R * sigv.v * stat.p.transpose();
    P = stat.R * sigv.P * stat.R.transpose() + rp + rp.transpose() + N * stat.p * stat.p.transpose();
  }
};

const double threshold = 0.1;
bool esti_plane(Eigen::Vector4d &pca_result, const pcl::PointCloud<PointType> &point)
{
  Eigen::Matrix<double, NMATCH, 3> A;
  Eigen::Matrix<double, NMATCH, 1> b;
  b.setOnes();
  b *= -1.0f;

  for (int j = 0; j < NMATCH; j++)
  {
    A(j, 0) = point[j].x;
    A(j, 1) = point[j].y;
    A(j, 2) = point[j].z;
  }

  Eigen::Vector3d normvec = A.colPivHouseholderQr().solve(b);

  for (int j = 0; j < NMATCH; j++)
  {
    if (fabs(normvec.dot(A.row(j)) + 1.0) > threshold)
      return false;
  }

  double n = normvec.norm();
  pca_result(0) = normvec(0) / n;
  pca_result(1) = normvec(1) / n;
  pca_result(2) = normvec(2) / n;
  pca_result(3) = 1.0 / n;
  return true;
}

#endif

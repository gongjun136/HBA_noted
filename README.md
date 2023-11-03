# HBA: A Globally Consistent and Efficient Large-Scale LiDAR Mapping Module

## **1. Introduction**
**HBA** is developed to resolve the issue that the divergence within the point cloud map cannot be fully eliminated after pose graph optimization (PGO). Though PGO is time-efficient, it does not directly optimize the mapping consistency. LiDAR bundle adjustment (BA) could mitigate this issue; however, it is too time-consuming on large-scale maps. HBA proposes a hierarchical structure to divide the huge LidAR BA problem into multiple smaller BA problems and use the PGO to smoothly update the entire LiDAR poses. HBA could achieve similar accuracy compared with the original BA method but with much less computation time.

<div align="center">
  <div align="center">
    <img src="figure/lidar_frame.png"  width="100%" />
  </div>
  <font color=#a0a0a0 size=2>Pyramid structure of our proposed Hierarchical Bundle Adjustment.</font>
</div>

## **2. Paper and Video**
Our paper has been publised on [IEEE RA-L](https://ieeexplore.ieee.org/abstract/document/10024300) and the corresponding video could be watched on [Bilibili](https://www.bilibili.com/video/BV1Qg41127j9/?spm_id_from=333.999.0.0) or [YouTuBe](https://youtu.be/CuLnTnXVujw). If you find our work useful in your research, please consider citing:

```
@ARTICLE{10024300,
  author={Liu, Xiyuan and Liu, Zheng and Kong, Fanze and Zhang, Fu},
  journal={IEEE Robotics and Automation Letters}, 
  title={Large-Scale LiDAR Consistent Mapping Using Hierarchical LiDAR Bundle Adjustment}, 
  year={2023},
  volume={8},
  number={3},
  pages={1523-1530},
  doi={10.1109/LRA.2023.3238902}}
```

## **3. Run the Code**
### 3.1 Prerequisites
Our code is tested under [Ubuntu 20.04](https://releases.ubuntu.com/focal/) with [ROS Noetic](https://wiki.ros.org/noetic/Installation/Ubuntu) and [Ubuntu 18.04](https://releases.ubuntu.com/18.04/) with [ROS Melodic](https://wiki.ros.org/melodic/Installation/Ubuntu), [PCL 1.10.0](https://github.com/PointCloudLibrary/pcl/releases), [Eigen 3.3.7](https://gitlab.com/libeigen/eigen/-/releases/3.3.7), and [GTSAM 4.1.1](https://github.com/borglab/gtsam).

### 3.2 文件结构
要测试您自己的数据，请准文如下的文件结构：包含点云的`pcd` 文件和每次LiDAR扫描的初始位姿的`pose.json`文件。注意位姿的格式是 `tx ty tz qw qx qy qz`.
```
.
├── pcd
│   ├── 0.pcd
│   └── 1.pcd
└── pose.json
```
### 3.3 重要参数
#### in `hba.launch` file
* `total_layer_num`: HBA中使用的层数，默认是`3`。
* `pcd_name_fill_num`: pcd文件名中前缀0的数量(例如，如果pcd文件以 `00000.pcd`开头, 请填写 `5`)。默认是 `0`。
* `thread_num`: CPU并行计算使用的线程数， 默认是 `16`。
#### in `hba.hpp` file
* `downsample_size`: 用于每次LiDAR扫描的点云下采样的叶子大小。默认是`0.1`。
* `voxel_size`: LiDAR BA中使用的初始体素大小。默认值`4.0`。
* `eigen_ratio`: 用于确定该体素是否包含有效平面特征的阈值。值越大意味着阈值越不严格。默认值为 `0.1`。
* `reject_ratio`: 用于拒绝优化中使用的最大数量的体素（残差）的阈值，默认值为 `0.05`。
#### in `ba.hpp` file
* `WIN_SIZE`: 局部BA使用的窗口大小，默认为 `10`。
* `GAP`: 每两个相邻窗口开始之间的步幅大小，默认为`5`。
* `layer_limit`: LiDAR BA 中体素重切次数的最大数量，默认为`2`。

备注：在`global_ba` 函数中，我们使用比local BA中稍微大的`eigen_ratio=0.2` 参数, 这通常会导致更快地收敛。但是，您可以在`hba.cpp` 文件中的`cut_voxel`函数中调整它们。较小的体素尺寸和特征比参数通常会导致较高的精度，但计算时间较长。 

### 3.4 Try Our Data
我们自己收集的[park](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/xliuaa_connect_hku_hk/EbIOZhucOxJMs_m5_4JheNUBIj1FILxNmgVX0vt4C7madw?e=LD1XQh) 和 [KITTI 07](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/xliuaa_connect_hku_hk/EdDPCY06Xa5CnmKG7_fbmq8BW9jpNZGRsVxtHLt80voKaw?e=nJWZjF)数据集的压缩pcd文件及其初始位姿上传到了OneDrive 。

备注：`hba.launch`仅优化LiDAR位姿，但不能可视化点云地图，`visualize.launch`用于查看点云地图。另一件事是，当您启动`hba.launch` 时，当启动完成时，它只会优化一次位姿。因此，如果您对结果不满意，可以再次执行启动。

## **4. Applications**

### 4.1 Globally Optimize the Point Cloud Mapping Consistency

#### 4.1.1 Close the Loop in Odometry (See Our Results Below on KITTI Sequences)

<div align="center"><img src="figure/kitti.gif"  width="100%" /></div>
<div align="center"><img src="figure/kitti05.jpg"  width="100%" /></div>

#### 4.1.2 Further Optimize the Mapping Consistency (See Our Results on Self-Collected Dataset)
<div align="center"><img src="figure/lg02.jpg"  width="100%" /></div>

### 4.2 Provide a Point Cloud Map with Centimeter-Level of Precision

<!-- #### 3.2.1 HILTI SLAM Challenge
HBA has participated in [HILTI SLAM Challenge](https://hilti-challenge.com/) and achieved centimeter level of precision (millimeter level in some sequences). Currently, HBA ranked #1 on leaderboard of the [HILTI SLAM Challenge 2022](https://hilti-challenge.com/leader-board-2022.html) and #2 on leaderboard of HILTI SLAM Challenge 2023.

<!-- #### 3.2.2 MARSIM -->
[MARSIM](https://github.com/hku-mars/MARSIM) is a lightweight point-realistic simulator for LiDAR-based UAVs, to which **HBA** has contributed more than ten real-world point cloud maps with centimeter-level of precision.

## **5. Acknowledgements**
In the development of **HBA**, we stand on the state-of-the-art work: [BALM2](https://github.com/hku-mars/BALM).

## **6. License**
The source code is released under GPLv2 license.

We are still working on improving the performance and reliability of our codes. For any technical issues, please contact us via email xliuaa@connect.hku.hk. For commercial use, please contact Dr. Fu Zhang fuzhang@hku.hk.
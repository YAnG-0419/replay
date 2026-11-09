# 双臂 EEF Replay（仅离线）

> **安全边界：本目录不包含真机执行功能。** 它不导入 ROS，不访问 FR3/Sharpa
> IP，不连接控制器，也不调用现有 MoveIt 服务。生成的 JSON 只是待审查的离线计划，
> `execution_enabled` 永远为 `false`。

本项目把头部相机坐标系 `camera_color_optical_frame` 下的左右手 EEF 绝对位姿，
分别转换到左、右 FR3 的 libfranka `O` 基座坐标系。手部使用 22 个关节名称做直接映射，
不会依赖容易错位的裸下标。

## 目录

- `config/handeye_calibration.json`：2026-10-30 标定交付原件的只读副本。
- `src/eef_replay/`：标定加载、位姿转换、手关节映射和离线计划生成。
- `examples/synthetic_camera_eef.json`：不接真机的合成输入例子。
- `experimental_pinocchio_pyroki/`：独立的 Pinocchio IK 与 PyRoki 平滑实验代码。
- `tests/`：只使用合成数据的单元测试。

## 输入约定

输入 JSON 必须声明：

- `reference_frame`: `camera_color_optical_frame`
- `translation_unit`: `metre`
- `pose_mode`: `absolute`
- `controlled_frame`: 必须是 `fr3_link8`
- 每个样本严格递增的 `t_s`
- EEF 位姿为 `position_m` + `quaternion_wxyz`，或 4x4 `matrix`
- 手部为完整的 22 关节名称到弧度值的映射

当前 replay 已按采集定义收紧为 `fr3_link8`。现有双 FR3 URDF 中左右
`fr3v2_link8` 都通过固定 `joint8` 连接到 `link7`，变换为 `xyz="0 0 0.107"`、
`rpy="0 0 0"`。代码不接受 link7、笼统的 flange 或 Sharpa TCP 输入，防止无意中
引入 107 mm 或工具偏置错误。

## 离线生成

episode 数据使用短入口即可：

```bash
./replay_episode 395                         # Pinocchio IK + PyRoki 平滑
./replay_episode 395 --start-from-current    # 再生成当前姿态到第一帧的离线过渡
./replay_episode 395 --ik moveit             # MoveIt/KDL + 到第一帧的 plan-only 规划
./replay_episode 395 --ik both               # 两套结果都生成
```

短入口把固定的数据根目录、标定文件、双 FR3 URDF 和 30 Hz 采样率收进了默认配置。
它会复用已经生成的中间文件；需要从 rosbag 重新提取时加 `--force`。

默认路径先运行 Pinocchio IK，再在本目录的独立 `.venv-pyroki` 环境中运行 PyRoki；
不会改动采集环境。环境已经装好，需重建时运行一次 `./setup_pyroki`。
`--start-from-current` 只订阅双臂 joint state，生成一个限速、限加速度的
minimum-jerk 过渡到第一帧。这个过渡没有碰撞检查，不能直接当作真机放行依据。

MoveIt 模式只订阅关节状态并调用 `/compute_ik`、`/plan_kinematic_path`，不会创建
控制器、轨迹 action 或 command topic。`moveit_start_to_frame0.json` 是从执行命令时
的双臂实测状态到 replay 第一帧的碰撞检查规划，但仍然只是 JSON，不会被提交执行。
运行该模式前必须由现场已有流程启动 MoveIt/KDL 服务。

在本目录执行：

```bash
PYTHONPATH=src python -m eef_replay.cli \
  examples/synthetic_camera_eef.json \
  /tmp/offline_replay_plan.json \
  --calibration config/handeye_calibration.json
```

输出的 `left.base_T_link8_target` 和 `right.base_T_link8_target` 分别属于两个独立基座，
不能把其中一个直接当作 `lychee_root`。之后可在独立实验目录中做每臂 IK 和关节平滑。

## 上真机前仍缺少的确认

1. 确认实际 IK URDF 的目标帧就是左右 `link8`，不能传 `link7`。
2. 确认单臂 URDF 的 base 与各自 libfranka `O` 完全一致。
3. 离线检查 IK 可达性、关节限位、速度/加速度、双臂及环境碰撞。
4. 做可视化和低速空载 dry-run，并由现场人员单独批准执行层。

10 月标定自身只承诺坐标变换，不代表绝对定位、TCP 或轨迹执行已经验收。

## 真机执行入口

`execute_episode` 使用 MoveIt 在 100% scaling 下从当前双臂状态规划到第一帧，随后
将 PyRoki 结果按 episode 原始时间戳同步提交给左右 `FollowJointTrajectory`，同时以
相同 30 Hz 时间轴发布左右 Sharpa 22 关节命令。它不做 replay 速度缩放。

先单独启动现有 Sharpa Wave 驱动（不启动 MANUS/Avatar 命令源）：

```bash
./start_sharpa_hands
```

这个入口读取项目现有的左右手序列号，将 Sharpa 速度系数设为固件允许的最大值
`0.5`，并关闭命令超时阈值。Replay 是 `/sharpa/{left,right}/command` 的唯一发布者。

```bash
./execute_episode 395 --check-live
./execute_episode 395 --execute
```

需要现有 `moveit-real` 与 `arm-ui` 已启动；不要启动 O30i 的 Docker `hand-control`。
Ctrl-C 会向左右臂同时请求取消；轨迹正常结束后保持最后一帧。结束本次 replay 会话后
可运行 `./stop_sharpa_hands` 关闭独立 Sharpa 驱动。


# IK 整理与扩展

## 稳定入口与可选入口

| 名称 | 入口 / 实现 | 用途 |
| --- | --- | --- |
| 原 Pinocchio | `experimental_pinocchio_pyroki/solve_bimanual_plan.py` / `pinocchio_ik.py` | 已有阻尼最小二乘 IK，上一帧作为 seed，仍是默认流程 |
| MoveIt / KDL | `experimental_pinocchio_pyroki/moveit_kdl_plan_only.py` | 已有服务接口，另用 OMPL RRTConnect 规划到首帧 |
| continuous | `eef_replay.ik.continuous.ContinuousIK` | 在目标完整位姿约束下，优先接近上一帧与参考构型，保持限位余量 |
| differential | `eef_replay.ik.differential.DifferentialIK` | 由当前 seed 反复求有界雅可比增量，适合小步目标跟踪 |
| placo | `eef_replay.ik.placo_backend.PlacoIK` | 可选 PlaCo 0.11.0 QP 适配器，逐帧使用相同 seed、边界和后验验证 |
| 局部可微分接口 | `eef_replay.ik.sensitivity.solution_jacobian` | 计算 continuous 解对目标局部扰动的梯度，供离线优化使用 |
| PyRoki | `experimental_pinocchio_pyroki/pyroki_smoothing.py` | 原流程中平滑已有 IK 轨迹，不重新搜索 IK 分支 |

原 Pinocchio 的两个文件与基线提交 `68d1076` 保持逐字节一致。新包独立实现
运动学封装，不修改/替换旧类；注册器惰性加载，缺少可选依赖不影响旧入口。
默认 `./replay_episode N` 及 `--ik pinocchio|moveit|both` 保留原流程和原输出路径。

## 可选环境

在单独的离线环境中安装，不需要改动正在使用的采集/执行环境：

```bash
python -m pip install -e '.[ik]'
prepare-ik-replay --help
# 直接从源码运行也可以：
PYTHONPATH=src python -m eef_replay.ik.cli --help
```

`ik` extra 包含 Pinocchio、SciPy 和 PyYAML。基础包仍只要求 NumPy。
Placo 环境和源码快照的使用方法见 [Placo 接入说明](placo.md)。

## 起始状态与 seed

每一帧的硬约束区域都以**上一帧已经求出的解**为中心。求首帧前可以显式提供
home 或采集得到的当前状态；未提供时，仅在输入首帧携带完整的
`source_arm_joint_positions_rad` 时使用它作初值，不补零、不随机初始化。
种子不代表机器人已经在该构型。

起始 JSON 的关节顺序必须与本体配置完全一致：

```json
{
  "execution_enabled": false,
  "robot_profile": "franka-sharpa",
  "state_kind": "named_home_target_not_live_measurement",
  "joint_names": ["left_fr3v2_joint1", "...填写完整14个关节名..."],
  "joint_positions_rad": [0.0, "...填写同序14个数值..."]
}
```

上面是结构说明，不能直接作为输入。已有的 Franka pose YAML 可通过下面的工具转换：

```bash
python tools/home_pose_to_start_state.py /path/poses.yaml output/chahua_state.json \
  --label chahua --model-joint-variant fr3v2 --angle-unit radian
```

若提供 `joint_limits_rad.lower/upper`，求解时与 URDF 限位取交集。

## 求解命令

直接使用独立入口，可以保留、对比多个实验结果：

```bash
python experimental_pinocchio_pyroki/solve_continuous_plan.py \
  output/base_link8_plan.json output/experiments/continuous/ik.json \
  --urdf /path/dual_fr3_kinematics.urdf --robot franka-sharpa \
  --backend continuous --start-state output/chahua_state.json

# 更换为微分 IK；其余输入和约束相同
python experimental_pinocchio_pyroki/solve_continuous_plan.py \
  output/base_link8_plan.json output/experiments/differential/ik.json \
  --urdf /path/dual_fr3_kinematics.urdf --robot franka-sharpa \
  --backend differential --start-state output/chahua_state.json
```

只评估首帧与进场段时加 `--first-frame-only`；输出明确标记为单帧，不能冒充整段回放。
可以显式使用 `--frame0-backend continuous` 处理较远的首帧目标，再用
`--backend differential|placo` 追踪；不指定时所有帧都使用同一个 backend。
`replay_episode` 也支持两个 backend，每次建立独立的 `run-*` 目录：

```bash
./replay_episode 553 --ik continuous --home-state output/chahua_state.json
./replay_episode 553 --ik differential --start-from-current
```

短入口新增模式可以设置 `EEF_REPLAY_IK_PYTHON=/path/to/isolated-env/bin/python`，
选择对应的离线求解环境；默认仍是 `/usr/bin/python3`，旧路径解释器保持原配置。

`--start-from-current` 通过已有 snapshot 工具只读采集。未提交实验中的
`--home-state` / `--posture-aware` 在短入口中保留，选择 continuous；旧的
`solve_bimanual_plan.py` 不接受这些新参数，需改用独立入口。

## 参考构型与约束

- `--reference home`（默认）：除接近上一帧外，轻度偏向最初起始构型。
- `--reference previous`：参考构型就是上一帧，不再额外向起始姿态拉回。
- `--reference source`：逐帧参考采集关节角，要求每帧每臂都有完整源关节数据；缺失时报错。
- `--reference-weight` 默认 `0.05`，`--limit-weight` 默认 `0.001`；它们影响偏好，不放宽目标误差。
- 首帧每个关节相对起点默认不超过 `1.2 rad`；后续每帧不超过 `0.12 rad`。
- 后续每帧同时满足 URDF 的 `velocityLimit × Δt × velocity-scale`；`velocity-scale` 默认 1，最大 1。
- 每个末端最终都独立验证位置误差 ≤ `1e-5 m`、旋转误差 ≤ `1e-4 rad`。
- 保留时间戳和配置指定的手部 payload；关节名按名称映射，不依赖 URDF 的枚举顺序。

`continuous` 使用 SLSQP 和完整位姿等式约束；`differential` 使用有界阻尼最小二乘
求增量，再以线搜索减小位姿误差。两个后端均锁住非当前臂关节；所有内部迭代的总位移
受同一个帧间边界限制，不会通过多次小步绕过约束。它们都是局部方法，不承诺全局最短路径。

整段全部通过后才导出新轨迹。失败只写同名 `.failure.json`，包含 backend、本体、
失败帧、时间、误差原因和关节边界。已有成功输出不会被覆盖，重跑需选择新路径。
失败记录中的 `output_written=false` 不代表目标全局不可达。

单独的 `_start_transition.json` 使用五次插值，保留起止位置，并让起止速度和加速度为零；
默认速度/加速度/jerk 上限为 `0.09 rad/s`、`0.108 rad/s²`、`0.432 rad/s³`。
这些进场约束不等于整段 replay 已通过加速度和 jerk 检验。

新模式目前不自动平滑、不覆盖旧平滑结果，也不自动交给执行入口。
若另行调用 PyRoki，仍需对平滑后的末端误差重新验证。

## 微分 IK 与可微分 IK

`differential` 是用雅可比求关节增量的**微分 IK**。`solution_jacobian` 是另一种接口：
在 continuous 已收敛的局部分支上，对优化问题做隐式微分，返回
`dq / dxi`（形状为全关节数 × 6）。`xi` 为目标的右乘局部扰动
`T_target @ exp6(xi)`，顺序为局部平移三维、旋转三维，单位 m / rad。

```python
from eef_replay.ik.model import RobotModel
from eef_replay.ik.continuous import ContinuousIK
from eef_replay.ik.sensitivity import solution_jacobian

robot = RobotModel(urdf_path)
solver = ContinuousIK(robot, "left_fr3v2_link8", range(7))
q, result = solver.solve_near(target_matrix, previous_q, max_step=0.12)
dq_dtarget = solution_jacobian(solver, target_matrix, previous_q, q, max_step=0.12)
```

该接口要求远离激活边界、任务雅可比满秩、解满足驻点条件且 KKT 系统条件良好；
不满足就明确报错。种子、参考姿态、硬边界在求导时保持固定，不跨 IK 分支求导。
约束雅可比来自 Pinocchio，拉格朗日 Hessian 由雅可比的中心差分得到；这不是
PyTorch/JAX 的自动反向传播封装。测试以扰动目标后重新求 IK 的差分验证返回梯度。

## 新本体与新求解器的扩展位置

内置配置位于 `src/eef_replay/ik/profiles/franka-sharpa.json`。复制
`config/robots/template.json`，填写 `name`、各臂关节名、基座/末端帧、输入目标字段和
需要透传的手部字段，设置 `template=false`，再使用：

```bash
prepare-ik-replay plan.json output/other-body/ik.json --urdf /path/other.urdf \
  --robot-config config/robots/my-body.json --start-state start_state.json --backend continuous
```

本体配置不硬编码每臂 7 关节，也不固定双臂数量。当前封装支持固定基座、有限位的
标量旋转关节；URDF 的全部活动关节须与配置中的手臂关节集合一致。手部数据仅透传，
因此应使用手臂运动学 URDF，不能直接把含有额外活动手关节或浮动基座的模型塞进来。
相机数据准备、硬件驱动、旧平滑和执行入口仍按 franka-sharpa 设计；新本体先使用
已经准备好的基座坐标目标和独立 CLI，不能仅凭换一个配置就操作新硬件。

新 backend 在 `src/eef_replay/ik/backends.py` 注册。实现构造方法
`(robot, frame, active_indices)` 和 `solve_near(target, seed, reference=..., max_step=..., ...)`，
返回完整关节向量和 `IKResult`；使用同一套 CLI 后验验证和失败协议。

## 已验证范围

运行 `PYTHONPATH=src python -m unittest discover -s tests -v`。测试覆盖旧 CLI 在可选模块
被禁止导入时仍可运行、连续性和速度边界、左右臂隔离、目标/时间戳/手部透传、不同关节顺序、
6 个活动关节、失败不导出部分轨迹、源参考缺失拒绝，以及局部可微分接口的数值验证。

episode553 的已修复 664 帧目标上，continuous 首帧候选保持原先约 68.75° 的最大 home
偏移；整段仍复现第 529 帧失败。这次整理保留该限制，不将单帧成功写成整段成功。

当前可选求解不含双臂/环境碰撞检查。实测起点、碰撞模型和整段动力学约束需独立验证。

本次环境版本、31 项测试结果与 episode553 实际结果见 [验证记录](ik-validation.md)。

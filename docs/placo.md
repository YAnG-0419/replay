# Placo 源码快照与可选 IK 后端

官方项目：[Rhoban/placo](https://github.com/Rhoban/placo)。本仓库的快照位于
`third_party/placo/`，固定到提交 `93adb57776115eeecc713a9bb0a906daf7d86af5`，
上游版本为 **0.11.0**。上游源文件未作本地修改。

快照保留 C++ 源码、Python 绑定、辅助模块、构建文件、文档、MIT 许可证和随附
qpmad 的 Apache 2.0 许可证。省略上游的 `python/tests/` 大体积测试模型、对应测试启动脚本
和 `.github/` CI 配置；不复制嵌套 `.git`。`UPSTREAM.json` 记录提交、排除项和逐文件
SHA-256，便于核对和更新。复现复制可运行：

```bash
# 在目标目录尚不存在的 checkout 中运行；已有快照更新前需先审查并另行保存
python tools/vendor_placo.py /path/to/clean-official-placo-checkout \
  --revision 93adb57776115eeecc713a9bb0a906daf7d86af5
```

## 独立安装

源代码快照可供阅读、维护和编译；Python 后端使用已编译的 Placo 包，不会自动编译
或修改快照。建议创建独立环境，不在已有采集/执行环境中安装：

```bash
python3 -m venv .venv-placo
.venv-placo/bin/python -m pip install -e '.[ik,placo]'
.venv-placo/bin/python -m pip check
```

`placo` extra 固定为 `placo==0.11.0`，只有显式创建 Placo 后端时才导入它。
未安装时会明确报错，不会自动切回别的求解器。若要编译快照的 C++ 接口，参照其
`README.md`、`pyproject.toml` 与上游安装文档准备 coal、eigenpy、tinyxml2 等构建依赖。

## 使用同一套输入

```bash
.venv-placo/bin/prepare-ik-replay \
  output/base_link8_plan.json output/experiments/placo/ik.json \
  --urdf /path/dual_fr3_kinematics.urdf --robot franka-sharpa \
  --backend placo --start-state output/chahua_state.json

# 短入口中的数据准备仍使用已有 ROS 环境，IK 使用指定的独立解释器
EEF_REPLAY_IK_PYTHON="$PWD/.venv-placo/bin/python" \
  ./replay_episode 553 --ik placo --home-state output/chahua_state.json
```

当 home 与首帧相差很大时，局部 QP 可能在首帧就停住。可以**显式**增加
`--frame0-backend continuous`，先在同样的首帧范围内求初始化，再由 Placo 追踪后续帧。
该选择同时写入输出的 `frame0_backend` 和每帧的 `backend`，不自动切换求解器。
首帧初始化和后续求解仍须整段成功才导出完整结果。

新本体仍通过 `--robot-config` 扩展，不在 Placo 适配器里硬编码 Franka 的关节数量或名称。
生成文件使用独立输出目录，不覆盖旧 Pinocchio 或 PyRoki 结果。

## 适配器行为

适配器把上一帧解写入 Placo，固定浮动基座并屏蔽非当前臂关节。它将 URDF/部署限位与
上一帧允许的变化范围取交集，作为**整次求解期间不变的关节边界**。
每次内部 QP 的增量还被限制在 `0.1 rad` 内；允许多个内部迭代，但它们不能越过
整帧边界。真实采样间隔与速度上限由公共 CLI 转为整帧边界。

末端使用 Placo 的 scaled 任务，结合接近上一帧、参考姿态和远离限位的次级代价。
若一次线性化不能到目标，按缩放增量逐步逼近。最终仍以公共 Pinocchio FK 独立核对
每个末端的 `10 μm / 0.1 mrad` 误差要求，并验证非当前臂关节不变。数值求解失败时
只导出失败记录，不放宽约束、不输出部分回放。

Placo 适配器本身是局部 QP 方法，没有整段分支搜索，也未启用碰撞约束。这里忽略碰撞
模型加载，并在输出明确标记 `collision_checked=false`；原有 MoveIt 路径独立保留。
三种新后端使用相同验证协议，但其姿态代价的数值形式不同，不能将相同权重视为完全相同的优化问题。

## 验证

```bash
PYTHONPATH=src .venv-placo/bin/python -m unittest discover -s tests -v
```

安装 Placo 后，共用的连续跟踪测试会同时检查 continuous、differential、placo。
另外检查 Placo 的左右臂隔离、过小范围内不可解时拒绝输出、失败后重用同一实例恢复求解，
以及源码快照和许可证完整性。未安装 Placo 时，只跳过其数值测试，其他后端与原流程仍可验证。

本次完整结果与已知限制见 [IK 验证记录](ik-validation.md)。

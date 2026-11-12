# 本体配置入口

当前内置本体为 `franka-sharpa`，配置原件在
`src/eef_replay/ik/profiles/franka-sharpa.json`，会随 Python 包安装。

新增本体可以先复制本目录的 `template.json` 并通过 `--robot-config` 试验。
模板故意带有 `template=true`，未填写时求解器会拒绝使用。请为不同本体选择唯一名称，
提供独立的 URDF、具名起始状态、标定与输出目录。

手臂配置、IK 算法和执行驱动是独立层。完整约定和命令见 [IK 使用说明](../../docs/ik.md)。

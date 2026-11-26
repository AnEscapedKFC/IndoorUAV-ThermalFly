# IndoorUAV-Agent 在会议室与走廊的 RGB 飞行验证

会议室空调任务沿用冻结观察策略，当前评分分为严格完成（距空调 0.4–0.8 米且居中）和大致完成（0.5–1.5 米）。[空调任务入口与评分说明](grounding_study/README.md)

[动力学飞行：13.5 米在线里程、航点检查开关与可见目标观察](grounding_study/metric_flight/REPORT.md)

本目录独立于 `../gs_gradnav_simulator`。模型进程运行作者发布的室内 π₀ 权重，仿真进程导入原项目的 RGB 3DGS 场景、控制器与四旋翼动力学。所有新代码、权重、环境和运行记录都位于此目录。任务指令、起点、目标评分和专家轨迹来自原项目的 `outputs/expert_rgbt_20261123`。

四条原指令的飞行录像、终点误差、换指令结果和限幅实验见[本机实测报告](REPORT.md)。

## 独立云台俯仰

导航时加 `--gimbal auto`，会议室启用学习得到的俯仰角，走廊保持固定云台。省略此选项可复现原来的固定云台导航。云台只改变下一帧的观察方向，原 π₀ 权重继续输出十个四维航点。推理使用当前 RGB、当前云台角和当前下达的英文指令，按 1 Hz 预测一秒后的附加俯仰，在 20 Hz 控制周期内限速更新。该版本按指令中出现的六类会议室目标名及其别名编码语言。长指令沿用接入层当前的分段方式。

在本目录启动下节的策略服务后，可用以下命令运行。更换输出路径可保留多次飞行结果。

```bash
sim_python=/home/zr_fan/miniconda3/envs/thermal_gaussian_5070/bin/python
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 "$sim_python" -B sim_bridge.py \
  --episode room_search_laptop_sofa --gimbal auto \
  --worker 'python -B bridge_client.py' --output runs/my_gimbal_search
```

`--mode replay --gimbal learned` 用专家机身控制与学习云台做同路径对照，`--mode replay --gimbal fixed` 固定起始角，`--mode replay --gimbal recorded` 使用记录的专家云台。学习云台的目标角、实际限速指令和渲染角分别保存在 `gimbal.jsonl` 与 `frames.jsonl`。检查点和训练用的冻结 RGB 编码器在 `gimbal/`，重训命令为 `"$sim_python" -B gimbal/train.py --encoder gimbal/squeezenet1_0-b66bff10.pth`。[训练数据与权重摘要](gimbal/room_pitch.json) · [云台配对评测](results/gimbal/REPORT.md)

## 运行环境

在本机以本目录为工作目录。模型使用本地独立环境和作者的 openpi 版本，仿真使用已有的 `thermal_gaussian_5070` 环境。模型资源来自[官方仓库](https://github.com/valyentinee/IndoorUAV-Agent)和[官方权重页](https://modelscope.cn/models/valyentine/IndoorUAV-Agent)。归一化统计取自作者随仓库发布的训练数据元信息。

在一个终端启动权重服务，等待日志中出现 `IndoorUAV-Agent ready`。服务按本机 Unix 套接字处理请求，供离线评测和不同场景共享一份权重。

```bash
cd /home/zr_fan/3DGS_env/indooruav_rgb_bridge
python -B bridge_client.py serve -- scripts/model_worker.sh
```

在另一个终端设置仿真解释器与线程数。下列示例使用新的输出路径，飞行程序会拒绝覆盖已有回合。

```bash
cd /home/zr_fan/3DGS_env/indooruav_rgb_bridge
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
sim_python=/home/zr_fan/miniconda3/envs/thermal_gaussian_5070/bin/python
"$sim_python" -B sim_bridge.py --episode room_inspect_ac --worker 'python -B bridge_client.py' --output runs/repeat/room_inspect_ac --max-seconds 60
"$sim_python" -B sim_bridge.py --episode corridor_high_return --worker 'python -B bridge_client.py' --output runs/repeat/corridor_high_return --max-seconds 60
```

切换到 `room_search_laptop_sofa` 或 `corridor_climb_accelerate` 即可运行测试划分中的两个回合。附加 `--subinstructions tasks/room_search_laptop_sofa.json` 或相应的走廊文件，可使用人工从原指令切分的短句。切换时机由运动输出决定，新参考图取自实际飞行时的观测。

## 评测设计

- **真实动作回放**逐帧施加原专家的角速度和推力，同时恢复记录的云台角，检验起点、标定、渲染和动力学。
- **专家未来航点对照**用专家未来位姿替代模型输出，让同一个新接入的航点跟踪器执行；该实验单独检验跟踪误差。
- **原指令闭环**向模型提供当前 RGB、任务起点参考 RGB、完整英文指令和当前四维位姿。使用模型原生 10 个位姿中的最后一个作为约一秒的目标，跟踪后重新渲染。平移的水平原点和航向按起点重设，高度沿用场景米制高度。会议室与走廊均固定起始云台。控制器只看到当前运动状态和推理得到的目标。地标、路线与真值终点供评分。
- **子指令闭环**使用相同权重和控制器。原版 VLN 运行器读取预先分好的短句。此处取 `tasks/` 中的人工文本分解，单独报告这项结果。完成判定沿用官方运动阈值，小于 0.2 米且偏航变化小于 15° 时转入下一阶段或触发末端悬停。π₀ 输出四维位姿，主动停止能力需结合这套规则解读。
- **离线换指令**在四个回合各取 4 个专家决策点，对固定的当前图、参考图、位姿和随机种子，仅改变指令。比较未来一秒的模型位姿与专家实飞位姿，另列原地不动基线。离线误差和闭环完成率衡量不同的能力。

所有闭环飞行保留 `rgb_fpv.mp4`、`frames.jsonl` 和 `summary.json`。模型飞行另保留 `decisions.jsonl`，其中记录送入权重的图像路径、指令、随机种子、当前位姿、原始十步输出及实际采用的目标。场景碰撞表面和已观测自由空间取自原项目，越出已知空间与碰撞分别记录。

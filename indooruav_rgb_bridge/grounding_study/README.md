# IndoorUAV-Agent 方位提示与航点检查对照

[会议室家具任务、640×480 双模态录像和深度辅助飞行](metric_flight/DEPTH_FURNITURE.md)

[实验报告](REPORT.md) · [预先固定的设计](protocol.md) · [背向起点补充设计](reverse_protocol.md) · [完整指标](results/metrics.json) · [配对与完整性校验](results/verification.json)

策略和数据记录位于独立接入层。检测器接收原始 RGB 及当前物体名；历史方位只保存曾检测到的朝向、观测时的位置和决策编号。航点检查只读取指令、模型输出和飞行器自身运动状态。执行脚本中的地标投影只写入日志，目标坐标、专家末帧与抵近距离阈值在评分阶段使用。真实起点及初速按原实验初始化，新起点使用预先声明的相对偏移。协议里的 max-step 是研究设计的工程假设。没有使用 RBF、邻近轨迹查询、专家拟合或本地训练。

## 运行入口

沿用接入层的 29999 检查点、模型服务与 thermal_gaussian_5070 渲染环境。检测器使用独立的系统包可见虚拟环境，增加 transformers 4.49.0、timm 1.0.15。权重来自 [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny/tree/a2bb814dd30d776dcf7e30523b00659f4f141c71)，主权重 SHA256 为 `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3`。相机估计从检测框中心解算校准后的三维光线，输出机身水平航向坐标中的角度。方位记忆沿固定世界方向传播，移动超过 0.35 米或超过 8 次决策即弃用。

在接入层根目录启动模型服务。

```bash
python -B bridge_client.py --socket runtime/grounding-study.sock serve -- scripts/model_worker.sh
```

在本目录执行以下命令。已完成的回合按结果文件跳过。输入完全相同的模型推理使用内容哈希缓存，哈希包含两张图、姿态、指令和种子；动力学在各组独立推进。原始画面、逐步模型候选与执行航点、每个物理控制周期的实际状态、投影与终止原因分别存于对应回合目录。

```bash
./launch.sh panel.py
./launch.sh run.py direct baseline hint guard both
./launch.sh reverse.py direct
./launch.sh run.py flight baseline hint guard both
./launch.sh reverse.py flight
./launch.sh score.py
./launch.sh verify.py
./launch.sh plot_bearings.py
```

主实验回合位于 [results/runs](results/runs)，大角度补充回合位于 [results/reverse/runs](results/reverse/runs)。两者都按 direct/flight、baseline/hint/guard/both、起点名称分目录。每个回合有 episode.json、frames.json、decisions.json 及每次推理输入的原始 PNG。单步方向对照分别保存在 [左右方位记录](results/static.json) 与 [机体方位记录](results/angles.json)。

走廊数值任务和会议室空调任务由 `./launch.sh run.py flight corridor --waypoint-check auto` 和 `./launch.sh run.py flight room --room-policy visual` 启动。关闭数值检查使用 `--waypoint-check off`。无明确数值的高度/航程指令自动跳过检查。旧四组研究冻结的执行器保存在 frozen_run.py；重放旧运行须使用该快照，新开关的产物独立保存。[走廊与旧会议室对照实验](metric_flight/REPORT.md)

会议室的冻结策略指令要求机体中心距空调标注中心 0.5–1.5 米（含边界），空调中心在 RGB 画面水平、垂直各 30%–70% 的范围内。机载控制根据当前 RGB 及相机运动估距，按指令完成观察动作，再稳定悬停两秒。评估以独立标注复核全段几何，并记录停止时及悬停四个时刻的 RGB 检测。运行 `./launch.sh metric_flight/score_object.py` 可按下述两档标准对同起点的历史记录和新版本重新评分。旧会议室实验可通过 `--waypoint-speed legacy --room-goal reference` 复跑。[空调逐回合指标](metric_flight/results/room_ac_0p5_1p5/metrics.json)

六个既有起点按新距离和居中条件重算，旧策略有四条在整个悬停段满足几何判据；新策略飞行六条均通过几何与 RGB 检查。另用两个本轮新增起点飞行，两条均通过。八条的悬停终点距离为 1.377–1.455 米，碰撞和越界各零次。复跑新增起点可使用 `./launch.sh run.py flight room --waypoint-speed legacy --cases fresh_room_inspect_ac fresh_room_obj_ac_west_to_east`。[八条飞行的末帧 RGB](metric_flight/results/room_ac_0p5_1p5/terminal_rgb.png)


### 双模态动力学复飞与当前评分

严格完成要求整个两秒稳定悬停段距空调 0.4–0.8 米，空调中心位于 RGB 图像中央 40% 内；大致完成要求整个稳定悬停段距空调 0.5–1.5 米。两档独立计算，共同要求完成指令中的爬升或绕行、无碰撞且在已知空间内。指标只参与评分。视频组使用上一轮的任务文本、固定云台和视觉控制参数；绕过模型与检测缓存重新推理。每步用两台相机的标定同步渲染，红外只用于录像。

在接入层启动模型服务后，于本目录依次执行：

```bash
./launch.sh run.py flight room --waypoint-speed legacy --record-rgbt repeat
./launch.sh run.py flight room --waypoint-speed legacy --record-rgbt shifted
./launch.sh metric_flight/audit_rgbt.py
./launch.sh metric_flight/rgbt_report.py
```

[全部双模态录像、评分与来源审计](metric_flight/results/room_ac_rgbt/INDEX.md)

### 中间航点速度衔接

会议室与走廊默认用 `--waypoint-speed through` 给普通运动段指定三维平移末速度，并把该速度用于下一连续运动段的起始参考。云台角、偏航插值、动力学与评分沿用原设置。原航点插值可用 `--waypoint-speed legacy` 复现。走廊默认指令为前飞 13 米，历史 13.5 米任务可用 `--corridor-distance 13.5`。

```bash
./launch.sh run.py flight room --record-rgbt repeat
./launch.sh run.py flight room --record-rgbt shifted
./launch.sh run.py flight corridor --corridor-distance 13 --record-rgbt repeat
./launch.sh run.py flight corridor --corridor-distance 13 --waypoint-speed legacy --record-rgbt repeat
./launch.sh metric_flight/waypoint_speed_report.py
```

[逐路线录像、飞行结果和抖动对照](metric_flight/results/waypoint_speed/INDEX.md)

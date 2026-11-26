# Lab · 墙面修复 RGB + PlanarGS 锚定红外 30k

`models/rgb.ply` 为 PlanarGS 墙面修复 30k 高斯模型，`models/thermal.ply` 为 PlanarGS 锚定 30,000 步的红外高斯模型。相机内参和注册视角在 `inputs/manifest.json`，该红外终点的双相机变换在 `inputs/rig.json`。渲染对应时刻的两幅图时，使用 `W_thermal = T_rgb_to_thermal @ W_rgb`。

RGB 和红外高斯保留原始 SfM 坐标，碰撞表面与自由空间使用模拟器米制坐标。两者通过 [frame.json](inputs/frame.json) 对齐。

原始模型来源见 [source.json](inputs/source.json)，模型和输入文件的校验值见 [assets.json](inputs/assets.json)。

两路模型已通过模拟器 GPU 渲染核对。[注册机位验证记录](inputs/registered_view_check.json)

## 双模态查看器

在 WSL 运行以下命令，然后打开 [本地查看器](http://127.0.0.1:7914/)。交互查看器通过训练配置读取本机原始数据集。

```bash
bash /home/zr_fan/ThermalGaussian_Surface/capture65/python.sh viewer.py \
  --scene-bundle /home/zr_fan/3DGS_env/gs_gradnav_simulator/scenes/lab_wall_anchored_30000 \
  --view-mode rig --start rgb_c01_0000159 --port 7914
```

## 碰撞地图

[障碍表面 PLY](collision/rgb_surface_m.ply) 包含 82,978 个表面点，融合体素边长为 4 cm。[已观测自由空间](collision/free_volume.npz) 包含 82,915 个被标为空闲的 10 cm 网格。机体及安全余量所覆盖的网格都需要通过自由空间检查。网格未标记或越界时视为未知。[1.10 m 高度的水平切片](collision/map.png) · [构建记录](collision/build.json)

场景地面为 Z=0。现场实测尺度缺失，采用层高 2.80 m 的假设，比例为 0.7451 m/SfM 单位。使用本场景的 RGB 深度、与 SfM 对齐的预测深度、多个分离机位支持的特征点约束表面与自由空间。设备、工位间隙和未观测区域可能缺测。

验证使用 0.60 × 0.60 × 0.12 m 机体、0.05 m 安全余量，覆盖出生点、地面、墙面和室内水平物体的连续扫掠。留出相机中可见、未直接加入障碍点的 SfM 特征到碰撞表面的距离中位数为 2.7 cm。落入自由网格的特征有 11/2483。这些特征与地图共享 SfM 坐标，现场精度仍受尺度假设和观测覆盖影响。[验证数据和测试坐标](collision/validation.json)

### 在模拟器中加载

在模拟器项目根目录使用 `thermal_gaussian_5070` 环境。场景适配器将米制机体相机转换回原 RGB 和红外重建坐标，障碍点云以 `up_axis="z", scale=1` 加载。

```python
import json
from pathlib import Path
from engine import PolicyFlightSimulator
from scenes.indoor_rig import IndoorKnownSpace, IndoorRigScene

scene = Path("scenes/lab_wall_anchored_30000")
verified = json.loads((scene / "collision/validation.json").read_text())
known = IndoorKnownSpace(scene)
sim = PolicyFlightSimulator(
    scene / "models/rgb.ply",
    point_cloud=scene / "collision/rgb_surface_m.ply",
    scene_factory=IndoorRigScene,
    spawn=verified["verified_spawn_m"],
    yaw=verified["verified_spawn_yaw_deg"],
)
# 规划时用 known.contains_box(position, body_rotation, half_size_with_margin) 检查空域。
print(sim.last_receipt["collision"])
sim.close()
```

如取得实测层高，可用原训练数据在新的输出目录中重建，`--height-m` 填入现场测量值。

```bash
bash /home/zr_fan/ThermalGaussian_Surface/capture65/python.sh -B \
  /home/zr_fan/3DGS_env/gs_gradnav_simulator/scripts/build_indoor_collision.py \
  /home/zr_fan/3DGS_env/gs_gradnav_simulator/scenes/lab_wall_anchored_30000 \
  /home/zr_fan/PlanarGS/lab_native /tmp/lab_wall_anchored_30000_rebuilt --height-m 2.80
```

# IndoorUAV-ThermalFly

[导航接入层](indooruav_rgb_bridge/README.md) · [动力学模拟器](gs_gradnav_simulator/README.md) · [会议室家具任务及基线](indooruav_rgb_bridge/grounding_study/metric_flight/results/furniture/REPORT.md)

两个子目录保持与工作环境相同的相对位置。通过 `git submodule update --init` 获取固定版本的 IndoorUAV-Agent。场景模型、碰撞与标定资产、运行环境和权重由各模块按本地路径加载，详见[权重配置](indooruav_rgb_bridge/model_setup.md)。

[640×480 深度辅助飞行与复跑命令](indooruav_rgb_bridge/grounding_study/metric_flight/DEPTH_FURNITURE.md)

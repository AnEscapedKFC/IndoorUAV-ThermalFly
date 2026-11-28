# IndoorUAV-ThermalFly

[导航接入层](indooruav_rgb_bridge/README.md) · [动力学模拟器](gs_gradnav_simulator/README.md) · [会议室家具任务及基线](indooruav_rgb_bridge/grounding_study/metric_flight/results/furniture/REPORT.md)

两个子目录保持与工作环境相同的相对位置。通过 `git submodule update --init` 获取固定版本的 IndoorUAV-Agent。场景模型、碰撞与标定资产、运行环境和权重由各模块按本地路径加载，详见[权重配置](indooruav_rgb_bridge/model_setup.md)。

[640×480 深度辅助飞行与复跑命令](indooruav_rgb_bridge/grounding_study/metric_flight/DEPTH_FURNITURE.md)

[640×480 配对测试结果和视频](indooruav_rgb_bridge/grounding_study/metric_flight/results/furniture_640/REPORT.md)

## 会议室持续观察实验

加入沙发全貌和门口独立黑色扶手椅。24 个动力学回合中，取得并保持合格画面 14/24，自主确认完成 11/24。每回合记录同步 RGB 与热红外。可查看[逐次验收结果](indooruav_rgb_bridge/grounding_study/metric_flight/window_results/REPORT.md)和[新标准及运行命令](indooruav_rgb_bridge/grounding_study/metric_flight/OBSERVATION_WINDOW.md)。

## 空调与笔记本的观察高度实验

新版接入层支持平视正面、从下方看出风口和从上方看屏幕与键盘。云台机械范围为 −60° 至 +30°，高度与接近程度由语言方向、实时 RGB 检测框和本次飞行的 RGB-D 记录共同决定。预扫描之后检查静止机体包络，遇到深度冲突先在原地复看。实验使用原四旋翼动力学，记录 640×480 同步 RGB 与热红外。运行说明及验收规则见[协同观察实验](indooruav_rgb_bridge/grounding_study/metric_flight/COORDINATED_INSPECTION.md)。

严格完成 4/12 个空调与笔记本回合，平视空调 0/4，仰视出风口 1/4，俯视笔记本 3/4。另有一次沙发起点冲突在导航前被拦截。13 次记录零碰撞、一次越出已知空间。[逐次结果与失败原因](indooruav_rgb_bridge/grounding_study/metric_flight/coordinated_results/REPORT.md) · [悬停画面](indooruav_rgb_bridge/grounding_study/metric_flight/coordinated_results/hover_views.jpg)

## 历史 RGB-D 目标观察

会议室新增沙发全貌、笔记本屏幕与键盘、空调面板与出风口三类无绝对距离指令，每类有多个起点。接入层保存同步 RGB、米制深度、相机位姿与云台角，融合当次飞行看过的目标表面，使用遮挡、有效像素和跨视角边缘与深度一致性选择关键帧、调整观察位置，并保留局部机体安全检查。运行命令、输出格式、逐帧画面验收和局限见[任务协议与复跑方法](indooruav_rgb_bridge/grounding_study/metric_flight/HISTORY_INSPECTION.md)。[逐次画面、航程与验收结果](indooruav_rgb_bridge/grounding_study/metric_flight/history_review_results/REPORT.md)分别记录原定、补充和冻结参数后的较远起点。录像、逐帧深度及完整表面观测索引在运行机器本地生成。

最终版在同一会议室的 12 架次中完成 10 次，沙发 3/4、笔记本 3/4、空调 4/4，零碰撞、零越界。三个冻结参数后选择的较远起点均通过。验收要求目标部位可辨、主动停止并稳定悬停两秒。两次失败都在起点触发了局部深度安全制动。

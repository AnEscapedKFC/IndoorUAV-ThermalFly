# 历史 RGB-D 目标观察飞行实测

会议室 RGB 3DGS 与同步热红外 3DGS，动力学仿真，20 Hz 单模态 640×480。每条英文指令不含绝对距离，每类任务使用四个起点。助手按预先约定的画面要求检查飞行中和终止时的 RGB 帧；对主动停止的架次另核验末段两秒的姿态稳定性。飞行距离是机体累计航程。本表记录所有起点，包括零航程的安全制动。

## 指令

- Fly toward the black sofa and show its whole front. Keep both ends, the backrest and the seat in the picture while hovering.
- Fly toward the open laptop and inspect its screen and keyboard together. Hover with both details clearly in view.
- Fly toward the wall-mounted air conditioner. Inspect its front panel and lower air outlet together, then hold that view.

## 结果

完整完成 10/12。碰撞事件 0 次，越界采样 0 次。共核验 80 个同步关键帧。画面可辨性 11/12，初始六架次完整完成 4/6，随后三个斜向起点 3/3，冻结后的三个较远起点 3/3。沙发 3/4、笔记本 3/4、空调 4/4；两次失败均在起点被局部表面证据连续拦截。

| 目标与起点 | 分组 | 航程 m | 任务时长 s | 关键帧 | 末段画面 | 稳定悬停 | 完整完成 | 画面与终止原因 |
|---|---|---:|---:|---:|---|---|---|---|
| [sofa_east](sofa_east.jpg) | 前六 | 0.00 | 3.5 | 1 | 是 | 否 | 否 | 两端、靠背与坐面在远景可辨；起点局部图反复制动，未执行抵近与主动悬停。 |
| [sofa_west](sofa_west.jpg) | 前六 | 0.87 | 6.5 | 5 | 是 | 是 | 是 | 两端与靠背完整，桌后仍能看到坐面，末段构图稳定。 |
| [sofa_diagonal](sofa_diagonal.jpg) | 补充斜向 | 1.16 | 7.5 | 5 | 是 | 是 | 是 | 斜向视角可同时辨认两侧扶手、完整靠背与坐面；白布局部遮挡。 |
| [sofa_long](sofa_long.jpg) | 冻结后新起点 | 1.73 | 10.0 | 6 | 是 | 是 | 是 | 正面略带斜角，两个扶手与靠背、坐面同时可辨，茶几边缘清楚，悬停稳定。 |
| [laptop_east](laptop_east.jpg) | 前六 | 1.56 | 7.5 | 8 | 是 | 是 | 是 | 屏幕内图形可辨，键列和键盘底座完整同框，末段稳定。 |
| [laptop_west](laptop_west.jpg) | 前六 | 0.00 | 3.5 | 1 | 否 | 否 | 否 | 起点远景能辨认笔记本，键列细节不足；八次制动后结束。 |
| [laptop_diagonal](laptop_diagonal.jpg) | 补充斜向 | 1.13 | 6.0 | 6 | 是 | 是 | 是 | 屏幕显示内容与下方键列同时清楚，末段稳定。 |
| [laptop_long](laptop_long.jpg) | 冻结后新起点 | 1.94 | 9.5 | 8 | 是 | 是 | 是 | 远处由整张会议桌进入笔记本视图，末段显示屏图形与键列同时可辨，悬停稳定。 |
| [ac_east](ac_east.jpg) | 前六 | 2.79 | 13.5 | 11 | 是 | 是 | 是 | 完整面板、左右边界和下沿出风条可分辨，悬停时仍在画面内。 |
| [ac_west](ac_west.jpg) | 前六 | 2.30 | 12.0 | 10 | 是 | 是 | 是 | 侧面观察仍能区分机身面板与下方风口，末段稳定。 |
| [ac_diagonal](ac_diagonal.jpg) | 补充斜向 | 2.33 | 12.5 | 9 | 是 | 是 | 是 | 正面视角能区分完整机身与下沿出风条，末段稳定。 |
| [ac_long](ac_long.jpg) | 冻结后新起点 | 2.46 | 12.5 | 10 | 是 | 是 | 是 | 由带墙面与顶部的大视野接近机身，末段完整面板、下沿横向出风条同框，悬停稳定。 |

最初六次的东西起点、随后补充的三个斜向起点均用于开发诊断。修正前后这九次的首帧 RGB 和热红外哈希逐对相同。冻结最终参数后，对每个目标另列四个较远候选；排除起点包络净空不足、首帧目标不可辨、或起点预观察后初始短前移不能获得自由空间支持的候选。在合格候选中，选目标掩码内首帧光轴深度中位数最大的一个，只运行该新起点一次。这三次是有条件的同场景新起点验证。所有候选和筛选结果见[起点筛选记录](long_start_selection.json)。

初版采用颜色前景分割，有时将前方茶几计入沙发。最终改为 SlimSAM 实例掩码，部件分界仍按 RGB 边缘估计，例如沙发前侧可能随坐面归类。笔记本东侧与斜向起点初版分别飞满 60 s，航程 11.29 m 和 13.13 m，出现反复进退。修正云台限位下的机体高度调节与部件像素判据后，两者均稳定停止。未做去掉历史的同条件消融，也未在新房间验证。

## 关键帧与数据

[沙发关键帧](sofa_diagonal_keyframe.jpg) · [笔记本关键帧](laptop_east_keyframe.jpg) · [空调关键帧](ac_east_keyframe.jpg)。三栏对应同一时刻的 RGB、部件分割和深度。不同实例视角可能有分割误差。格子计数只代表本次飞行测得的目标表面，不代表真实物体全貌覆盖率。

[逐次指标与精确起点](summary.json) · [协议、输出格式与复跑命令](../HISTORY_INSPECTION.md)。本机完整数据位于 metric_flight/results/history_inspection_final/，每个起点的 rgb_thermal_fpv.mp4、rgb_fpv.mp4、thermal_fpv.mp4 和 observations/ 均可单独打开。仓库提供静帧证据。

from evaluate import OUT,HERE,load
m=load(OUT/'metrics.json')
labels={'corridor_high_return':'原定高返航起点','corridor_climb_accelerate':'原爬升起点','new_corridor_high_return':'偏航返航起点','new_corridor_climb_accelerate':'偏航爬升起点','reverse_corridor_high_return':'东段反向起点','room_inspect_ac':'单独观察空调','new_room_inspect_ac':'偏转后观察空调','room_obj_ac_east_to_west':'原东向西弧线','room_obj_ac_west_to_east':'原西向东弧线','holdout_room_obj_ac_east_to_west':'新东向西弧线','holdout_room_obj_ac_west_to_east':'新西向东弧线'}
text=[];add=text.append
def tab(head,rows):
    add('');add('| '+' | '.join(head)+' |');add('|'+'|'.join(['---']*len(head))+'|')
    for row in rows:add('| '+' | '.join(str(x) for x in row)+' |')
    add('')
def yn(v):return '是' if v else '否'
def paired(a,b,key):return f'{a[key]:.2f} → {b[key]:.2f}'
add('# 动力学飞行中的 13.5 米航程和可见物体观察')
add('')
c_off=list(m['corridor']['off'].values());c_auto=list(m['corridor']['auto'].values())
r_before=list(m['room']['hint'].values());r_after=list(m['room']['visual'].values())
def cnt(rs,k):return sum(r[k] for r in rs)
add(f"开启自动检查后，走廊按无碰撞计 {cnt(c_auto,'no_collision')}/{len(c_auto)}，按无碰撞且达到 13.5±0.4 米计 {cnt(c_auto,'strict')}/{len(c_auto)}。关闭时两项分别为 {cnt(c_off,'no_collision')}/{len(c_off)} 和 {cnt(c_off,'strict')}/{len(c_off)}。会议室仅提示到优化版的严格成功数为 {cnt(r_before,'strict')} → {cnt(r_after,'strict')}（各 {len(r_after)} 条），STOP 前连续一秒目标居中为 {cnt(r_before,'last_second_centered')} → {cnt(r_after,'last_second_centered')}。")
add('')
add('检查器设为自动时，数值必须与飞行高度或飞行距离明确关联，两者任一即可触发。只有高度时只修正高度；只有距离时执行相对位移与里程控制，不引入旧高度设定。关闭开关时仍计算里程、记录碰撞并按相同规则评分。会议室三类指令均没有数值高度或航程，检查器全程自动关闭。视觉观察策略有独立开关。')
add('\n## 走廊\n')
add('所有比较组使用同一句英文 Fly forward 13.5 meters.。每 0.05 秒根据真实机体位置变化累计三维路径长度，两组也都获得累计值、剩余值及沿起始机头的有符号净位移。碰撞判定优先，首次达到 13.5 米即完成飞行。主指标分别是全程无碰撞、无碰撞且实际航程误差 ≤0.4 米。初始机头净进度及沿该方向直行 13.5 米的平面终点误差另外列出。走廊不要求悬停、特定终点偏航或旧专家终点。已知空域越界单独记录。')
tab(['起点','检查关：实飞米数／碰撞','自动：实飞米数／碰撞','自动：初始前向净进度（米）','自动：直线终点平面误差（米）'],[[labels[n],f"{a['path_m']:.3f}／{yn(a['collision'])}",f"{b['path_m']:.3f}／{yn(b['collision'])}",f"{b['forward_m']:.3f}",f"{b['straight_goal_xy_error_m']:.2f}"] for n,b in m['corridor']['auto'].items() for a in [m['corridor']['off'][n]]])
add('自动组根据剩余距离约束模型航点，持续推进直至在线里程到达；从当前 RGB 线段的消失方向估计走廊轴，保留起始前向半平面。关组让模型独立提出航点和提前停止。开启后的成功属于冻结模型与外部里程控制的组合。初版仅沿初始航向闭环，两条小偏航路线发生碰撞；正式版在这两条均达到航程。反向起点从 x=31.55 米向东出发，在 x≈37.25 米处发生碰撞，此时路径约 5.81 米，场景标定的东侧 x 上界为 38 米。')
add('\n## 会议室\n')
add('六个起点的初始空调均可见，固定云台，种子 42。新的两个弧线起点相对原机身分别前移/后移 0.16 米、侧移 0.12 米、改变偏航 ±8°。四个已有提示组使用相同的真实飞行记录重算，两处新增起点的提示组和全部六条优化组实际飞行。接近时用检测框、相机内外参和里程估计目标方向与距离，以 0.80 米的通用空调宽度初始化尺度，获得足够视差后通过多帧视线修正。弧线阶段增加沿目标切线的移动，用目标水平边缘估计正面朝向，选择互补视角；检测短暂失配时用前后向一致的当前帧光流继续确认。达到观察角度并连续居中后减速，执行 2 秒悬停。优化组以 0.5 秒间隔做视觉修正；提示基线沿用 1 秒策略决策。本对照测量整套观察执行器的效果，包括控制周期与减速方式。')
tab(['任务','最近可见距离（米）','抵近后连续可见绕角（°）','停止前 1 秒持续居中','终点误差（米）','0.4 米严格成功'],[[labels[n],paired(a,b,'min_visible_distance_m'),paired(a,b,'continuous_visible_arc_deg') if b['arc_task'] else '无绕行要求',yn(a['last_second_centered'])+' → '+yn(b['last_second_centered']),paired(a,b,'terminal_error_m'),yn(a['strict'])+' → '+yn(b['strict'])] for n,b in m['room']['visual'].items() for a in [m['room']['hint'][n]]])
add('表中为仅方位提示 → 优化。抵近沿用每条任务原有观察距离阈值，具体值见逐回合 JSON。连续可见绕角取首次抵近之后、几何目标中心保持在画面内的单个连续区间最大跨度；居中绕角另按中央 80% 计算。停止前考察下达 STOP 之前最后 1 秒每一帧，目标中心必须在图像两轴各 30%–70% 内，悬停中居中单独核算。评分使用地标投影，可能受遮挡影响；在线策略使用实际 RGB。房间主严格指标只将终点位置容差从 0.12 米放宽至 0.4 米，沿用 ≤8° 航向误差、安全、完整稳定悬停与任务语义；漂移容差仍为 0.12 米。包含新停止前后居中要求的更严格结果在 metrics.json 的 strict_with_prestop_center 项。')
add('\n![动力学轨迹与走廊在线里程](results/physical_comparison.png)')
add('\n## 复跑\n')
add('在 indooruav_rgb_bridge 启动已有的 29999 模型服务，进入 grounding_study 执行以下入口。原始模型候选、最终下发航点、在线里程、物体方位、阶段和逐帧飞行状态保存在各回合目录。')
add('''\n```bash
# 在接入层根目录，用独立终端启动
python -B bridge_client.py --socket runtime/grounding-study.sock serve -- scripts/model_worker.sh
# 在 grounding_study 目录
./launch.sh run.py flight corridor --waypoint-speed legacy --corridor-distance 13.5 --waypoint-check off
./launch.sh run.py flight corridor --waypoint-speed legacy --corridor-distance 13.5 --waypoint-check auto
./launch.sh run.py flight room --waypoint-speed legacy --room-goal reference --room-policy hint --cases holdout_room_obj_ac_east_to_west holdout_room_obj_ac_west_to_east
./launch.sh run.py flight room --waypoint-speed legacy --room-goal reference --room-policy visual
./launch.sh metric_flight/evaluate.py
./launch.sh metric_flight/check.py
```''')
add('\n[逐任务指标](results/metrics.json) · [校验记录](results/verification.json) · [走廊飞行过程](results/corridor) · [会议室飞行过程](results/room) · [实验口径](protocol.md) · [控制代码](control.py) · [评分代码](evaluate.py)')
(HERE/'REPORT.md').write_text('\n'.join(text)+'\n')

"""Generate the tables from scored episodes after all four arms finish."""
from score import ROOT,OUT,get
M=get(OUT/'metrics.json')
arms={'baseline':'原模型','hint':'仅方位','guard':'仅航点检查','both':'两者合用'}
cohorts={'all':'全部起点','old':'既有起点','new':'小幅平移及转向的新起点','reverse':'大角度背向的新起点'}
names={'room_inspect_ac':'会议室／靠近空调','room_obj_ac_east_to_west':'会议室／空调东向西弧线','room_obj_ac_west_to_east':'会议室／空调西向东弧线','room_search_laptop_sofa':'会议室／笔记本→沙发','corridor_high_return':'走廊／1.82 米返航','corridor_climb_accelerate':'走廊／爬升加速再下降'}
items=M['episodes'];tables=[];put=tables.append
def table(header,data):
    put('| '+' | '.join(header)+' |');put('|'+ '|'.join(['---']*len(header))+'|')
    for row in data:put('| '+' | '.join(map(str,row))+' |')
    put('')
def title(x):put('');put('## '+x);put('')
def select(m,g='all',a=None,n=None):
    return [r for r in items if r['mode']==m and (g=='all' or r['cohort']==g) and (a is None or r['arm']==a) and (n is None or r['case'].removeprefix('new_')==n)]
def count(x,y):return f'{x}/{y}'
def val(v):return '—' if v is None else f'{v:.2f}'
def yes(x):return '✓' if x else '×'
def changes(k):return ' / '.join(str(v) for v in k)
put('# IndoorUAV-Agent 真实 RGB 方位与航点干预实验')
put('')
put('六条既有任务、四个新起点、两个背向新起点，各以四组策略执行直接渲染和动力学飞行，共 96 个独立执行回合。主判据为动力学严格任务成功率。所有试验使用种子 42 起步，各组推理噪声按决策编号配对。部署策略始终为冻结的 IndoorUAV 和冻结的 Grounding DINO Tiny，加上同一套通用规则。')
title('一、真实 RGB 能给对哪种方向')
put('四张已做目视核对的会议室 RGB 产生六个可见的图像–目标组合，匹配三个种子（16、42、77）。十个不可见组合来自两张走廊图、会议室搜索起点和一张没有笔记本的空调图。可见正例采用此前目视核验的标签与原图，负例按原图逐张复查；扩展方位的数值真值由地标坐标和已知相机转角计算。种子重复不算作新图像。[标签图](results/label_sheet.jpg) · [历史核验记录](../diagnoses/image/findings.md)')
put('')
table(['条件','可见时正确转向','正确且超过 2°','不可见时错误方向提示（含居中）','其中错误左/右','不可见时实际转动 >2°'],[[label,count(v['right_sign'],v['n']),count(v['beyond_2'],v['n']),count(v['absent_any_hint'],v['absent_n']),count(v['absent_direction_count'],v['absent_n']),count(v['absent_turn_2'],v['absent_trials'])] for k,label in [('base','原模型'),('gold','核验方位提示'),('rgb','检测器提示')] for v in [M['panels']['static'][k]]])
put('原模型从来不发显式方向提示；在目标不可见时发生转动可能是搜索行为，不能直接据此判为误报。检测器误给的一侧方位为走廊定高返航图上的“左侧空调”，另外两例分别在走廊爬升起点和会议室搜索起点误报“沙发居中”。')
put('')
put('另在上述四个会议室机位保持位置不动，以 0°、+90°、−90°、180° 转动机体，共 24 个机位–目标–朝向组合，前半平面、后半平面各 12 个。机头为 0°，右为正，八方位每格 45°。当前帧仅使用当前观测。历史组可使用该旋转前在同一地点真实看到的一帧和当前帧。实际转向成功定义为一次推理后与目标的绝对角误差缩小超过 2°，包括朝两侧转向正后方的情况。')
put('')
table(['条件','全部 24 个：给出方位 / 前后正确 / 八方位正确','后方 12 个：给出 / 前后正确 / 八方位正确','前方转向进步','后方转向进步'],
[[label,changes([all_['provided'],all_['hemisphere_correct'],all_['sector_correct']]),changes([back['provided'],back['hemisphere_correct'],back['sector_correct']]),count(front['better_2deg'],front['trials']),count(back['better_2deg'],back['trials'])] for k,label in [('base','原模型'),('gold','几何真值'),('rgb_current','仅当前 RGB'),('rgb_memory','RGB 加偏航历史')] for v in [M['panels']['azimuth'][k]] for all_,front,back in [(v['all'],v['front'],v['rear'])]])
put('后方的 12 个组合中，7 次当前画面误检覆盖了先前方位，另外 5 次使用历史观测，均正确判为后方。事后固定转身前同一目标的检测方位、仅按已知原地转角更新，24/24 的八方位标签正确。这里的六个初始图像–目标恰好全部检测正确。这是纯转动的离线诊断，未用作本次闭环策略；平移、遮挡与目标重现仍需要单独处理。')
put('')
put('![转身前后实际图像、检测框与相对方位](results/front_rear_examples.png)')
title('二、四组闭环')
put('每组每种执行方式各 12 条。已知区外直接渲染继续推理，直接执行没有实体扫掠和稳定悬停，物理严格成功与碰撞记为未测。越界和碰撞按至少发生一次的回合数各自统计；同一回合可同时发生两者，越界计数可能高于以越界终止的回合数。2 米 SR 和平均 NE 统一以专家轨迹实际终点计分；严格评分用任务指定目标。')
put('')
for mode,title_ in [('flight','动力学飞行'),('direct','直接渲染')]:
    put('### '+title_);put('')
    matrix=[]
    for g in cohorts:
        for a,label in arms.items():
            v=M['groups'][f'{mode}/{g}/{a}'];n=v['n']
            matrix.append([cohorts[g],label,'—' if v['strict'] is None else count(v['strict'],n),count(v['SR'],n),val(v['NE']),'—' if v['collision'] is None else v['collision'],v['outside'],count(v['ordered_rooms'],v['rooms'])])
    table(['起点','策略','严格成功','2 米 SR','NE（米）','碰撞','越界','会议室抵近观察顺序达成¹'],matrix)
put('¹ 只统计任务列出的抵近观察对象。单目标只需满足一次，多目标必须依序满足；弧线覆盖和悬停另计入严格成功。目标中心投影位于中央 80% 区域、且达到事后参考抵近范围作为观察代理，几何投影不能排除被遮挡。')
title('三、任务在哪里偏离')
put('数字按原模型／仅方位／仅航点检查／两者合用排列。下表为六条原任务直接渲染时首次触犯独立任务判据的决策编号。原始候选与真正执行的指令分别打分。“—”表示未观测到该表定义的偏离，仍可能未到终点。')
put('')
rows=[]
for n,label in names.items():
    raw=[];applied=[]
    for a in arms:
        r=select('direct','old',a,n)[0]
        for k,to in [('raw',raw),('executed',applied)]:
            e=r['first_task_violation'][k];to.append(str(e['n']) if e else '—')
    rows.append([label,changes(raw),changes(applied)])
table(['任务','首个原始候选违令步','首个执行违令步'],rows)
put('独立判据检查搜索未见目标前的过大平移、可见目标进一步偏离朝向、未观察即停止、走廊定高带、先爬升后下降以及偏离走廊方向。另存通用单步运动上限与原始、执行检查原因，其中位移过大仍可发生在沿着正确方向的首步。所有规则均为可观测代理。[逐回合判据及首错类别](results/metrics.json)')
put('')
rows=[]
for g in ['old','new','reverse']:
    for n in ['corridor_high_return','corridor_climb_accelerate','room_search_laptop_sofa','room_obj_ac_east_to_west','room_obj_ac_west_to_east']:
        entries=select('flight',g,None,n)
        if not entries:continue
        v=[next(r for r in entries if r['arm']==a) for a in arms]
        if 'high_return' in n:key='高度 MAE（米）';values=[val(r['altitude_MAE']) for r in v]
        elif 'climb_accelerate' in n:key='先升后降 / 先加后减';values=[yes(r['semantics']['climb_then_descend'])+'、'+yes(r['semantics']['accelerate_then_slow']) for r in v]
        elif 'search' in n:key='按序抵近笔记本、沙发';values=['、'.join(yes(o['hit_at'] is not None) for o in r['ordered']) for r in v]
        else:key='抵近后连续居中弧度跨度（°）';values=[val(r['arc_deg']) for r in v]
        rows.append([cohorts[g],names[n],key,*values])
table(['起点','实际飞行任务','指标','原模型','仅方位','仅检查','两者'],rows)
put('定高误差是按实际飞行时间加权的绝对误差；提前碰撞或停止会缩短测量时间。先升后降要求高度先增加至少 0.12 米、随后减少至少 0.12 米；先加后减要求水平速度先增加至少 0.15 m/s、随后至少减少 0.15 m/s。速度达标仍可能源于提前减速或碰撞，因此须与严格成功及终止原因合看。弧线用实际机身相对空调的方位跨度计量。')
put('走廊定高返航的起点相同。在原六条的动力学对照中，航点检查将时间加权高度误差由约 0.85 米降至 0.0045 米，终点距离却由 7.76 米变为 8.64 米。走廊爬升任务的飞行终点距离由 10.45 米降至 1.45 米，仍然越界，且没有达到先爬升后下降的规定幅度。两种现象分别说明了纠正高度、进入 2 米半径与完成任务的区别。')
put('')
put('大角度新起点的空调实际在右后约 167°，无人机此前没有见过目标。仅方位组没有产生方位提示，第一步即越界；两者合用组在第二步越界。相反，旋转探针中能成功推断后方的五个条件全部利用了先前观测。解决这种起步搜索需要能在本机实际转动中获取新证据的行为。')
put('')
title('四、划分、证据边界与复跑')
put('六个新起点分开报告。四个轻扰动起点比原起点平移约 0.14–0.18 米，偏航改变 12°。两条背向任务有更大平移和航向变化，空调初始位于机头右后约 167°，走廊目标位于左后约 160°。六个均与所有已有轨迹的起点坐标不同。所有组采用相同的初态、固定云台、预算和种子。既有六条直接和动力学基线重跑并与历史记录逐条核对。走廊任务没有指名物体，原模型和仅方位组、仅检查和合用组构成输入相同的对照。')
put('')
put('策略仅以指令、真实渲染的 RGB、相机标定、自身姿态与运动历史决策。专家首帧用于复原共同初态，地标投影只写评测日志，专家末帧和抵近阈值只用于评分。没有在本项目的轨迹上拟合权重，也没有路线检索或 RBF 回归。单次固定种子闭环和两个已有物理场景不能证明跨建筑泛化，背向诊断的四个机位也无法代表所有遮挡形式。')
put('')
put('语言顺序另外做了反事实检查。冻结版在当前会议室搜索指令中按笔记本→沙发查询；将英文改为先找沙发再找笔记本，它仍以笔记本为先。一个独立的按文本提取目标顺序的解析器通过四条措辞对照，且在本次六条原指令上产生与冻结版完全相同的策略初态。四组飞行结果始终来自同一冻结版，反序短句只验证解析，没有做反序任务飞行。[顺序对照记录](results/language_control.json)')
put('')
put('[预先固定的方案](protocol.md) · [背向起点方案](reverse_protocol.md) · [策略代码](policy.py) · [检测器代码](detector.py) · [评分代码](score.py) · [逐回合指标](results/metrics.json) · [完整性校验](results/verification.json) · [复跑与原始日志](README.md)')
(ROOT/'REPORT.md').write_text('\n'.join(tables)+'\n')

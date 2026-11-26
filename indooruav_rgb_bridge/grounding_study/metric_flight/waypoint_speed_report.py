"""Matched flight outcomes, waypoint boundary checks and observed camera motion."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run
from metric_flight.experiment import CORRIDORS
from metric_flight.record_rgbt import SHIFTS
from metric_flight.rgbt_report import BASE as ROOM_ARCHIVE, NAMES, combine_videos
from metric_flight.evaluate import load, write, measure
from metric_flight.score_object import score
from metric_flight.audit_rgbt import policy_hashes
from audit_results import audit_inputs

BASE=HERE/'results/waypoint_speed'
GROUPS={'room/repeat':list(SHIFTS),'room/shifted':list(SHIFTS),'corridor_13m/repeat':CORRIDORS}


def paths(group, case):
    folder=BASE/group
    old=ROOM_ARCHIVE/group.split('/')[1]/case if group.startswith('room/') else folder/'legacy'/case
    return old,folder/'through'/case


def median_rows(rows):
    if not rows:
        return {}
    result={key:float(np.median([r[key] for r in rows])) for key in rows[0]}
    result['segments']=len(rows)
    return result


def motion(folder, ep, frames, ds, room):
    model='room65_selected_anchored' if room else 'corridor_native'
    scene=run.SIM/'scenes'/model
    with (folder/'video_frames.jsonl').open() as stream:
        video=[json.loads(s) for s in stream]
    rec=load(folder/'recording.json')
    assert rec['thermal_source']==f'{model}/models/thermal.ply'
    assert rec['frames']==len(video)==len(frames) and rec['fps']==20
    assert len(rec['streams'])==3 and all(v['decoded_frames']==len(frames) for v in rec['streams'].values())
    np.testing.assert_array_equal([r['t'] for r in video],[r['t'] for r in frames])
    assert all(i==r['frame']==r['step'] for i,r in enumerate(video))
    W=np.array([r['rgb_w2c'] for r in video])
    np.testing.assert_allclose([r['thermal_w2c'] for r in video],
                               np.array(load(scene/'inputs/rig.json')['rgb_to_thermal'])@W,atol=1.e-9,rtol=0)
    scene_rotation=np.array(load(scene/'inputs/frame.json')['rotation'])
    flip=np.diag([1.,-1.,-1.])
    mount=Rotation.from_matrix([[.990,0,.140],[0,1,0],[-.140,0,.990]]).as_matrix()
    axes=np.array([[0,0,-1],[1,0,0],[0,-1,0]])
    cam=scene_rotation@W[:,:3,:3].transpose(0,2,1)@flip
    fixed=flip@mount@axes@Rotation.from_euler('x',ep['fixed_pitch_deg'],degrees=True).as_matrix()
    body=cam@fixed.T
    roll,pitch,yaw=Rotation.from_matrix(body).as_euler('xyz',degrees=True).T
    np.testing.assert_allclose((yaw-[r['yaw'] for r in frames]+180)%360-180,0,atol=1.e-7)
    elevation=np.rad2deg(np.arcsin(np.clip(-cam[:,2,2],-1,1)))
    vxy=np.linalg.norm(np.array([r['v'] for r in frames])[:,:2],axis=1)
    rows=defaultdict(list)
    prior=None
    for i,d in enumerate(ds):
        start=round(d['t']*20)
        with Image.open(folder/f'{i:03}.png') as image:
            assert hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()==video[start]['rgb_sha256']
        seconds=.5 if room else 1.
        n=round(seconds*20)
        if ep.get('waypoint_speed')=='through' and 'passing_waypoint' in d:
            vout=np.array(d['reference_velocity_out'])
            if d['passing_waypoint']:
                np.testing.assert_allclose(vout,(np.array(d['world_target'][:3])-frames[start]['xyz'])/seconds,atol=1e-8)
                if prior is not None:np.testing.assert_allclose(prior,d['reference_velocity_in'],atol=1e-8)
                prior=vout
            else:
                np.testing.assert_allclose(vout,0.,atol=1e-8)
                prior=None
        if d['stopping'] or start+n>=len(frames):continue
        if np.linalg.norm(np.array(d['world_target'][:2])-frames[start]['xyz'][:2])<.05:continue
        seg=slice(start,start+n+1)
        phase=d['info']['phase'] if room else 'cruise'
        row=dict(pitch_span_deg=float(np.ptp(pitch[seg])),roll_span_deg=float(np.ptp(roll[seg])),
                 camera_elevation_span_deg=float(np.ptp(elevation[seg])),
                 horizontal_speed_mps=float(np.median(vxy[seg])))
        if room:
            objects=[f['objects']['ac'] for f in frames[seg]]
            if all(o['in_image'] for o in objects):
                row['target_y_span_px']=float(np.ptp([o['uv'][1] for o in objects]))
        rows[phase].append(row)
    return rows


def summarize_rows(rows):
    if not rows:return {}
    keys=set.intersection(*(set(r) for r in rows))
    stats=median_rows([{k:r[k] for k in keys} for r in rows])
    for axis in ['pitch_span_deg','roll_span_deg','camera_elevation_span_deg']:
        stats[axis+'_p95']=float(np.percentile([r[axis] for r in rows],95))
    valid=[r['target_y_span_px'] for r in rows if 'target_y_span_px' in r]
    if valid:
        stats['visible_segments']=len(valid)
        stats['target_y_span_px']=float(np.median(valid))
    return stats


def run_report():
    assert policy_hashes()==load(ROOM_ARCHIVE/'audit.json')['frozen_sources']
    metrics={};pooled=defaultdict(list)
    for group,cases in GROUPS.items():
        room=group.startswith('room/')
        result=metrics[group]={}
        for case in cases:
            old,new=paths(group,case)
            result[case]={}
            prev=load(old/'episode.json');now=load(new/'episode.json')
            for k in ['spawn','spawn_yaw','seed','fixed_pitch_deg','instruction']:
                assert prev[k]==now[k],(case,k)
            assert (old/'000.png').read_bytes()==(new/'000.png').read_bytes()
            old_first=load(old/'decisions.json')[0];new_first=load(new/'decisions.json')[0]
            for k in ['prompt','raw_world','world_target']:
                assert old_first[k]==new_first[k],(case,k)
            for mode,folder,ep in [('legacy',old,prev),('through',new,now)]:
                f=load(folder/'frames.json');ds=load(folder/'decisions.json')
                stats=score(folder) if room else measure(folder,False)
                if not room:assert ep['distance_setpoint']==13.
                rows=motion(folder,ep,f,ds,room)
                stats.update(frames=len(f),duration_s=ep['duration'],reason=ep['reason'],
                             path_m=f[-1]['cumulative_m'],phase={k:summarize_rows(v) for k,v in rows.items()})
                result[case][mode]=stats
                for phase,values in rows.items():
                    pooled[('room' if room else 'corridor',mode,phase)].extend(values)
        combine_videos([paths(group,c)[1]/'rgb_thermal_fpv.mp4' for c in cases],
                       BASE/(group.replace('/','_')+'_all.mp4'),sum(result[c]['through']['frames'] for c in cases))
        if not room:
            combine_videos([paths(group,c)[0]/'rgb_thermal_fpv.mp4' for c in cases],BASE/'corridor_13m_legacy_all.mp4',
                           sum(result[c]['legacy']['frames'] for c in cases))
    overall={'/'.join(k):summarize_rows(v) for k,v in pooled.items()}
    write(BASE/'metrics.json',dict(groups=metrics,pooled_motion=overall,integrity=audit_inputs()))
    write_index(metrics,overall)
    return metrics,overall

def link(folder, name):
    return os.path.relpath(folder/name,BASE)


def write_index(metrics,motion_stats):
    titles={'room/repeat':'会议室原起点','room/shifted':'会议室平移起点','corridor_13m/repeat':'走廊 13 米'}
    names=dict(NAMES)
    names.update(dict(corridor_high_return='原定高返航起点',corridor_climb_accelerate='原爬升起点',
                      new_corridor_high_return='偏航返航起点',new_corridor_climb_accelerate='偏航爬升起点',
                      reverse_corridor_high_return='东段反向起点'))
    lines=['# 中间航点带速度通过：动力学飞行和同步第一视角录像','',
           '会议室复用上一轮 8 个原起点与 8 个平移起点、指令和物体观察策略；走廊按本次要求飞 13.0 米，原来的五个起点各自另飞同距离基线。每对飞行均校验出发位姿、固定云台角、首帧 RGB、首个模型输出和航点一致。所有新回合的模型与检测推理绕过旧结果缓存。','',
           '通过中间航点时，平移参考末速度取当前航点位移除以决策周期。连续运动段把上一段的参考末速度传给下一段的起速度，起点位置仍按实测位置重规划。初次起步采用实测速度。在悬停、丢失目标和减速等待阶段恢复原有的零末速度。走廊旧实现已具有非零末速度，主要差异是下一段起速度的来源。走廊继续按累计三维航程达标结束。偏航边界、控制增益、动力学、固定云台和停机条件沿用原配置。','',
           '左侧为策略使用的 RGB，右侧为各场景独立热红外 3DGS 的伪彩色渲染。每个模态 320 × 240；按同一 20 Hz 动力学时间戳和标定外参成对记录。','',
           '## 视角稳定性','',
           '仅统计水平航点位移至少 5 厘米且完整执行的运动段；会议室每段 0.5 秒，走廊每段 1 秒。角度为单段实际最大值减最小值，再合并各路线取中位数。机体姿态由标定和录像相机外参还原，并逐帧对照动力学偏航。像素统计仅使用目标整个区间均在画面里的分段，来自空调标注中心的投影。','',
           '| 场景与阶段 | 运动段数，旧 / 新 | 俯仰摆幅，旧 → 新 | 横滚摆幅，旧 → 新 | 目标纵向摆幅，旧 → 新 | 水平速度，旧 → 新 |',
           '|---|---:|---:|---:|---:|---:|']
    for scene,phase,title in [('room','approach','会议室接近'),('room','arc','会议室绕行'),('corridor','cruise','走廊巡航')]:
        a,b=[motion_stats['/'.join((scene,m,phase))] for m in ['legacy','through']]
        def pair(key,unit):
            return f'{a[key]:.2f} → {b[key]:.2f} {unit}'
        px=pair('target_y_span_px','px') if scene=='room' else '—'
        lines.append(f"| {title} | {a['segments']} / {b['segments']} | {pair('pitch_span_deg','°')} | {pair('roll_span_deg','°')} | {px} | {pair('horizontal_speed_mps','m/s')} |")
    lines.extend(['','## 逐路线结果和视频','',
                  '会议室沿用严格 / 大致两档标准（稳定悬停距空调 0.4–0.8 米且居中 / 0.5–1.5 米，均须完成指令动作并安全）；表内距离为终点到空调的距离。走廊列出无碰撞且航程达到 13±0.4 米的完成情况，距离为实际累计航程。',''])
    for group,cases in metrics.items():
        room=group.startswith('room/')
        def count(key, mode):
            return sum(bool(v[mode][key]) for v in cases.values())
        flags=['strict_success','approximate_success','collision','outside'] if room else ['strict','collision','outside']
        labels=['严格完成','大致完成','碰撞','越出已知空间'] if room else ['无碰撞且里程达标','碰撞','越出已知空间']
        summary='，'.join(f'{label} {count(key,"legacy")} → {count(key,"through")}' for label,key in zip(labels,flags))
        lines.extend([f'### {titles[group]}','',f'{len(cases)} 条配对路线，旧 → 新：{summary}。','',f"[连续播放新版全部路线]({group.replace('/','_')}_all.mp4)",'',
                      '| 路线 | 旧 → 新完成情况 | 旧 → 新距离（米） | 旧 → 新时间（秒） | 新版终止原因 | 第一视角录像 |',
                      '|---|---|---:|---:|---|---|'])
        for case,both in cases.items():
            old,new=paths(group,case);a=both['legacy'];b=both['through']
            def outcome(r):
                if room:return '严格' if r['strict_success'] else '大致' if r['approximate_success'] else '失败'
                return '达标' if r['strict'] else '失败'
            k='end_distance_m' if room else 'path_m'
            links=' · '.join(f'[{label}]({link(folder,file)})' for label,folder,file in [
                ('新并排',new,'rgb_thermal_fpv.mp4'),('RGB',new,'rgb_fpv.mp4'),('红外',new,'thermal_fpv.mp4'),('旧并排',old,'rgb_thermal_fpv.mp4')])
            lines.append(f"| {names[case]} | {outcome(a)} → {outcome(b)} | {a[k]:.3f} → {b[k]:.3f} | {a['duration_s']:.2f} → {b['duration_s']:.2f} | {b['reason']} | {links} |")
        lines.append('')
    lines.extend(['[13 米走廊旧策略录像合集](corridor_13m_legacy_all.mp4) · [全部逐回合指标、摆幅第 95 百分位及源码校验](metrics.json)',''])
    (BASE/'INDEX.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    run_report()

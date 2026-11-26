"""Check filmed physical flights and publish the two-tier result and video index."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run
from metric_flight.audit_rgbt import policy_hashes
from metric_flight.record_rgbt import SHIFTS, shifted_spawn
from metric_flight.score_object import load, write, score, RESULT_FOLDER
from record_sequence import ffmpeg_executable, verify_video

BASE = HERE/'results/room_ac_rgbt'
OLD = HERE/'results'/RESULT_FOLDER/'visual'
NAMES = {
    'room_inspect_ac':'直接观察', 'new_room_inspect_ac':'直接观察（原偏移）',
    'room_obj_ac_east_to_west':'东向西绕行', 'room_obj_ac_west_to_east':'西向东绕行',
    'holdout_room_obj_ac_east_to_west':'东向西绕行（原偏移）',
    'holdout_room_obj_ac_west_to_east':'西向东绕行（原偏移）',
    'fresh_room_inspect_ac':'直接观察（上一轮新增）',
    'fresh_room_obj_ac_west_to_east':'西向东绕行（上一轮新增）',
}


def audit_case(group, case, calibration):
    p=BASE/group/case
    old=OLD/case
    ep=load(p/'episode.json')
    prev=load(old/'episode.json')
    f=load(p/'frames.json')
    ds=load(p/'decisions.json')
    previous=load(old/'frames.json')
    old_ds=load(old/'decisions.json')
    rec=load(p/'recording.json')
    with (p/'video_frames.jsonl').open() as stream:
        video=[json.loads(row) for row in stream]
    assert ep['seed']==prev['seed'] and ep['instruction']==prev['instruction']
    assert ep['mode']=='flight' and ep['recording_group']==group
    assert len(f)==len(video)==rec['frames']
    assert all(s['decoded_frames']==len(f) for s in rec['streams'].values())
    assert len(rec['streams'])==3 and rec['fps']==20
    np.testing.assert_array_equal([r['t'] for r in video],[r['t'] for r in f])
    assert [r['frame'] for r in video]==list(range(len(f)))
    assert [r['step'] for r in video]==list(range(len(f)))
    np.testing.assert_allclose(np.array([r['thermal_w2c'] for r in video]),
                               calibration @ np.array([r['rgb_w2c'] for r in video]),rtol=0,atol=1e-9)
    assert all(r['rgb_sha256']!=r['thermal_sha256'] for r in video)
    for i,d in enumerate(ds):
        with Image.open(p/f'{i:03}.png') as im:
            digest=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
        row=video[round(d['t']*20)]
        assert abs(row['t']-d['t'])<1e-7 and digest==row['rgb_sha256']==d['detection']['image_sha']
    with Image.open(p/'end.png') as end:
        assert hashlib.sha256(end.tobytes()).hexdigest()==video[-1]['rgb_sha256']
    x0=np.asarray(prev['spawn'])
    yaw=prev['spawn_yaw']
    if group=='shifted':
        x0,yaw=shifted_spawn(x0,yaw,SHIFTS[case])
    np.testing.assert_allclose(x0,ep['spawn'],rtol=0,atol=1e-10)
    assert abs(yaw-ep['spawn_yaw'])<1e-10
    initial=ds[0]['detection']
    visible=f[0]['objects']['ac']['in_image'] and initial['object']=='ac' and bool(initial['boxes'])
    assert visible
    result=score(p)
    write(p/'two_tier_metrics.json',result)
    result.update(video_frames=len(f), decisions=len(ds), initial_ac_visible=bool(visible),
                  initial_detection_score=initial['boxes'][0]['score'],
                  initial_detection_bbox=initial['boxes'][0]['xyxy'],
                  spawn=ep['spawn'],spawn_yaw=ep['spawn_yaw'],duration_s=ep['duration'],
                  displacement_from_previous_spawn_m=float(np.linalg.norm(np.subtract(ep['spawn'],prev['spawn']))))
    if group=='repeat':
        n=min(len(f),len(previous))
        delta=np.array([r['xyz'] for r in f[:n]])-np.array([r['xyz'] for r in previous[:n]])
        dyaw=np.array([a['yaw']-b['yaw'] for a,b in zip(f,previous)])
        result.update(same_time_grid=[r['t'] for r in f]==[r['t'] for r in previous],
                      max_xyz_difference_m=float(np.linalg.norm(delta,axis=1).max()),
                      max_yaw_difference_deg=float(abs((dyaw+180)%360-180).max()),
                      identical_raw_outputs=[d['raw_world'] for d in ds]==[d['raw_world'] for d in old_ds],
                      identical_issued_waypoints=[d['world_target'] for d in ds]==[d['world_target'] for d in old_ds],
                      identical_stop_reason=prev['reason']==ep['reason'])
    return result


def combine_videos(files, output, total):
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        manifest=Path(tmp)/'clips.txt'
        manifest.write_text(''.join(f"file '{path}'\n" for path in files))
        video=Path(tmp)/'all.mp4'
        subprocess.run([ffmpeg_executable(),'-v','error','-f','concat','-safe','0','-i',str(manifest),
                        '-c','copy','-movflags','+faststart',str(video)],check=True)
        verify_video(video,total)
        video.replace(output)


def combined(group, total):
    files=[BASE/group/case/'rgb_thermal_fpv.mp4' for case in SHIFTS]
    combine_videos(files,BASE/f'{group}_all.mp4',total)


def publish():
    assert load(BASE/'audit.json')['frozen_sources']==policy_hashes()
    rig=np.array(load(run.SIM/'scenes/room65_selected_anchored/inputs/rig.json')['rgb_to_thermal'])
    metrics={};lines=['# 会议室空调任务动力学复飞与双模态录像','',
        '视频按 20 帧/秒记录机体运动，每个模态 320 × 240 像素。并排视频左侧为 RGB，右侧为会议室热红外 3DGS 伪彩色。红外使用同一控制周期的机体与云台位姿和标定外参。策略只接收 RGB。','',
        '距离采用机体中心到空调标注中心的三维欧氏距离，边界计入。严格完成要求整个两秒稳定悬停均在 0.4–0.8 米，空调中心在画面两轴中央 40%，并有 RGB 观测确认。大致完成要求整个稳定悬停均在 0.5–1.5 米。两档均要求完成指令动作、无碰撞且全程处于已知空间。逐回合 JSON 也列出了仅按终点距离评分的结果。','',
        '主动停止沿用原指令和固定参数。模型收到的指令仍要求在 0.5–1.5 米处居中悬停。估计距离落在 0.60–1.40 米且目标连续居中时，视觉控制器可提前停止。','',
        '根据调用路径，专家数据用于指令和任务元数据、首帧初态与离线评分。首帧提供位置、偏航、速度和固定云台俯仰。目标标注坐标在飞行中只写投影日志。闭环中没有读取专家后续位姿或动作。模型和检测器直接推理，未使用旧推理缓存。控制器使用约 0.80 米的空调宽度先验、相机内外参、检测框及在线多帧视线；三角测量只拟合本次观测。仓库内由专家样本训练的云台核回归模块在本次执行路径中未加载。实验沿用已在该会议室调试过的手工视觉控制规则。','',
        '[飞行输入与记录](../../experiment.py) · [冻结模型入口](../../../../model_worker.py) · [视觉航点计算](../../control.py) · [停止判定](../../object_goal.py) · [单独的云台训练程序](../../../../gimbal/train.py)','',
        '[权重与源文件校验](audit.json) · [汇总数据](metrics.json) · [新起点首帧及检测框](shifted_starts.png)','']
    sheet=Image.new('RGB',(4*320,2*265),'white')
    for group,title in [('repeat','原起点复飞'),('shifted','更换起点')]:
        results={case:audit_case(group,case,rig) for case in SHIFTS}
        metrics[group]=results
        n=sum(r['video_frames'] for r in results.values())
        combined(group,n)
        lines.extend([f'## {title}','',f'严格完成 {sum(r["strict_success"] for r in results.values())}/8，大致完成 {sum(r["approximate_success"] for r in results.values())}/8。[连续播放本组八条飞行]({group}_all.mp4)','',
                      '| 任务 | 合集开始时间 | 终点距离（米） | 严格 / 大致 | 终止原因 | 视频 |',
                      '|---|---:|---:|---|---|---|'])
        offset=0
        for i,(case,r) in enumerate(results.items()):
            links=' · '.join(f'[{label}]({group}/{case}/{stem}_fpv.mp4)' for label,stem in [('并排','rgb_thermal'),('RGB','rgb'),('红外','thermal')])
            stamp=f'{int(offset)//60:02}:{offset%60:05.2f}'
            flags=' / '.join('通过' if r[k] else '未通过' for k in ['strict_success','approximate_success'])
            lines.append(f'| {NAMES[case]} | {stamp} | {r["end_distance_m"]:.3f} | {flags} | {r["reason"]} | {links} |')
            offset+=r['video_frames']/20
            if group=='shifted':
                with Image.open(BASE/group/case/'000.png') as im:
                    tile=im.copy()
                box=np.array(r['initial_detection_bbox'])*[320,240,320,240]
                ImageDraw.Draw(tile).rectangle(tuple(box),outline='lime',width=2)
                x=i%4*320; y=i//4*265
                sheet.paste(tile,(x,y+25))
                ImageDraw.Draw(sheet).text((x+4,y+5),case,fill='black')
        lines.append('')
    lines.extend(['## 复刻精度与飞行结果',''])
    repeat=list(metrics['repeat'].values())
    exact=sum(r['same_time_grid'] and r['max_xyz_difference_m']==0 and r['max_yaw_difference_deg']==0
              and r['identical_raw_outputs'] and r['identical_issued_waypoints'] and r['identical_stop_reason'] for r in repeat)
    lines.append(f'逐帧对照共 {sum(r["video_frames"] for r in repeat):,} 帧。{exact}/8 条的时间序列、位置、偏航、原始模型输出、下发航点和终止原因均完全一致。')
    lines.append('')
    for group,title in [('repeat','原起点'),('shifted','新起点')]:
        rs=list(metrics[group].values())
        end=[r['end_distance_m'] for r in rs]
        lines.append(f'{title}终点距离为 {min(end):.3f}–{max(end):.3f} 米。首帧可见 {sum(r["initial_ac_visible"] for r in rs)}/8，停机及悬停 RGB 采样居中 {sum(r["hold_rgb_centered"] for r in rs)}/8，碰撞回合 {sum(r["collision"] for r in rs)}，越界回合 {sum(r["outside"] for r in rs)}。')
        lines.append('')
    example=metrics['shifted']['new_room_inspect_ac']
    if example['reason']=='target_lost':
        ds=load(BASE/'shifted/new_room_inspect_ac/decisions.json')
        tracked=next((d for d in ds if d['info'].get('tracking')=='tracked_current_rgb' and d['detection']['boxes']
                      and abs(d['info']['cx']-d['detection']['boxes'][0]['x'])>.25),None)
        if tracked:
            lines.extend([f'直接观察的一个新起点在首帧检测到空调。第 {tracked["n"]} 次决策时，检测器的框中心在画面宽度 {tracked["detection"]["boxes"][0]["x"]:.1%} 处，控制器采用的光流框中心却在 {tracked["info"]["cx"]:.1%} 处。随后继续转向，在距空调 {example["end_distance_m"]:.3f} 米处以 target_lost 终止。[失败回合的并排视频](shifted/new_room_inspect_ac/rgb_thermal_fpv.mp4) · [逐次决策](shifted/new_room_inspect_ac/decisions.json)',''])
    sheet.save(BASE/'shifted_starts.png')
    write(BASE/'metrics.json',metrics)
    (BASE/'INDEX.md').write_text('\n'.join(lines)+'\n')
    return metrics

if __name__=='__main__':
    publish()

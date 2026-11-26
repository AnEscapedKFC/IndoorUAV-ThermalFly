"""Score actual receipts, independent end-frame detections and the four final hover frames."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent), str(HERE.parents[1])]
from detector import Detector
from metric_flight.view_control import PITCH_LIMITS, framing
from metric_flight.observation_memory import write_json


def read(path):
    return json.loads(path.read_text())


def audit(folder, detector):
    ep=read(folder/"episode.json")
    frames=read(folder/"frames.json")
    decisions=read(folder/"decisions.json")
    obs=read(folder/"observations/index.json")
    survey=read(folder/"depth_preflight.json")
    video=read(folder/"recording.json")
    assert len(frames)==video["frames"] and video["rgb_wh"]==video["thermal_wh"]==[640,480]
    assert all(stream["passed"] and stream["decoded_frames"]==len(frames) for stream in video["streams"].values())
    receipts=[json.loads(row) for row in (folder/"video_frames.jsonl").read_text().splitlines()]
    assert len(receipts)==len(frames) and len(decisions)==len(obs["frames"])
    for i, name in ((0,"start"),(-1,"end")):
        rgb=np.asarray(Image.open(folder/(name+".png")))
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==receipts[i]["rgb_sha256"]
    for o,d in zip(obs["frames"],decisions):
        idx=o["receipt_frame"]-receipts[0]["receipt_frame"]
        assert abs(o["time_s"]-d["t"])<1e-6 and np.allclose(o["rgb_w2c_native"],receipts[idx]["rgb_w2c"])
    pitch=[f["rig_pitch_deg"] for f in frames]
    survey_angles=[s["pitch_deg"] for s in survey["samples"]+survey["rechecks"]]
    assert min(pitch+survey_angles)>=PITCH_LIMITS[0] and max(pitch+survey_angles)<=PITCH_LIMITS[1]
    # Case id is only for grouping. Independent end/start RGB use the target recorded at runtime.
    endpoints=[]
    for name in ("start","end"):
        boxes=detector(folder/(name+".png"),obs["target"])["boxes"]
        endpoints.append(framing(boxes[0]["xyxy"] if boxes else None))
    end=endpoints[1]
    hold=ep["hold"]
    held=bool(ep["reason"]=="stop" and len(hold)==41 and
              max(np.linalg.norm(np.subtract(x["xyz"],hold[0]["xyz"])) for x in hold)<=.12 and
              max(x["speed"] for x in hold[-10:])<=.10)
    rows=ep["hold_rgb"]
    stable_view=(len(rows)==4 and all(r["framing_ok"] and r["view_ok"] and r["parts_ok"] for r in rows))
    blocked=survey["after"]["status"]!="free"
    if blocked:
        assert not decisions and len(frames)==1 and ep["reason"]=="start_clearance_unconfirmed"
    collisions=sum(f["collision"] for f in frames)
    outside=sum(not f["inside"] for f in frames)
    sampled=[d["info"] for d in decisions if d["info"].get("los_deg") is not None]
    los=rows[-1]["los_deg"] if rows else (sampled[-1]["los_deg"] if sampled else None)
    z=[r["xyz"][2] for r in frames]
    return dict(case=folder.name,strict=bool(held and stable_view and not blocked and not collisions and not outside),
                reason=ep["reason"],spawn=ep["spawn"],duration_s=ep["duration"],path_m=frames[-1]["cumulative_m"],
                instruction=ep["instruction"],decisions=len(decisions),video_frames=len(frames),collision_samples=collisions,
                outside_samples=outside,precheck=survey["before"],postcheck=survey["after"],rescan_frames=len(survey["rechecks"]),
                start_bbox_area=endpoints[0]["area"],end_bbox_area=end["area"],end_centered=end["centered"],
                end_box_clipped=end["clipped"],start_z=z[0],end_z=z[-1],z_min=min(z),z_max=max(z),
                end_gimbal=pitch[-1],gimbal_range=[min(pitch),max(pitch)],end_optical_elevation=frames[-1]["optical_elevation_deg"],
                last_measured_los=los,los_time_s=rows[-1]["t"] if rows else next((r["t"] for r in reversed(decisions)
                  if r["info"].get("los_deg") is not None),None),held=held,hold_frames=rows,
                guard_last=decisions[-1]["info"].get("depth_guard") if decisions else None)


def hover_sheet(folders, out):
    held=[folder for folder in folders if list(folder.glob("hold_*.png"))]
    if not held:
        return
    sheet=Image.new("RGB",(4*256,len(held)*218),"white")
    draw=ImageDraw.Draw(sheet)
    for row,folder in enumerate(held):
        metrics=read(folder/"episode.json")["hold_rgb"]
        for col,(path,r) in enumerate(zip(sorted(folder.glob("hold_*.png")),metrics)):
            x,y=col*256,row*218
            sheet.paste(Image.open(path).resize((256,192),Image.Resampling.LANCZOS),(x,y))
            area="n/a" if r["area"] is None else f'{r["area"]:.1%}'
            draw.text((x+3,y+197),f'{folder.name}  {r["t"]:.1f}s  {area}',fill="black")
    sheet.save(out/"hover_views.jpg",quality=92)


def plot_telemetry(root, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    examples={"level_diagonal":"AC horizontal", "outlet_diagonal":"AC outlet from below", "desk_east":"Laptop from above"}
    fig, axes=plt.subplots(3,1,sharex=True,figsize=(8,7),layout="constrained")
    for name,label in examples.items():
        folder=root/name
        if not (folder/"episode.json").exists():
            continue
        frames=read(folder/"frames.json")
        decisions=read(folder/"decisions.json")
        hold=read(folder/"episode.json")["hold_rgb"]
        t=[f["t"] for f in frames]
        axes[0].plot(t,[f["xyz"][2] for f in frames],label=label)
        axes[1].plot(t,[f["rig_pitch_deg"] for f in frames])
        times=[r["t"] for r in decisions]+[r["t"] for r in hold]
        areas=[r["info"]["area"] for r in decisions]+[r["area"] for r in hold]
        axes[2].plot(times,[np.nan if v is None else v*100 for v in areas],".-",markersize=3)
    axes[0].legend(fontsize=9)
    axes[0].set_ylabel("Body height (m)")
    axes[1].set_ylabel("Mechanical pitch (deg)")
    axes[2].set_ylabel("RGB box area (%)")
    axes[2].set_xlabel("Flight time after launch survey (s)")
    axes[2].axhspan(20,60,color="green",alpha=.10)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(out/"flight_profiles.png",dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=HERE/"results/coordinated_inspection")
    parser.add_argument("--out",type=Path,default=HERE/"coordinated_results")
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    from metric_flight.furniture import deny_expert_reads
    deny_expert_reads()
    detector=Detector(use_cache=False, device=args.device)
    args.out.mkdir(parents=True,exist_ok=True)
    results=[]
    folders=sorted(p.parent for p in args.root.glob("*/episode.json") if not p.parent.name.startswith("."))
    for folder in folders:
        result=audit(folder,detector)
        results.append(result)
        print(folder.name,result["strict"],result["reason"],result["end_bbox_area"],flush=True)
    write_json(args.out/"results.json",results)
    if folders:
        thumb=(320,240)
        sheet=Image.new("RGB",(3*thumb[0],((len(folders)+2)//3)*(thumb[1]+32)),"white")
        draw=ImageDraw.Draw(sheet)
        for i,(folder,r) in enumerate(zip(folders,results)):
            x=thumb[0]*(i%3)
            y=(thumb[1]+32)*(i//3)
            sheet.paste(Image.open(folder/"end.png").resize(thumb,Image.Resampling.LANCZOS),(x,y))
            area=r["end_bbox_area"]
            caption=f"{folder.name}  z={r['end_z']:.2f}m  area="+(f"{area:.1%}" if area is not None else "n/a")
            draw.text((x+4,y+thumb[1]+8),caption,fill="black")
        sheet.save(args.out/"end_views.jpg",quality=92)
        hover_sheet(folders,args.out)
        plot_telemetry(args.root,args.out)

if __name__=="__main__":
    main()

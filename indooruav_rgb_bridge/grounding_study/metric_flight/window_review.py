"""Reproduce observation and autonomous scores from fresh receipt images and review annotations."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parents[1])]
from metric_flight.instance_mask import instance_mask
from metric_flight.observation_window import INSPECTION_AREA, hold_seconds, window_verdict
from metric_flight.view_control import VIEW_BANDS, elevation, framing, view_intent
from metric_flight.observation_memory import write_json


def read(path):
    return json.loads(path.read_text())


def manual_view(folder, step, annotation, view, area_range=(.20,.60)):
    path=folder/"review_frames"/f"{step:04}.png"
    rgb=np.asarray(Image.open(path))
    h,w=rgb.shape[:2]
    box=np.asarray(annotation["box"],float)/[w,h,w,h]
    metric=framing(box,area_range)
    with np.load(path.with_suffix(".npz")) as calibration:
        depth=calibration["depth_mm"]*.001
        if "part_roi" in annotation:
            l,t,r,b=annotation["part_roi"]
            mask=np.zeros((h,w),bool)
            mask[t:b,l:r]=True
        else:
            mask=instance_mask()(rgb,box)
            # SAM may extend outside the annotated visible extent.
            l,t,r,b=annotation["box"]
            bounded=np.zeros((h,w),bool)
            bounded[t:b,l:r]=mask[t:b,l:r]
            mask=bounded
        rows,cols=np.nonzero(mask & (depth>.08))
        los=None
        if len(rows)>=32:
            optical=(np.c_[cols,rows,np.ones(len(rows))]@np.linalg.inv(calibration["K"]).T)*depth[rows,cols,None]
            delta=np.median(optical@calibration["axes"].T,axis=0)
            los=elevation(delta)
    band=VIEW_BANDS.get(view)
    ok_angle=los is not None and (band is None or band[0]<=los<=band[1])
    return dict(**metric,step=step,los_deg=los,view_ok=ok_angle,parts_ok=annotation["parts_visible"],
                box_px=annotation["box"],valid=bool(metric["framing_ok"] and ok_angle and annotation["parts_visible"]))


def hold_images(folder, attempt_number, attempt):
    paths=[folder/f"{attempt['decision']:03}.png"]
    paths.extend(folder/f"attempt_{attempt_number:02}_{10*n:03}.png"
                 for n in range(1,len(attempt['checks'])+1))
    return paths


def audit(folder, review):
    ep=read(folder/"episode.json")
    frames=read(folder/"frames.json")
    decisions=read(folder/"decisions.json")
    attempts=read(folder/"hold_attempts.json")
    survey=read(folder/"depth_preflight.json")
    rec=read(folder/"recording.json")
    receipts=[json.loads(line) for line in (folder/"video_frames.jsonl").read_text().splitlines()]
    assert len(receipts)==len(frames)==rec["frames"]
    assert rec["rgb_wh"]==rec["thermal_wh"]==[640,480] and all(v["passed"] and v["decoded_frames"]==len(frames) for v in rec["streams"].values())
    for i,f in enumerate(frames):
        assert abs(receipts[i]["t"]-f["t"])<1e-6 and -60<=f["rig_pitch_deg"]<=30
        if i%10==0:
            png=folder/"review_frames"/f"{i:04}.png"
            assert hashlib.sha256(np.asarray(Image.open(png)).tobytes()).hexdigest()==receipts[i]["rgb_sha256"]
    collision=sum(f["collision"] for f in frames)
    outside=sum(not f["inside"] for f in frames)
    safe=not collision and not outside and survey["after"]["status"]=="free"
    safety_path=folder/"safety_events.json"
    safety=read(safety_path) if safety_path.is_file() else None
    if safety is not None:
        assert safety["collision"]==int(bool(collision))
        assert safety["boundary"]==int(ep["reason"]=="out_of_bounds")
        if outside:
            assert safety["coverage"] or safety["collision"] or safety["boundary"]
        safe=not collision and not safety["boundary"] and survey["after"]["status"]=="free" and ep["reason"]!="map_coverage_unresolved"
    seconds=hold_seconds(ep["instruction"])
    confirmed=[a for a in attempts if a["confirmed"]]
    for attempt_number,a in enumerate(attempts,1):
        if not a["confirmed"]:
            continue
        start=round(a["t"]*20)
        for j,path in enumerate(hold_images(folder,attempt_number,a)):
            pixels=np.asarray(Image.open(path))
            assert hashlib.sha256(pixels.tobytes()).hexdigest()==receipts[start+10*j]["rgb_sha256"]
        end=start+10*len(a["checks"])
        positions=[np.asarray(f["xyz"]) for f in frames[start:end+1]]
        assert len(positions)==end-start+1
        measured_drift=max(np.linalg.norm(pos-positions[0]) for pos in positions)
        assert np.isclose(a["max_drift_m"],measured_drift,atol=1e-8)
        assert np.allclose([r["t"] for r in a["checks"]],a["t"]+.5*np.arange(1,len(a["checks"])+1))
        assert a["qualified"] and measured_drift<=.12
        if safety is None:
            assert len(a["checks"])==round(2*seconds)
            assert all(r["view_ok"] and r["parts_ok"] and r["framing_ok"] for r in a["checks"])
        else:
            assert seconds<=.5*len(a["checks"])<=seconds+1
            checked=window_verdict([a["first"],*a["checks"]],seconds,measured_drift)
            assert checked["confirmed"] and np.isclose(a["median_area"],checked["median_area"])
    stop_signal=bool(safe and ep["reason"]=="stop" and confirmed)
    observed=False
    checks=[]
    drift=None
    if review["method"]=="online_frames":
        assert confirmed
        observed=bool(stop_signal and review["identity_ok"] and review["visually_verified"])
    elif review["method"]=="offline_boxes":
        start=review["start_step"]
        assert len(review["samples"])==1+round(2*seconds)
        stop=start+round(20*seconds)
        assert stop<len(frames)
        drift=max(np.linalg.norm(np.subtract(f["xyz"],frames[start]["xyz"])) for f in frames[start:stop+1])
        checks=[manual_view(folder,start+10*j,row,view_intent(ep["instruction"]),
                            INSPECTION_AREA if safety is not None else (.20,.60)) for j,row in enumerate(review["samples"])]
        valid=all(r["valid"] for r in checks)
        if safety is not None:
            samples=[dict(c,t=(start+10*j)/20,source="detector" if c["area"] is not None else "missing") for j,c in enumerate(checks)]
            valid=window_verdict(samples,seconds,drift)["confirmed"]
        observed=bool(safe and review["identity_ok"] and drift<=.12 and valid)
    elif review["method"]!="no_eligible_view":
        raise ValueError(review["method"])
    return dict(case=folder.name,observation=observed,autonomous=bool(stop_signal and observed),controller_stop=stop_signal,reason=ep["reason"],
                visual_method=review["method"],comment=review["comment"],frames=len(frames),
                path_m=frames[-1]["cumulative_m"],duration_s=ep["duration"],collision_samples=collision,
                map_coverage_events=len(safety["coverage"]) if safety is not None else None,
                boundary_events=safety["boundary"] if safety is not None else None,
                collision_events=int(bool(collision)),outside_samples=outside,start_check=survey["after"]["status"],attempt_count=len(attempts),
                confirmations=[dict(t=a["t"],max_drift_m=a["max_drift_m"],checks=a["checks"]) for a in confirmed],
                manual_drift_m=drift,manual_samples=checks,
                first_association=(decisions[0]["detection"].get("association") if decisions else None))


def evidence_sheets(folders, annotations, rows, out):
    selected=[(folder, row) for folder,row in zip(folders,rows)
              if row["observation"] and (row["case"].startswith("chair") or
                 row["visual_method"]=="offline_boxes" or row["case"] in {"overview_west","outlet_new","desk_east"})]
    if selected:
        sheet=Image.new("RGB",(3*270,len(selected)*230),"white")
        draw=ImageDraw.Draw(sheet)
        for j,(folder,row) in enumerate(selected):
            note=annotations[folder.name]
            if note["method"]=="offline_boxes":
                start=note["start_step"]
                boxes=[r["box"] for r in note["samples"]]
                paths=[folder/"review_frames"/f"{start+10*j:04}.png" for j in range(len(boxes))]
            else:
                number,a=next((n,a) for n,a in enumerate(read(folder/"hold_attempts.json"),1) if a["confirmed"])
                paths=hold_images(folder,number,a)
                start=round(20*a["t"])
                decision=read(folder/"decisions.json")[a["decision"]]
                boxes=[decision["detection"]["boxes"][0]["xyxy"],*[c["xyxy"] for c in a["checks"]]]
                boxes=[(np.asarray(b)*[640,480,640,480]).tolist() if b is not None else None for b in boxes]
            for k,b in enumerate(boxes[:3]):
                step=start+10*k
                im=Image.open(paths[k]).convert("RGB")
                if b is not None:ImageDraw.Draw(im).rectangle(b,outline="yellow" if note["method"]=="offline_boxes" else "#43e798",width=2)
                x=k*270;y=j*230
                sheet.paste(im.resize((270,202),Image.Resampling.LANCZOS),(x,y))
                draw.text((x+4,y+206),f"{folder.name}  {step/20:.1f}s",fill="black")
        sheet.save(out/"verified_windows.jpg",quality=92)
    chairs=[folder for folder in folders if folder.name.startswith("chair")]
    if chairs:
        sheet=Image.new("RGB",(2*400,((len(chairs)+1)//2)*330),"white")
        draw=ImageDraw.Draw(sheet)
        for j,folder in enumerate(chairs):
            im=Image.open(folder/"start.png").convert("RGB")
            decisions=read(folder/"decisions.json")
            boxes=decisions[0]["detection"]["boxes"] if decisions else []
            if boxes:
                b=(np.asarray(boxes[0]["xyxy"])*[640,480,640,480]).tolist()
                if b is not None:ImageDraw.Draw(im).rectangle(b,outline="#43e798",width=3)
            x=(j%2)*400;y=(j//2)*330
            sheet.paste(im.resize((400,300),Image.Resampling.LANCZOS),(x,y))
            status=("matched" if boxes else "unmatched") if decisions else "not run (launch blocked)"
            draw.text((x+5,y+307),f"{folder.name}  first-frame association: {status}",fill="black")
        sheet.save(out/"door_chair_starts.jpg",quality=93)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=HERE/"results/observation_window")
    parser.add_argument("--out",type=Path,default=HERE/"window_results")
    args=parser.parse_args()
    from metric_flight.furniture import deny_expert_reads
    deny_expert_reads()
    annotations=read(args.out/"annotations.json")
    folders=sorted(p.parent for p in args.root.glob("*/episode.json"))
    assert {f.name for f in folders}==set(annotations)
    rows=[audit(folder,annotations[folder.name]) for folder in folders]
    write_json(args.out/"results.json",rows)
    sheet=Image.new("RGB",(4*270,((len(folders)+3)//4)*230),"white")
    draw=ImageDraw.Draw(sheet)
    for i,(folder,row) in enumerate(zip(folders,rows)):
        x=i%4*270;y=i//4*230
        sheet.paste(Image.open(folder/"end.png").resize((270,202),Image.Resampling.LANCZOS),(x,y))
        draw.text((x+4,y+207),f"{folder.name}   view={int(row['observation'])} auto={int(row['autonomous'])}",fill="black")
    sheet.save(args.out/"end_views.jpg",quality=92)
    evidence_sheets(folders,annotations,rows,args.out)
    print(len(rows),sum(r['observation'] for r in rows),sum(r['autonomous'] for r in rows))

if __name__=="__main__":
    main()

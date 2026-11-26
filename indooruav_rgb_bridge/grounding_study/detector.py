"""Frozen text-conditioned detection; uses RGB pixels and a noun phrase only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

ROOT = Path(__file__).resolve().parent
NOUN = {'ac':'air conditioner', 'laptop':'laptop', 'sofa':'sofa'}
class Detector:
    def __init__(self, *, use_cache=True):
        self.use_cache=use_cache
        weights = ROOT/'assets'
        self.proc = AutoProcessor.from_pretrained(weights, local_files_only=True)
        self.net = AutoModelForZeroShotObjectDetection.from_pretrained(weights,local_files_only=True).eval()
        self.cache = ROOT/'results/detections'
        self.cache.mkdir(exist_ok=True)
    @torch.inference_mode()
    def __call__(self, rgb_path, obj):
        im = Image.open(rgb_path).convert('RGB')
        digest = hashlib.sha256(im.tobytes()).hexdigest()
        file = self.cache / f'{digest}_{obj}.json'
        if self.use_cache and file.exists():
            return json.loads(file.read_text())
        inp = self.proc(images=im,text=NOUN[obj]+'.',return_tensors='pt',size={'shortest_edge':640,'longest_edge':960})
        out = self.net(**inp)
        pred = self.proc.post_process_grounded_object_detection(out,inp.input_ids,box_threshold=.35,text_threshold=.25,target_sizes=[(im.height,im.width)])[0]
        boxes=[]
        for box,score,label in zip(pred['boxes'],pred['scores'],pred['labels']):
            if not label.strip():continue
            x0,y0,x1,y1=np.clip(box.numpy()/np.array([im.width,im.height,im.width,im.height]),0,1)
            x=float((x0+x1)/2)
            boxes.append(dict(score=float(score),label=label,xyxy=[x0,y0,x1,y1],x=x,area=float((x1-x0)*(y1-y0)),side='left' if x<.45 else 'right' if x>.55 else 'center'))
        boxes.sort(key=lambda b:-b['score'])
        result=dict(object=obj,side=boxes[0]['side'] if boxes else None,boxes=boxes,image_sha=digest)
        if self.use_cache:file.write_text(json.dumps(result))
        return result

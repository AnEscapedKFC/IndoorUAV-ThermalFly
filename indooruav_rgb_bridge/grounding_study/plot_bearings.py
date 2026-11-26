import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import Image
p=Path(__file__).resolve().parent/'results'
font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False})
fig,axs=plt.subplots(2,2,figsize=(8.6,6.5),layout='constrained')
angles=json.loads((p/'angles.json').read_text())
for j,(name,obj,noun) in enumerate([('laptop_left_sofa_right','sofa','沙发'),('ac_left','ac','空调')]):
    for i,dy in enumerate((0,180)):
        path=p/'azimuth_views'/f'{name}_{dy}.png';im=Image.open(path)
        key=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
        detection=json.loads((p/'detections'/f'{key}_{obj}.json').read_text())
        v=next(v for v in angles if v['arm']=='rgb_memory' and v['seed']==42 and v['dy']==dy and v['view']==name and v['obj']==obj)
        ax=axs[j,i];ax.imshow(im)
        if detection['boxes']:
            x0,y0,x1,y1=detection['boxes'][0]['xyxy']
            ax.add_patch(Rectangle((x0*im.width,y0*im.height),(x1-x0)*im.width,(y1-y0)*im.height,fill=False,lw=2,ec='#ffd449'))
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(f'{noun} · '+('转身前' if not dy else '原地转身 180° 后'))
        source='当前帧' if v['source']=='current' else '先前观测'
        ax.set_xlabel(f"真值 {v['truth']:+.0f}°    估计 {v['estimate']:+.0f}°（{source}）")
fig.suptitle('同一位置，机体转向如何改变目标方位',fontsize=16)
fig.text(.5,-.025,'0° 为机头正前，正值为右，±180° 为正后。黄框为检测器实际选择的框。',ha='center')
fig.savefig(p/'front_rear_examples.png',dpi=155,bbox_inches='tight',pad_inches=.15)

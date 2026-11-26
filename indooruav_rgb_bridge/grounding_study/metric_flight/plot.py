from evaluate import OUT,STUDY,DATA,load
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':9,'axes.unicode_minus':False})
fig,axs=plt.subplots(1,3,figsize=(11,3.65),layout='constrained')
for ax,n,label in zip(axs[:2],['room_obj_ac_east_to_west','room_obj_ac_west_to_east'],['会议室 · 东向西弧线','会议室 · 西向东弧线']):
 for key,color,name in [('hint','#b66a27','仅提示'),('visual','#087f82','视觉观察')]:
  folder=OUT/'room'/key/n
  if not (folder/'frames.json').exists():folder=STUDY/'results/runs/flight/hint'/n
  q=np.array([v['xyz'] for v in load(folder/'frames.json')]);ax.plot(q[:,0],q[:,1],color=color,lw=1.6,label=name);ax.scatter(*q[-1,:2],s=30,color=color);ax.scatter(*q[0,:2],marker='s',facecolor='white',edgecolor='k',s=26,zorder=3)
 goal=load(DATA/'episodes'/n/'episode.json')['task_spec']['goal_w_m']
 ax.scatter(*goal[:2],marker='*',s=90,color='#202b58',label='评分终点',zorder=5)
 ax.set(xlabel='场景 x（米）',ylabel='场景 y（米）',title=label,aspect='equal');ax.grid(alpha=.2)
axs[0].legend(loc='best',fontsize=8)
ax=axs[2];n='new_corridor_high_return'
for key,color,title in [('off','#b66a27','检查关：累计里程'),('auto','#087f82','自动：累计里程')]:
 f=load(OUT/'corridor'/key/n/'frames.json')
 ax.plot([v['t'] for v in f],[v['cumulative_m'] for v in f],color=color,label=title)
 if key=='auto':ax.plot([v['t'] for v in f],[v['forward_m'] for v in f],'--',color='#294b74',label='自动：初始前向净进度')
ax.axhline(13.5,ls=':',color='k',lw=1);ax.set(title='走廊 · 偏航返航起点',xlabel='动力学时间（秒）',ylabel='距离（米）');ax.grid(alpha=.2);ax.legend(loc='best',fontsize=8)
fig.savefig(OUT/'physical_comparison.png',dpi=160,bbox_inches='tight',pad_inches=.12)

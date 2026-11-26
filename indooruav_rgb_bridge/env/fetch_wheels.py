import hashlib, os, threading, time, concurrent.futures as ft
from pathlib import Path
import requests
from packaging.tags import cpython_tags, compatible_tags
from packaging.utils import parse_wheel_filename
root=Path(__file__).resolve().parent
dest=root/'wheels'
dest.mkdir(exist_ok=True)
tags=list(cpython_tags((3,12)))+list(compatible_tags((3,12),"cp312"))
ranks={t:i for i,t in enumerate(tags)}
jobs=[ln.split('==') for ln in (root/'runtime.lock').read_text().splitlines() if '==' in ln and not ln.lstrip().startswith('#')]
chunk_len=1024*1024
local=threading.local()
def req(url,**kwargs):
 if not hasattr(local,'s'):
  local.s=requests.Session();local.s.trust_env=False
 for t in range(5):
  try:
   r=local.s.get(url,timeout=(8,30),**kwargs);r.raise_for_status();return r
  except requests.RequestException:
   if t==4:raise
def get_file(nv):
 n,v=nv
 files=req(f'https://pypi.org/pypi/{n}/{v}/json').json()['urls']
 wheels=[]
 for f in files:
  if not f['filename'].endswith('.whl'):continue
  matches=parse_wheel_filename(f['filename'])[3] & ranks.keys()
  if matches:wheels.append((min(ranks[t] for t in matches),f))
 if not wheels:raise ValueError(nv)
 return min(wheels,key=lambda h:h[0])[1]
def part(job):
 f,start=job; output=dest/f['filename']; end=min(start+chunk_len,f['size'])-1
 url=f['url']; resp=req(url,headers={'Range':f'bytes={start}-{end}'})
 if len(resp.content)!=end-start+1:raise ValueError((output.name,start,end,len(resp.content)))
 with output.with_suffix('.part').open('r+b',buffering=0) as h:
  os.pwrite(h.fileno(),resp.content,start)
with ft.ThreadPoolExecutor(max_workers=24) as ex:
 files=list(ex.map(get_file,jobs))
print('resolved',len(files),sum(f['size'] for f in files)//1024**2,flush=True)
queue=[]
for f in sorted(files,key=lambda a:a['size'],reverse=True):
 output=dest/f['filename']
 if output.exists():continue
 with output.with_suffix('.part').open('wb') as h:h.truncate(f['size'])
 for start in range(0,f['size'],chunk_len):queue.append((f,start))
with ft.ThreadPoolExecutor(max_workers=48) as ex:
 futs=[ex.submit(part,item) for item in queue]
 for i,future in enumerate(ft.as_completed(futs),1):
  future.result()
  if i%40==0:print('parts',i,len(queue),flush=True)
for f in files:
 output=dest/f['filename']
 if output.exists():continue
 p=output.with_suffix('.part')
 with p.open('rb') as h:digest=hashlib.file_digest(h,'sha256').hexdigest()
 if digest != f['digests']['sha256']:raise ValueError((f['filename'],'sha256'))
 p.rename(output)
print('verified all',flush=True)

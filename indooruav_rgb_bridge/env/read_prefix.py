"""Read first archive members while later training-state members finish downloading."""
import io
from pathlib import Path
import shutil
import time
import zipfile
import requests

root=Path(__file__).resolve().parent.parent
target=root/'env/early-checkpoint'
target.mkdir(exist_ok=True)
size=9468831797
length=32*1024*1024
tail_start=size-65536
s=requests.Session()
s.trust_env=False
r=s.get('https://modelscope.cn/models/valyentine/IndoorUAV-Agent/resolve/master/29999.zip',headers={'Range':f'bytes={tail_start}-{size-1}'},timeout=10)
r.raise_for_status()
assert len(r.content)==size-tail_start
tail=r.content
class Parts(io.RawIOBase):
    def __init__(self): self.pos=0
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,pos,whence=0):
        self.pos=pos+[0,self.pos,size][whence]
        return self.pos
    def read(self,n=-1):
        n=min(n,size-self.pos) if n>=0 else size-self.pos
        if self.pos>=tail_start:
            result=tail[self.pos-tail_start:self.pos-tail_start+n]
            self.pos+=len(result)
            return result
        buf=bytearray()
        while n>0:
            start=self.pos//length*length
            p=root/'assets/download_parts'/str(start)
            for i in range(120):
                if p.exists():break
                time.sleep(1)
            count=min(n,start+length-self.pos)
            with p.open('rb') as f:
                f.seek(self.pos-start); chunk=f.read(count)
            if len(chunk)!=count:raise ValueError('missing range')
            self.pos+=count; n-=count
            buf.extend(chunk)
        return bytes(buf)
with zipfile.ZipFile(Parts()) as z:
    entries=[m for m in z.infolist() if m.filename.startswith('29999/params/')]
    for e in entries:
        z.extract(e,target)
print('verified params member CRCs:',len(entries),flush=True)

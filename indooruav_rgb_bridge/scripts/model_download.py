"""Fetch the official model archive with validated HTTP byte ranges and resumable parts."""
import concurrent.futures
import hashlib
from pathlib import Path
import threading
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
URL = "https://modelscope.cn/models/valyentine/IndoorUAV-Agent/resolve/master/29999.zip"
API = "https://modelscope.cn/api/v1/models/valyentine/IndoorUAV-Agent/repo/files"
thread_local = threading.local()


def session():
    if not hasattr(thread_local, "http"):
        thread_local.http = requests.Session()
        thread_local.http.trust_env = False
    return thread_local.http


def fetch_part(start, end, size, out):
    key = out / str(start)
    if key.exists() and key.stat().st_size == end-start+1:
        return
    for attempt in range(6):
        try:
            with session().get(URL, headers={"Range":f"bytes={start}-{end}"}, params={"range_start":start}, stream=True, timeout=(15,45)) as resp:
                expected = f"bytes {start}-{end}/{size}"
                if resp.status_code != 206 or resp.headers.get("Content-Range") != expected:
                    raise ValueError((resp.status_code, resp.headers.get("Content-Range"), expected))
                tmp = key.with_suffix(".partial")
                with tmp.open("wb") as writer:
                    for chunk in resp.iter_content(1024*1024):
                        writer.write(chunk)
                if tmp.stat().st_size != end-start+1:
                    raise ValueError("short read")
                tmp.replace(key)
                return
        except (requests.RequestException, OSError, ValueError) as e:
            print(f"retry {start} {attempt}: {e}", flush=True)
            if attempt == 5: raise
            time.sleep(2**attempt)


def main():
    file = next(f for f in session().get(API,params={"Recursive":"false"},timeout=20).json()["Data"]["Files"] if f["Name"]=="29999.zip")
    size = file["Size"]
    print("model",size,file["Sha256"],flush=True)
    parts = ASSETS / "download_parts"
    parts.mkdir(exist_ok=True,parents=True)
    path = ASSETS/"29999.zip"
    stride=32*1024*1024
    starts=list(range(0,size,stride))
    if not path.exists():
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
            futures=[ex.submit(fetch_part,st,min(size,st+stride)-1,size,parts) for st in starts]
            for i,f in enumerate(concurrent.futures.as_completed(futures),1):
                f.result()
                if i%10==0:print(f"{i}/{len(starts)} parts", flush=True)
        digest=hashlib.sha256()
        with path.with_suffix(".tmp").open("wb") as dst:
            for start in starts:
                with (parts/str(start)).open("rb") as src:
                    while b:=src.read(4*1024*1024):
                        dst.write(b);digest.update(b)
        if digest.hexdigest().lower()!=file["Sha256"].lower():raise RuntimeError("archive digest mismatch")
        path.with_suffix(".tmp").replace(path)
        for start in starts:(parts/str(start)).unlink()
        parts.rmdir()
    elif hashlib.file_digest(path.open("rb"),"sha256").hexdigest().lower()!=file["Sha256"].lower(): raise RuntimeError("archive digest mismatch")
    print("extracting verified archive",flush=True)
    with zipfile.ZipFile(path) as z:
        for member in z.infolist():
            if member.filename.startswith("29999/params/") or member.filename == "29999/_CHECKPOINT_METADATA":
                z.extract(member, ASSETS)
    print("complete",flush=True)

if __name__=="__main__":main()

"""Check two original short held-out IndoorUAV-VLA examples alongside the local evaluation."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import shlex
import subprocess
import zipfile

import requests

from offline_eval import HERE, metrics, extract_poses


PUBLIC = "https://modelscope.cn/datasets/valyentine/Indoor_UAV/resolve/master/"
ROOT = HERE / "results" / "offline_native_sanity"


class RangeArchive(io.RawIOBase):
    """Provide seek/read on two multi-GB archives by retrieving just ZIP metadata and selected members."""
    def __init__(self, filename):
        self.pos = 0
        self.session = requests.Session()
        head = self.session.get(PUBLIC + filename, headers={"Range":"bytes=0-0"}, timeout=20)
        head.raise_for_status()
        self.n = int(head.headers["Content-Range"].split("/")[-1])
        self.url = head.url
        self.received = len(head.content)

    def seek(self, offset, whence=0):
        self.pos = {0:0, 1:self.pos, 2:self.n}[whence] + offset
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def read(self, size=-1):
        remaining = self.n - self.pos
        size = remaining if size < 0 else min(size, remaining)
        if size == 0:
            return b""
        if size > 15_000_000:
            raise ValueError(f"Unexpected ZIP member size {size}")
        response = self.session.get(self.url, headers={"Range":f"bytes={self.pos}-{self.pos+size-1}", "Accept-Encoding":"identity"}, timeout=30)
        if response.status_code != 206 or len(response.content) != size:
            raise IOError(f"Unexpected ZIP byte range {response.status_code} from {self.pos}")
        self.received += size
        self.pos += size
        return response.content

    def close(self):
        self.session.close()
        super().close()


def make_fixtures():
    test = json.loads((HERE / "upstream/IndoorUAV-Agent/online_eval/vla_eval/test_vla.json").read_text())
    fixtures = []
    with RangeArchive("replica.zip") as source, RangeArchive("vla_ins.zip") as lang:
        with zipfile.ZipFile(source) as images, zipfile.ZipFile(lang) as instructions:
            # Select one labeled turning task and one approach/turn task, in different apartments.
            # The public pure-forward Replica example lasts only 0.8 s.
            selected = {"approach": "/replica/apartment_2/traj_2/vla_ins_3.json",
                        "turn": "/replica/apartment_0/traj_3/vla_ins_1.json"}
            for kind, key in selected.items():
                subtask = json.loads(instructions.read("vla_ins"+key).decode("gbk"))
                first, last = subtask["source"][:2]
                if last - first < 10:
                    raise ValueError("Native source annotation shorter than 1 s")
                base = "home/yuliu/dataset" + key.rsplit("/", 1)[0]
                poses_member = base + "/posture.json"
                rgb_member = base + f"/screenshots/{first}.png"
                poses = json.loads(images.read(poses_member))
                rgb = images.read(rgb_member)
                path = ROOT / f"{kind}_frame{first}.png"
                path.write_bytes(rgb)
                window = [[*p[:3], math.radians(p[3])] for p in poses[first-1:first+10]]
                if len(window) != 11:
                    raise ValueError("Native example misses frames")
                req = {"id":f"official-heldout-{kind}", "rgb_path":str(path), "ref_rgb_path":str(path),
                       "instruction":subtask["instruction"], "state":window[0], "seed":37+len(fixtures)}
                fixtures.append({"test_vla_key":key, "test_metadata":test[key], "annotation":subtask,
                                 "rgb_member":rgb_member, "rgb_sha256":hashlib.sha256(rgb).hexdigest(),
                                 "pose_member":poses_member, "first_target_frame_1based":first+1,
                                 "last_target_frame_1based":first+10, "target_poses":window[1:], "request":req})
        received = {"replica.zip":source.received,"vla_ins.zip":lang.received}
    return {"source_urls":[PUBLIC+"replica.zip",PUBLIC+"vla_ins.zip"], "retrieved_bytes":received, "examples":fixtures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", help="JSONL worker command")
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True, parents=True)
    fixture = ROOT / "manifest.json"
    if not fixture.exists():
        fixture.write_text(json.dumps(make_fixtures(), indent=2, ensure_ascii=False) + "\n")
    manifest = json.loads(fixture.read_text())
    for case in manifest["examples"]:
        image = Path(case["request"]["rgb_path"]).read_bytes()
        if hashlib.sha256(image).hexdigest() != case["rgb_sha256"]:
            raise ValueError("Stored source image differs")
    if not args.worker:
        print(json.dumps(manifest["retrieved_bytes"]))
        return
    results = []
    with subprocess.Popen(shlex.split(args.worker), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as worker:
        try:
            for case in manifest["examples"]:
                req = case["request"]
                worker.stdin.write(json.dumps(req) + "\n")
                worker.stdin.flush()
                response = json.loads(worker.stdout.readline())
                path = extract_poses(response, req)
                results.append({"test_vla_key":case["test_vla_key"], "response":response,
                                "model": metrics(path,case["target_poses"]),
                                "stationary":metrics([req["state"]]*10,case["target_poses"])})
        finally:
            worker.stdin.close()
    (ROOT / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

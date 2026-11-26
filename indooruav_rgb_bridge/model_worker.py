"""Local JSONL inference adapter for the published IndoorUAV-Agent π0 checkpoint."""
import contextlib
import json
import os
from pathlib import Path
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from openpi.models import model as model_api
from openpi.models import pi0, tokenizer
from openpi.policies import libero_policy, policy
from openpi.shared import normalize
from openpi import transforms

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"


def create_policy():
    """Use the evaluation-only subset of pi0_uav_low_mem_finetune from upstream config."""
    cfg = pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
                       action_horizon=10, max_token_len=240)
    checkpoint = Path(os.environ.get("INDOORUAV_CHECKPOINT", ASSETS / "29999"))
    params = model_api.restore_params(checkpoint / "params", dtype=jnp.bfloat16)
    network = cfg.load(params)
    stats = normalize.load(ASSETS)
    mask = transforms.make_bool_mask(4)
    return policy.Policy(
        network,
        transforms=[
            libero_policy.LiberoInputs(action_dim=cfg.action_dim, model_type=cfg.model_type),
            transforms.DeltaActions(mask),
            transforms.Normalize(stats),
            transforms.ResizeImages(224, 224),
            transforms.TokenizePrompt(tokenizer.PaligemmaTokenizer(cfg.max_token_len)),
        ],
        output_transforms=[transforms.Unnormalize(stats), transforms.AbsoluteActions(mask), libero_policy.LiberoOutputs()],
    )


def rgb_from(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def infer(network, req):
    state = np.asarray(req["state"], dtype=np.float32)
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("Expected four finite state coordinates")
    instruction = req["instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A nonempty instruction is required")
    seed = int(req.get("seed", 0))
    network._rng = jax.random.key(seed)
    obs = {"observation/state": state, "observation/image": rgb_from(req["rgb_path"]),
           "observation/ref_image": rgb_from(req["ref_rgb_path"]), "task": instruction}
    out = network.infer(obs)
    waypoints = np.asarray(out["actions"])
    if waypoints.shape != (10, 4) or not np.isfinite(waypoints).all():
        raise ValueError(f"Invalid trajectory output: {waypoints.shape}")
    return {"id": req["id"], "waypoints": waypoints.tolist(), "seed": seed,
            "infer_ms": out["policy_timing"]["infer_ms"]}


def main():
    output = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        print(f"Loading IndoorUAV-Agent with {jax.devices()}", file=sys.stderr, flush=True)
        network = create_policy()
        print("IndoorUAV-Agent ready", file=sys.stderr, flush=True)
        for line in sys.stdin:
            req = None
            try:
                req = json.loads(line)
                response = infer(network, req)
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                response = {"id": req.get("id") if isinstance(req, dict) else None, "error": str(error)}
            print(json.dumps(response, allow_nan=False), file=output, flush=True)


if __name__ == "__main__":
    main()

# IndoorUAV-Agent inference worker

The prepared environment is `env/runtime`. Run `scripts/model_worker.sh` from any working directory to start a persistent JSONL worker. It consumes one request per line with `id`, `instruction`, `state`, `rgb_path`, `ref_rgb_path` and optional integer `seed`. Both paths point to RGB images on this machine. `state` uses the four public IndoorUAV state coordinates. Each successful reply includes `waypoints` with 10 future four-coordinate poses. The official VLA evaluation selects waypoint index 9. See `model_provenance.json` for source and asset hashes.

The installed model uses the published `pi0_uav_low_mem_finetune` configuration with the upstream network, transforms and sampler. The local worker constructs the inference subset to avoid importing training datasets and loading their dependencies. Both image inputs go through the published Libero input adapter and the 224-pixel letterbox transform. The worker applies the published state normalization, action unnormalization and delta-to-absolute transforms. Each request seeds the upstream JAX sampling RNG independently. Upstream model code lives in `upstream/IndoorUAV-Agent`.

## Recreate the runtime

The following commands target this workstation with a CUDA 12.8 toolkit and the existing cuDNN/NCCL shared libraries. The shell wrapper reads these libraries from the read-only CUDA Python packages. Set `INDOORUAV_CUDA_LIBS` to supply a different NVIDIA shared-library root.

```bash
cd /home/zr_fan/3DGS_env/indooruav_rgb_bridge
mkdir -p env assets/cache/big_vision upstream model_logs
git clone https://github.com/valyentinee/IndoorUAV-Agent.git upstream/IndoorUAV-Agent
git -C upstream/IndoorUAV-Agent checkout 4626bbe9c54350a0e7b2631da6211ffca8ac809f
python -m pip install --target env/uvbin uv==0.13.9
env/uvbin/bin/uv venv --python /usr/bin/python3.12 env/runtime
env/uvbin/bin/uv pip install --python env/runtime/bin/python -r env/runtime.lock
unzip -p upstream/IndoorUAV-Agent/training_data.zip training_data/norm_stats.json > assets/norm_stats.json
curl -fLsS --retry 3 https://storage.googleapis.com/big_vision/paligemma_tokenizer.model -o assets/cache/big_vision/paligemma_tokenizer.model
env/runtime/bin/python scripts/model_download.py
env/uvbin/bin/uv pip check --python env/runtime/bin/python
scripts/model_worker.sh -c 'import jax; print(jax.devices())'
python -B bridge_client.py serve -- scripts/model_worker.sh
```

The verified checkpoint lives under `assets/29999`. The downloaded ZIP is retained for hash verification and offline recovery. See `model_logs/smoke_request.jsonl` for a complete request with recorded scene imagery.

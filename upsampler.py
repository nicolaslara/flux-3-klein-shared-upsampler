"""Serve the FLUX.3 Klein prompt upsampler alone on one GPU, shared by every image worker."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import modal

GPU = os.getenv("GPU", "B200").strip().upper()
if GPU not in ("H100", "H100!", "H200", "B200"):
    raise ValueError("Choose one GPU type: H100, H100!, H200 or B200")
REGION = os.getenv("REGION") or None
APP_NAME = os.getenv("APP_NAME", "flux3-upsampler")
MAX_SEQS = 64
MEMORY_FRACTION = "0.85"
PUP_REVISION = "3f3edb52b2c55de0d49feab28598b68f7f8e4830"
REGISTRY = "lmsysorg/sglang@sha256:b4f09a2ccbcfcc262d49600c3cd7f8581a793ea7d104947257b6a95500953b90"
SAMPLING = dict(temperature=0.0, top_p=0.95, top_k=20, min_p=0.0, max_tokens=3584, presence_penalty=1.5)
ROOT = Path(__file__).resolve().parent


def download_pup():
    from huggingface_hub import snapshot_download

    snapshot_download(
        "black-forest-labs/flux-3-image-klein-pup",
        revision=PUP_REVISION,
        local_dir="/weights/pup",
        token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
    )


image = (
    modal.Image.from_registry(REGISTRY)
    .apt_install("g++", "libgl1", "libglib2.0-0", "libnuma1")
    .env({"HF_HUB_OFFLINE": "0", "HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(download_pup, secrets=[modal.Secret.from_name("huggingface")], timeout=1800)
    .add_local_file(ROOT / "requirements-pup.txt", "/opt/requirements-pup.txt", copy=True)
    .run_commands(
        "python3 -m pip install --no-deps uv==0.9.18",
        "uv venv --python python3 /opt/pup",
        "uv pip install --python /opt/pup/bin/python --index-strategy unsafe-best-match -r /opt/requirements-pup.txt",
        "uv pip install --python /opt/pup/bin/python flashinfer-jit-cache==0.6.8.post1 --extra-index-url https://flashinfer.ai/whl/cu130",
        "ln -s lib /opt/pup/lib/python3.12/site-packages/nvidia/cu13/lib64",
        "ln -s libcudart.so.13 /opt/pup/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so",
    )
    .env({"GPU": GPU, "REGION": REGION or "", "APP_NAME": APP_NAME, "HF_HUB_OFFLINE": "1"})
)
app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=32768,
    min_containers=1,
    max_containers=1,
    startup_timeout=900,
    timeout=300,
    region=REGION,
)
@modal.concurrent(max_inputs=MAX_SEQS, target_inputs=MAX_SEQS)
class Upsampler:
    @modal.enter()
    def start(self):
        import httpx
        from transformers import AutoTokenizer

        self.started_at = time.time()
        self.task_id = os.getenv("MODAL_TASK_ID")
        self.tokenizer = AutoTokenizer.from_pretrained("/weights/pup", local_files_only=True)
        self.system_prompts = json.loads(Path("/weights/pup/system_prompts.json").read_text())
        self.stop_tokens = [self.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        cuda = "/opt/pup/lib/python3.12/site-packages/nvidia/cu13"
        env = {
            **os.environ,
            "CUDA_HOME": cuda,
            "LD_LIBRARY_PATH": f"{cuda}/lib",
            "VLLM_HOST_IP": "127.0.0.1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_USE_DEEP_GEMM": "0",
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCHINDUCTOR_CACHE_DIR": "/cache/pup/inductor",
            "TRITON_CACHE_DIR": "/cache/pup/triton",
            "CUDA_CACHE_PATH": "/cache/pup/cuda",
        }
        self.command = [
            "/opt/pup/bin/vllm",
            "serve",
            "/weights/pup",
            "--host",
            "127.0.0.1",
            "--port",
            "9002",
            "--served-model-name",
            "pup",
            "--dtype",
            "bfloat16",
            "--tensor-parallel-size",
            "1",
            "--enable-prefix-caching",
            "--generation-config",
            "vllm",
            "--max-model-len",
            "8192",
            "--max-num-seqs",
            str(MAX_SEQS),
            "--gpu-memory-utilization",
            MEMORY_FRACTION,
            "--limit-mm-per-prompt",
            '{"image":10}',
            "--attention-config",
            '{"backend":"TRITON_ATTN"}',
        ]
        self.process = subprocess.Popen(self.command, env=env, start_new_session=True)
        deadline = time.monotonic() + 850
        with httpx.Client(timeout=5) as client:
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError("vLLM exited during startup; inspect the container logs")
                try:
                    if client.get("http://127.0.0.1:9002/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("vLLM exceeded the startup deadline")
                time.sleep(1)
        with httpx.Client(timeout=180) as client:
            text = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompts["t2i"]},
                    {"role": "user", "content": "Two blue mugs on a wooden table by a window"},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            answer = (
                client.post(
                    "http://127.0.0.1:9002/v1/completions",
                    json={
                        "model": "pup",
                        "prompt": self.tokenizer.encode(text, add_special_tokens=False),
                        **SAMPLING,
                        "stop_token_ids": self.stop_tokens,
                    },
                )
                .raise_for_status()
                .json()["choices"][0]
            )
            if answer.get("finish_reason") != "stop" or not answer.get("text", "").strip():
                raise RuntimeError("Warmup prompt expansion did not complete")
        self.client = httpx.AsyncClient(timeout=240)
        self.ready_at = time.time()
        self.served = 0
        print(
            "UPSAMPLER_READY "
            + json.dumps({"task_id": self.task_id, "gpu": GPU, "startup_s": self.ready_at - self.started_at}),
            flush=True,
        )

    @modal.method()
    async def expand(self, payload: dict) -> dict:
        """Forward one upsampler request; the caller supplies the vLLM path and JSON body."""
        started = time.perf_counter()
        self.served += 1
        response = await self.client.post("http://127.0.0.1:9002" + payload["path"], json=payload["json"])
        return {
            "status": response.status_code,
            "result": response.json() if response.status_code == 200 else response.text[:2000],
            "upsampler_ms": (time.perf_counter() - started) * 1000,
            "task_id": self.task_id,
        }

    @modal.method()
    def health(self) -> dict:
        return {
            "task_id": self.task_id,
            "gpu": GPU,
            "region": os.getenv("MODAL_REGION"),
            "max_num_seqs": MAX_SEQS,
            "memory_fraction": MEMORY_FRACTION,
            "started_at": self.started_at,
            "ready_at": self.ready_at,
            "served": self.served,
            "alive": self.process.poll() is None,
            "command": self.command,
        }

    @modal.exit()
    def stop(self):
        if getattr(self, "process", None) is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)

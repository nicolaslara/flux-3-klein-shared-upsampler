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
APP_NAME = os.getenv("APP_NAME", "flux3-upsampler")
WARM = int(os.getenv("WARM", "1"))
REPLICAS = int(os.getenv("REPLICAS", "1"))
IDLE_SECONDS = int(os.getenv("IDLE_SECONDS", "120"))
SNAPSHOT = os.getenv("SNAPSHOT", "1") == "1"
MAX_SEQS = 64
MEMORY_FRACTION = "0.85"
PUP_REVISION = "3f3edb52b2c55de0d49feab28598b68f7f8e4830"
REGISTRY = "lmsysorg/sglang@sha256:b4f09a2ccbcfcc262d49600c3cd7f8581a793ea7d104947257b6a95500953b90"
SAMPLING = dict(temperature=0.0, top_p=0.95, top_k=20, min_p=0.0, max_tokens=3584, presence_penalty=1.5)
ROOT = Path(__file__).resolve().parent
PUP_CUDA = "/opt/pup/lib/python3.12/site-packages/nvidia/cu13"
# The compiled graph is keyed on the configuration, so the build and the container must start vLLM with
# the same command and environment for the copy in the image to be reused.
PUP_ENV = {
    "CUDA_HOME": PUP_CUDA,
    "LD_LIBRARY_PATH": f"{PUP_CUDA}/lib",
    "VLLM_HOST_IP": "127.0.0.1",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_USE_DEEP_GEMM": "0",
    "VLLM_SERVER_DEV_MODE": "1",
    "VLLM_CACHE_ROOT": "/cache/pup/vllm",
    "TORCHINDUCTOR_CACHE_DIR": "/cache/pup/inductor",
    "TRITON_CACHE_DIR": "/cache/pup/triton",
    "CUDA_CACHE_PATH": "/cache/pup/cuda",
}
COMMAND = [
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


def launch():
    return subprocess.Popen(COMMAND, env={**os.environ, **PUP_ENV, "HOME": "/root"}, start_new_session=True)


def wait_for_vllm(process, timeout):
    import httpx

    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=5) as client:
        while True:
            if process.poll() is not None:
                raise RuntimeError("vLLM exited during startup; inspect the container logs")
            try:
                if client.get("http://127.0.0.1:9002/health").status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("vLLM exceeded the startup deadline")
            time.sleep(1)


def expand_once(tokenizer, system_prompt, prompt, stop_tokens):
    import httpx

    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    with httpx.Client(timeout=180) as client:
        answer = (
            client.post(
                "http://127.0.0.1:9002/v1/completions",
                json={
                    "model": "pup",
                    "prompt": tokenizer.encode(text, add_special_tokens=False),
                    **SAMPLING,
                    "stop_token_ids": stop_tokens,
                },
            )
            .raise_for_status()
            .json()["choices"][0]
        )
    if answer.get("finish_reason") != "stop" or not answer.get("text", "").strip():
        raise RuntimeError("Warmup prompt expansion did not complete")
    return answer["text"]


def compile_graph():
    """Start vLLM on the build GPU once so its compiled graph and kernel caches ship in the image."""
    from transformers import AutoTokenizer

    started = time.monotonic()
    process = launch()
    try:
        wait_for_vllm(process, 1500)
        ready_s = time.monotonic() - started
        tokenizer = AutoTokenizer.from_pretrained("/weights/pup", local_files_only=True)
        system_prompt = json.loads(Path("/weights/pup/system_prompts.json").read_text())["t2i"]
        stop_tokens = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        expand_once(tokenizer, system_prompt, "Two blue mugs on a wooden table by a window", stop_tokens)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(60)
    written = sum(f.stat().st_size for f in Path("/cache/pup/vllm").rglob("*") if f.is_file())
    if not written:
        raise RuntimeError("The build warmup left the compiled graph cache empty; vLLM's cache path changed")
    print("UPSAMPLER_BUILD " + json.dumps({"ready_s": ready_s, "cache_bytes": written}), flush=True)


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
    .run_function(compile_graph, gpu=GPU, cpu=4, memory=32768, timeout=2400)
    .env(
        {
            "GPU": GPU,
            "APP_NAME": APP_NAME,
            "WARM": str(WARM),
            "REPLICAS": str(REPLICAS),
            "IDLE_SECONDS": str(IDLE_SECONDS),
            "SNAPSHOT": str(int(SNAPSHOT)),
            "HF_HUB_OFFLINE": "1",
        }
    )
)
app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu=GPU,
    cpu=4,
    memory=32768,
    min_containers=WARM,
    max_containers=REPLICAS,
    scaledown_window=IDLE_SECONDS,
    startup_timeout=900,
    timeout=300,
    enable_memory_snapshot=SNAPSHOT,
    experimental_options={"enable_gpu_snapshot": True} if SNAPSHOT else {},
)
@modal.concurrent(max_inputs=MAX_SEQS, target_inputs=MAX_SEQS)
class Upsampler:
    @modal.enter(snap=SNAPSHOT)
    def start(self):
        from transformers import AutoTokenizer

        self.started_at = time.time()
        self.source_task = os.getenv("MODAL_TASK_ID")
        self.process = launch()
        self.tokenizer = AutoTokenizer.from_pretrained("/weights/pup", local_files_only=True)
        self.system_prompts = json.loads(Path("/weights/pup/system_prompts.json").read_text())
        self.stop_tokens = [self.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        wait_for_vllm(self.process, 850)
        expand_once(
            self.tokenizer,
            self.system_prompts["t2i"],
            "Two blue mugs on a wooden table by a window",
            self.stop_tokens,
        )
        self.prepared_at = time.time()

    @modal.enter()
    def serve(self):
        import httpx

        resumed = time.time()
        self.task_id = os.getenv("MODAL_TASK_ID")
        wait_for_vllm(self.process, 120)
        self.client = httpx.AsyncClient(timeout=240)
        self.ready_at = time.time()
        self.served = 0
        print(
            "UPSAMPLER_READY "
            + json.dumps(
                {
                    "task_id": self.task_id,
                    "restored": self.task_id != self.source_task,
                    "gpu": GPU,
                    "startup_s": self.prepared_at - self.started_at,
                    "restore_hook_s": self.ready_at - resumed,
                }
            ),
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
    def system_prompts(self) -> dict:
        return self.system_prompts

    @modal.method()
    def health(self) -> dict:
        return {
            "task_id": self.task_id,
            "gpu": GPU,
            "region": os.getenv("MODAL_REGION"),
            "max_num_seqs": MAX_SEQS,
            "memory_fraction": MEMORY_FRACTION,
            "source_task_id": self.source_task,
            "restored": self.task_id != self.source_task,
            "started_at": self.started_at,
            "prepared_at": self.prepared_at,
            "ready_at": self.ready_at,
            "served": self.served,
            "alive": self.process.poll() is None,
            "command": COMMAND,
        }

    @modal.exit()
    def stop(self):
        if getattr(self, "process", None) is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)


@app.local_entrypoint()
def main(warm: int = -1):
    """Change how many prompt upsamplers stay up without a redeploy."""
    if warm >= 0:
        if warm > REPLICAS:
            raise ValueError("--warm must be between zero and REPLICAS")
        modal.Cls.from_name(APP_NAME, "Upsampler")().update_autoscaler(min_containers=warm)
        print(f"{APP_NAME}: {warm} kept up")

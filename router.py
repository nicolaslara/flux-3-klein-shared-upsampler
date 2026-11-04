"""Always-on entry point: expands the prompt and starts the worker at the same time, then forwards the caption."""

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

import modal
import modal.exception

APP_NAME = os.getenv("APP_NAME", "flux3-router")
UPSAMPLER_APP = os.getenv("UPSAMPLER_APP", "flux3-upsampler")
# model name -> worker app; a name that is not listed maps to flux3-klein-<name>-shared.
WORKERS = dict(item.split("=", 1) for item in os.getenv("WORKERS", "").split(",") if item)
DEADLINE = 900
SAMPLING = dict(model="pup", temperature=0.0, top_p=0.95, top_k=20, min_p=0.0, max_tokens=3584)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi==0.143.0", "uvicorn==0.54.0", "httpx==0.28.1", "pydantic==2.14.1", "pillow==12.3.0")
    .add_local_file(ROOT / "flux3_api.py", "/root/flux3_api.py", copy=True)
    .env(
        {
            "APP_NAME": APP_NAME,
            "UPSAMPLER_APP": UPSAMPLER_APP,
            "WORKERS": os.getenv("WORKERS", ""),
        }
    )
)
app = modal.App(APP_NAME)


def worker_app(model):
    return WORKERS.get(model, f"flux3-klein-{model}-shared")


def create_api(runtime):
    """Validate, expand the prompt while the worker wakes, forward the caption, relay the picture."""
    import httpx
    from fastapi import FastAPI, HTTPException
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse, Response
    from pydantic import Field

    from flux3_api import GenerationRequest, ServingPolicy, prepare_generation

    api = FastAPI()
    watches = {}

    class RouterRequest(GenerationRequest):
        model: str = Field(default="distilled", min_length=1, max_length=64, pattern=r"^[a-z0-9-]+$")

    class Watch:
        """One /wake poll loop per worker, shared by every request waiting for that worker."""

        def __init__(self, url):
            self.url = url
            self.ready_at = float("-inf")
            self.task = None

        async def wait(self):
            started = time.perf_counter()
            if time.monotonic() - self.ready_at > 1.0:
                if self.task is None or self.task.done():
                    self.task = asyncio.create_task(self.poll())
                await asyncio.shield(self.task)
            return (time.perf_counter() - started) * 1000

        async def poll(self):
            deadline = time.monotonic() + runtime.deadline
            while True:
                try:
                    response = await runtime.client.post(self.url + "/wake", timeout=10)
                    if response.status_code == 200:
                        self.ready_at = time.monotonic()
                        return
                    if response.status_code in (401, 403):
                        raise HTTPException(502, "Worker rejected the router's proxy token")
                except httpx.TransportError:
                    pass
                if time.monotonic() > deadline:
                    raise HTTPException(504, "Worker did not become ready")
                await asyncio.sleep(1)

    @api.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc):
        # Validation inputs can contain prompts or image data, so only return field names.
        fields = sorted({str(error["loc"][-1]) for error in exc.errors()})
        return JSONResponse(
            status_code=422, content={"detail": "Invalid request fields: " + ", ".join(fields)}
        )

    async def expand(prepared):
        # The chat endpoint serves text and edit requests alike; the worker pre-tokenizes text requests itself.
        started = time.perf_counter()
        body = {
            "messages": prepared.text.messages(runtime.system_prompts),
            **SAMPLING,
            "presence_penalty": prepared.text.presence_penalty,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            reply = await runtime.upsampler.expand.remote.aio({"path": "/v1/chat/completions", "json": body})
        except modal.exception.NotFoundError as exc:
            raise HTTPException(503, "Prompt upsampler app is not deployed") from exc
        except Exception as exc:  # noqa: BLE001 - any SDK or transport failure means the upsampler is unavailable
            raise HTTPException(502, "Prompt upsampler unavailable") from exc
        if reply["status"] in (400, 422):
            raise HTTPException(422, "Prompt upsampler rejected the request or its context length")
        if reply["status"] != 200:
            raise HTTPException(502, "Prompt upsampler failed")
        choice = reply["result"]["choices"][0]
        caption = choice["message"]["content"]
        if not isinstance(caption, str) or not caption.strip() or choice.get("finish_reason") != "stop":
            raise HTTPException(502, "Prompt expansion incomplete")
        tokens = reply["result"].get("usage", {}).get("completion_tokens", 0)
        return caption, (time.perf_counter() - started) * 1000, reply.get("upsampler_ms", 0.0), tokens

    async def gather(*coroutines):
        tasks = [asyncio.create_task(c) for c in coroutines]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            raise

    @api.get("/health")
    async def health():
        return {
            "upsampler_app": UPSAMPLER_APP,
            "workers": dict(runtime.urls),
            "deadline_s": runtime.deadline,
            "started_at": runtime.started_at,
        }

    @api.post("/generate")
    async def generate(body: RouterRequest):
        started = time.perf_counter()
        kind = "distilled" if body.model == "distilled" else "base"  # LoRA workers serve Base
        policy = ServingPolicy(quality="lossless" if kind == "distilled" else "exact")
        request = body.model_dump(exclude={"model"}, exclude_none=True)
        try:
            # Reference decoding and resizing are CPU work; keep them off the event loop.
            prepared = await asyncio.to_thread(prepare_generation, request, kind, policy)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        target = worker_app(body.model)
        try:
            async with asyncio.timeout(runtime.deadline):
                try:
                    url = await runtime.worker_url(target)
                except LookupError as exc:
                    raise HTTPException(503, f"Worker app {target} is not deployed") from exc
                watch = watches.setdefault(target, Watch(url))
                if prepared.text.needs_expansion:
                    wake_ms, (expansion, pup_ms, upsampler_ms, pup_tokens) = await gather(
                        watch.wait(), expand(prepared)
                    )
                else:
                    wake_ms, expansion, pup_ms, upsampler_ms, pup_tokens = (
                        await watch.wait(),
                        None,
                        0.0,
                        0.0,
                        0,
                    )
                request.pop("prompt", None)
                request["caption"] = prepared.text.finish(expansion)
                while True:
                    response = await runtime.client.post(
                        url + "/generate", json=request, timeout=runtime.deadline
                    )
                    if response.status_code == 503:
                        # The worker went away between its wake and this request; wait for it again.
                        watch.ready_at = float("-inf")
                        wake_ms += await watch.wait()
                    elif response.status_code == 429:
                        # Every worker is full; hold the request here instead of passing the 429 on.
                        queued = time.perf_counter()
                        await asyncio.sleep(1)
                        wake_ms += (time.perf_counter() - queued) * 1000
                    else:
                        break
        except TimeoutError as exc:
            raise HTTPException(504, "Router deadline exceeded") from exc
        except httpx.TransportError as exc:
            raise HTTPException(502, "Worker connection failed") from exc
        ours = {"x-pup-ms", "x-upsampler-ms", "x-pup-output-tokens"}
        headers = {
            name: value
            for name, value in response.headers.items()
            if name.lower().startswith("x-") and name.lower() not in ours
        }
        # The worker only ever sees captions, so the expansion timings come from the router.
        headers["X-Pup-Ms"] = str(pup_ms)
        headers["X-Upsampler-Ms"] = str(upsampler_ms)
        headers["X-Pup-Output-Tokens"] = str(pup_tokens)
        headers["X-Wake-Ms"] = str(wake_ms)
        headers["X-Router-Ms"] = str((time.perf_counter() - started) * 1000)
        return Response(
            response.content,
            status_code=response.status_code,
            media_type=response.headers.get("content-type"),
            headers=headers,
        )

    return api


@app.server(
    image=image,
    cpu=2,
    memory=2048,
    port=8000,
    min_containers=1,
    max_containers=1,
    target_concurrency=64,
    max_concurrency=64,
    startup_timeout=120,
    routing_region="us-east",
    # The routing region, where every worker URL is served; keeping the router there keeps the hops short.
    compute_region="us-east",
    unauthenticated=False,
    secrets=[modal.Secret.from_name("flux3-router")],
)
class Router:
    @modal.enter()
    def start(self):
        self.started_at = time.time()
        self.deadline = DEADLINE
        self.auth = {"Modal-Key": os.environ["MODAL_KEY"], "Modal-Secret": os.environ["MODAL_SECRET"]}
        self.upsampler = modal.Cls.from_name(UPSAMPLER_APP, "Upsampler")()
        self.system_prompts = self.upsampler.system_prompts.remote()
        self.urls = {}
        self.http = threading.Thread(target=lambda: asyncio.run(self.serve()), daemon=True)
        self.http.start()

    async def worker_url(self, app_name):
        if app_name not in self.urls:
            try:
                url = await modal.Server.from_name(app_name, "Server").get_url.aio()
            except modal.exception.NotFoundError:
                url = None
            if not url:
                raise LookupError(app_name)
            self.urls[app_name] = url
        return self.urls[app_name]

    async def serve(self):
        import httpx
        import uvicorn

        async with httpx.AsyncClient(headers=self.auth, timeout=DEADLINE) as client:
            self.client = client
            await uvicorn.Server(
                uvicorn.Config(create_api(self), host="0.0.0.0", port=8000, log_level="warning")
            ).serve()


@app.local_entrypoint()
def main():
    """Print the router's URL, the one clients use."""
    print(modal.Server.from_name(APP_NAME, "Router").get_url())

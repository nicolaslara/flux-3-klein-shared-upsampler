# FLUX.3 Klein on Modal, with a shared prompt upsampler

[FLUX.3 Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) (Distilled, [Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base), or Base with a LoRA) as authenticated HTTP endpoints on Modal. Each worker runs the image pipeline (SGLang, FP8R) on one B200; every prompt goes to one always-warm [upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) service (vLLM) on another. Same API as `../customer`. Workers scale to zero; the upsampler stays up.

Choose this layout for more than about one worker's worth of traffic (0.5 pictures per second sustained) or more than one endpoint: Distilled, Base and every LoRA are separate worker apps sharing the one upsampler. For a single quiet endpoint it is the wrong choice: the upsampler costs $6.70 per hour with no worker up, so use `../customer`.

Distilled, 1024x1024, B200, against `../customer`:

| | One container per replica | Shared upsampler and workers |
|---|---|---|
| Per picture, warm, one client at a time | 1.99 s | 2.02 s |
| 3 B200 at 1.6 pictures/s, p50 / p95 | 3.13 / 5.20 s | 2.25 / 3.15 s (2 workers + upsampler) |
| 3 B200, most pictures per second | 1.8 | 2.8 |
| Cold start with a snapshot, to the first picture | 23 to 37 s | 16 to 24 s |
| Cold start without one | 530 s | 230 s |
| Kept warm, per hour | $6.95 per container | $6.70 upsampler + $6.95 per worker |

`upsampler.py` is the shared upsampler service, `worker.py` the image worker, `flux3_api.py` its request handling, `requirements-pup.txt` the upsampler's locked vLLM environment (the name stays), `client.py` an example client.

## Quick start

Accept the Hugging Face terms for the model and the prompt upsampler, then deploy the upsampler first: a worker's first start calls it and fails if the upsampler app is not in the same environment.

```
uv sync
uv run modal setup
uv run modal secret create huggingface HF_TOKEN=hf_...
uv run modal deploy upsampler.py
uv run modal deploy worker.py
```

Each deploy builds its image (19 GB for the upsampler, 16 GB for a worker); check the upsampler with `uv run python -c "import modal; print(modal.Cls.from_name('flux3-upsampler', 'Upsampler')().health.remote())"` (a `modal run` of the file would start a second, ephemeral upsampler). The worker's deploy prints its URL; export it with a proxy auth token from the Modal dashboard:

```
export URL=https://<workspace>-<environment>--flux3-klein-distilled-shared-server.us-east.modal.direct
export MODAL_KEY=wk-...
export MODAL_SECRET=ws-...
uv run python client.py $URL "Two blue mugs on a wooden table by a window" --seed 42 --out image.jpg
```

With no worker up every request gets an immediate 503 and starts one, so `client.py` polls `/wake` until it answers 200, then sends `/generate` once; a non-200 means no picture was made.

## API

`POST /generate`, JSON body, unknown fields rejected: exactly one of `prompt` (expanded by the upsampler) and `caption` (used as is, the upsampler is not called); `images`, up to ten base64-encoded references of 10 MB each; `width` and `height`, multiples of 16 up to 4096 per side and 4 megapixels, default 1024 or the first reference's size; `seed` (0); `steps` (16 Distilled, 50 Base, up to 100); `guidance` (3.5, Base only); `output_format`, `jpeg` at quality 95 or `png`.

The response is the picture. `X-Pup-Ms` is the expansion time seen by the worker, `X-Upsampler-Ms` the part spent inside the upsampler service (the difference is the 40 ms hop), `X-Image-Queue-Ms` and `X-Image-Ms` the image stage, `X-Caption` the caption, URL-encoded. 422 is an invalid request, 429 a worker with 16 requests in flight, 502 an engine or upsampler failure, 503 no worker up or the upsampler unreachable, 504 the 300 s deadline. Editing and layouts are as in `../customer/README.md`.

## Cold starts and scaling

A worker stops 120 s after its last request, and the next request starts one: 16 to 24 s with a snapshot, 230 s without one (after a deploy, or on a new kind of worker); both are shorter than in `../customer` because the snapshot holds the image engine only. `uv run modal run worker.py --warm 1` keeps a worker up.

A worker accepts up to 16 requests, sends up to 4 of them to the upsampler at once and up to 2 to the image engine, which renders one while the other's JPEG is encoded. Modal adds workers as requests queue, up to `REPLICAS`. The upsampler is one container with 64 concurrent sequences; three workers at 4 pictures/s raised its expansion time from 1.37 s to 1.73 s, and its limit was not reached. A three-minute burst from zero at 1.5 requests/s served 257 of 274, first picture at 16.9 s, the rest 503s before the first worker was up (`../customer`: 230 of 274, first at 35 s).

## Configuration

Environment variables read at deploy time, as in `MODEL=base REPLICAS=4 uv run modal deploy worker.py`. A redeploy with the same code and settings keeps the snapshot; changing either makes the next start a fresh one.

`upsampler.py` reads `GPU` (`B200`; H200 was 0.2 to 0.3 s slower per expansion for $1.70 per hour less), `REGION` (unset; match the workers) and `APP_NAME` (`flux3-upsampler`).

`worker.py`:

| variable | default | notes |
|---|---|---|
| `MODEL` | `distilled` | `distilled` or `base`; separate apps |
| `LORA_REPO` | unset | Base only: a Hugging Face repo with one safetensors adapter, `org/name[@revision]`. Passed as `--lora-path`, which turns off the fused FP8 kernels; not measured |
| `GPU` | `B200` | H100 limits requests to 1536x1536 and two references |
| `WARM` | `0` | Workers kept up; `modal run worker.py --warm N` changes it live |
| `REPLICAS` | `1` | Most workers, up to 16 |
| `IDLE_SECONDS` | `120` | How long an idle worker stays up |
| `SNAPSHOT` | `1` | `0` makes every start fresh |
| `HIGH_RES` | `0` | Distilled on B200 only: text to image up to 6256 per side, one request at a time |
| `UPSAMPLER_APP` | `flux3-upsampler` | The upsampler app, in the same environment |
| `REGION` | unset | Worker region |
| `MAX_INFLIGHT` | `16` | Requests a worker accepts before 429, and its Modal concurrency; lower it to add workers sooner at the price of more cold starts |
| `IMAGE_SLOTS` | `2` | Requests at the image engine at once; `1` serializes JPEG encoding behind rendering |

## Measurements

Distilled FP8R, 1024x1024 JPEG, unique prompts, all in `us-east`, open-loop Poisson arrivals. Rows give the worker's image slots; admission was 4 in every run, not the 16 shipped here.

| fleet, image slots | offered/s | delivered/s | p50 / p95 s | expansion s | image s | $/h | $/1k pictures |
|---|---|---|---|---|---|---|---|
| 1 worker + upsampler, 1 | 1.5 | 1.26 | 3.10 / 5.25 | 1.37 | 0.68 | 13.65 | 3.01 |
| 1 worker + upsampler, 2 | 1.5 | 1.27 | 2.92 / 4.63 | 1.40 | 1.02 | 13.65 | 3.00 |
| 1 worker + upsampler, 2 | 4 clients | 1.52 | 2.51 / 3.11 | 1.45 | 1.06 | 13.65 | 2.50 |
| 2 workers + upsampler, 1 | 3.0 | 2.78 | 4.24 / 7.06 | 1.57 | 0.65 | 20.60 | 2.06 |
| 3 workers + upsampler, 1 | 4.0 | 3.75 | 3.12 / 4.97 | 1.73 | 0.65 | 27.55 | 2.04 |

A worker's image stage takes 0.65 s alone (1.0 s while two requests share it) against 0.87 to 1.17 s beside a colocated upsampler, which is where the extra pictures per GPU come from; one worker tops out near 1.5 pictures/s. Cold starts at the top are three probes each, wake to first picture, in-region client.

# FLUX.3 Klein on Modal, shared prompt upsampler

[FLUX.3 Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) (Distilled, [Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base), or Base with a LoRA) as authenticated HTTP endpoints on Modal, with the [prompt upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) running once for all of them. Text to image, editing with references, layout boxes, captions, JPEG or PNG.

## Architecture

Three Modal apps:

- The **prompt upsampler**: Black Forest Labs' 9B language model that turns a short prompt into the detailed caption the image model expects. One B200, always up.
- A **worker** per model: the image model itself, one app, one B200 per container. Starts on demand, stops after 120 s idle, restores from a GPU memory snapshot.
- The **router**: a small CPU app, always up, the one URL clients call. It expands the prompt and starts the worker at the same time, holds the request until the worker is ready, then hands it the caption.

```mermaid
sequenceDiagram
    participant C as Client
    participant R as Router (CPU, always up)
    participant U as Prompt upsampler (B200, always up)
    participant W as Worker (per model, B200, on demand)
    C->>R: POST /generate, prompt, model
    par
        R->>U: prompt
        U-->>R: caption
    and
        R->>W: wake, until ready
    end
    R->>W: caption
    W-->>R: picture
    R-->>C: picture
```

The caption takes about 1.3 s, the picture 0.7 s (Distilled) or 3.3 s (Base); a request that carries a `caption` skips the prompt upsampler. The router adds about 0.1 s to a warm picture and hides the caption inside the worker's start on a cold one. A worker called at its own URL asks the prompt upsampler itself and answers 503 while it starts.

## Against one container per replica

The alternative, [flux-3-klein-serving](https://github.com/nicolaslara/flux-3-klein-serving), runs both models on every B200. Distilled, 1024x1024, same day:

| | One container per replica | Shared prompt upsampler |
|---|---|---|
| Per picture, warm, one client | 2.0 s | 2.2 s |
| Cold start with a snapshot, to the first picture | 33 s | 17 s |
| Cold start without a snapshot | 9 min | 4 to 9 min |
| Burst from zero at 1.5 requests per second, first picture | 35 s | 17 s |
| Most pictures per second on three B200 | 1.8 | 2.8 |
| $ per 1,000 pictures at that rate | 3.20 | 2.06 |
| Kept warm, per hour | $6.95 per container | $6.70 prompt upsampler + $6.95 per worker |

Cold starts are shorter because a worker's snapshot holds only the image model. Pictures per GPU go up because the prompt upsampler no longer competes with the transformer: 1.5 pictures per second on a worker against 0.7 on a container doing both. The prompt upsampler is a fixed $6.70 per hour: one worker costs about the same per picture as one container, every further worker a third less. For one quiet endpoint, one container per replica is cheaper.

`upsampler.py` is the prompt upsampler, `worker.py` the worker, `router.py` the router, `flux3_api.py` the request handling shared by worker and router, `requirements-pup.txt` the prompt upsampler's locked environment.

## Quick start

Accept the terms on Hugging Face for the model and the prompt upsampler, then deploy in this order. The router stores a proxy auth token (Modal dashboard, Settings, Proxy Auth Tokens) in the `flux3-router` secret to call the workers:

```
uv sync
uv run modal setup
uv run modal secret create huggingface HF_TOKEN=hf_...
uv run modal deploy upsampler.py
uv run modal deploy worker.py
uv run modal secret create flux3-router MODAL_KEY=wk-... MODAL_SECRET=ws-...
uv run modal deploy router.py
```

The first deploys build the images (19 GB and 16 GB). The router's deploy prints its URL; export it with the client's proxy auth token:

```
export URL=https://<workspace>-<environment>--flux3-router-router.us-east.modal.direct
export MODAL_KEY=wk-...
export MODAL_SECRET=ws-...
```

The router holds the request while the worker starts; a non-200 means no picture was made.

```
curl -sf -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" \
  --json '{"prompt": "Two blue mugs on a wooden table by a window", "seed": 42}' \
  $URL/generate -o image.jpg
```

## API

`POST /generate`, JSON body. Unknown fields are rejected.

| field | default | notes |
|---|---|---|
| `prompt` | | Expanded by the prompt upsampler. Exactly one of `prompt` and `caption` |
| `caption` | | An already expanded caption, used as is |
| `images` | `[]` | Up to ten base64-encoded reference images, each up to 10 MB |
| `width`, `height` | 1024, or the first reference's size | Multiples of 16, up to 4096 per side and 4 megapixels |
| `seed` | `0` | Same seed and `caption`, same picture on the same GPU model; send `X-Caption` back as `caption` to reproduce |
| `steps` | 16 Distilled, 50 Base | 1 to 100 |
| `guidance` | 3.5 on Base | 0 to 20. Not used by Distilled |
| `output_format` | `jpeg` | `jpeg` (quality 95) or `png` |
| `model` | `distilled` | Router only: `distilled`, `base` or a `WORKERS` name |

The response is the picture; headers `X-Pup-Ms`, `X-Upsampler-Ms` and `X-Image-Ms` give the prompt expansion, its part inside the prompt upsampler, and the picture generation, `X-Caption` the caption URL-encoded, `X-Wake-Ms` the wait for the worker (the expansion overlaps it) and `X-Router-Ms` the total. The router never answers 503 during a cold start. 422 is an invalid request, 429 a worker holding 16 requests, 502 a prompt upsampler or engine failure, 503 an app that is not deployed, 504 the deadline, 900 s at the router and 300 s at a worker.

Editing: reference pictures go base64-encoded in `images`; the body goes in a file:

```
printf '{"prompt": "Make the mugs red", "images": ["%s"], "output_format": "png"}' "$(base64 < image.jpg | tr -d '\n')" > edit.json
curl -sf -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" --json @edit.json $URL/generate -o edit.png
```

Layouts: end the prompt with a JSON list of boxes, `[y_min, x_min, y_max, x_max]` on a 0 to 1000 canvas, origin top left: `"A blue mug on the left [{\"id\": \"mug\", \"bbox\": [200, 100, 800, 450], \"desc\": \"blue mug\"}]"`.

## Cold starts and scaling

A worker stops 120 s after its last request; the next request starts one, 17 s with a snapshot and four to nine minutes without (the first start after a deploy, or on a host that has no copy of the image or the snapshot), while the router expands the prompt and holds the request. `uv run modal run worker.py --warm 1` keeps a worker up. A worker holds 16 requests and renders one at a time; beyond that Modal queues and adds workers, up to `REPLICAS`. One prompt upsampler fed three workers at 3.75 pictures per second with headroom.

## Configuration

Environment variables read at deploy time, for example `MODEL=base REPLICAS=4 uv run modal deploy worker.py`. A redeploy with the same code and settings keeps the snapshot.

`upsampler.py`: `GPU` (`B200`; H200 is $1.70 per hour cheaper and 0.4 s slower per expansion), `REGION` (unset; use the same value as the workers), `APP_NAME` (`flux3-upsampler`).

`router.py`: `WORKERS` (unset; `mylora=flux3-klein-base-lora-shared,...`; a `model` not listed maps to `flux3-klein-<model>-shared`), `UPSAMPLER_APP` (`flux3-upsampler`), `APP_NAME` (`flux3-router`); the `flux3-router` secret holds its proxy token.

`worker.py`:

| variable | default | notes |
|---|---|---|
| `MODEL` | `distilled` | `distilled` or `base`; separate apps |
| `LORA_REPO` | unset | Base only: Hugging Face repo with one safetensors adapter, `org/name[@revision]`; not measured |
| `GPU` | `B200` | `B200`, `H200`, `H100` or `H100!` |
| `WARM` | `0` | Workers kept up |
| `REPLICAS` | `1` | Most workers, up to 16 |
| `IDLE_SECONDS` | `120` | How long an idle worker stays up |
| `SNAPSHOT` | `1` | `0` makes every start fresh |
| `HIGH_RES` | `0` | Distilled on B200 only: text to image up to 6256 per side |
| `UPSAMPLER_APP` | `flux3-upsampler` | The prompt upsampler app, same environment |
| `REGION` | unset | Match the prompt upsampler; unpinned pays about 0.1 s more per picture |
| `MAX_INFLIGHT` | `16` | Requests a worker holds before answering 429 |
| `IMAGE_SLOTS` | `2` | Requests at the image engine at once |

## Measurements

Distilled, 1024x1024, unique prompts, everything in `us-east`, requests arriving independently at the given rate; p50 and p95 are request to picture, waiting included.

| workers | requests per second | pictures per second | p50 / p95 | $ per 1,000 pictures |
|---|---|---|---|---|
| 1 worker | 1.5 | 1.27 | 2.8 / 4.1 s | 3.00 |
| 2 workers | 3.0 | 2.78 | 4.2 / 7.1 s | 2.06 |
| 3 workers | 4.0 | 3.75 | 3.1 / 5.0 s | 2.04 |

Three one-container replicas on the same three B200 deliver at most 1.8 pictures per second (p50 3.1 s, p95 5.2 s with 1.6 requests per second). In a burst from zero at 1.5 requests per second the first picture came after 17 s here and 35 s there. Through the router: warm pictures 2.0 s, a cold picture 19 s with a snapshot, and a one-minute burst from zero served 79 of 80 requests with no 503, first picture at 19 to 34 s. Base: 4.7 s per picture warm, 18 pictures per minute on one worker (p50 6.6 s).

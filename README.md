# FLUX.3 Klein on Modal, shared prompt upsampler

[FLUX.3 Klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) (Distilled, [Base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base), or Base with a LoRA) as authenticated HTTP endpoints on Modal. Two kinds of containers: one always-warm [prompt upsampler](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) service (vLLM) on a B200, and per model an image worker (text encoder, FP8R transformer and autoencoder in SGLang) on a B200 that scales to zero and restores from a GPU memory snapshot. Text to image, editing with reference images, layout boxes, captions, JPEG or PNG.

Use this layout for more than about 0.5 pictures per second sustained, or for more than one endpoint sharing the upsampler. For one quiet endpoint, the one-container-per-replica deployment in [flux-3-klein-serving](https://github.com/nicolaslara/flux-3-klein-serving) is cheaper, because the upsampler here costs $6.70 per hour even with no worker up.

| Distilled, 1024x1024 | |
|---|---|
| Per picture, warm | 2.2 s |
| Pictures per minute on one worker | 76 (p50 2.8 s, p95 4.1 s) |
| Most pictures per second on three B200 (two workers and the upsampler) | 2.8 (one container per replica: 1.8) |
| Cold start of a worker with a snapshot, to the first picture | 17 s |
| Cold start without one (first start after a deploy) | 4 min |
| Kept warm, per hour | $6.70 upsampler, $6.95 per worker |

`upsampler.py` is the upsampler service, `worker.py` the image worker (configuration, image build, container start, Modal server), `flux3_api.py` the request handling, `client.py` an example client, `requirements-pup.txt` the upsampler's locked environment.

## Quick start

Accept the terms on Hugging Face for the model and the prompt upsampler, then deploy the upsampler first; a worker's first start calls it and fails if it is missing:

```
uv sync
uv run modal setup
uv run modal secret create huggingface HF_TOKEN=hf_...
uv run modal deploy upsampler.py
uv run modal deploy worker.py
```

The first deploys build the images with the weights inside (19 GB and 16 GB). Check the upsampler with `uv run python -c "import modal; print(modal.Cls.from_name('flux3-upsampler', 'Upsampler')().health.remote())"`. The worker's deploy prints its URL; export it with a proxy auth token from the Modal dashboard (Settings, Proxy Auth Tokens):

```
export URL=https://<workspace>-<environment>--flux3-klein-distilled-shared-server.us-east.modal.direct
export MODAL_KEY=wk-...
export MODAL_SECRET=ws-...
```

While no worker is up, every request gets an immediate 503 and starts one. A worker whose engine call failed also answers 503 and then exits, and Modal replaces it. So poll `/wake` until it answers 200, then send `/generate` once; a non-200 means no picture was made. `client.py` does exactly that:

```
uv run python client.py $URL "Two blue mugs on a wooden table by a window" --seed 42 --out image.jpg
```

## API

`POST /generate`, JSON body. Unknown fields are rejected.

| field | default | notes |
|---|---|---|
| `prompt` | | Expanded by the upsampler. Exactly one of `prompt` and `caption` |
| `caption` | | An already expanded caption, used as is; the upsampler is not called |
| `images` | `[]` | Up to ten base64-encoded reference images, each up to 10 MB |
| `width`, `height` | 1024, or the first reference's size | Multiples of 16, up to 4096 per side and 4 megapixels |
| `seed` | `0` | Same seed and same `caption`, same picture on the same GPU model; a `prompt` is expanded afresh each time, so send back the `X-Caption` header as `caption` to reproduce a picture |
| `steps` | 16 Distilled, 50 Base | 1 to 100 |
| `guidance` | 3.5 on Base | 0 to 20. Not used by Distilled |
| `output_format` | `jpeg` | `jpeg` (quality 95) or `png` |

The response is the picture; `X-Pup-Ms`, `X-Upsampler-Ms` and `X-Image-Ms` headers give the expansion time seen by the worker, the part spent inside the upsampler, and the generation time; `X-Caption` is the expanded caption, URL-encoded. 422 is an invalid request (the body says what), 429 a worker that already holds 16 requests (retry), 502 an engine failure, 503 no worker up or the upsampler unreachable, 504 the 300 s deadline.

Editing: the reference pictures go base64-encoded in `images`. The body is too large for a command-line argument, so write it to a file:

```
printf '{"prompt": "Make the mugs red", "images": ["%s"], "output_format": "png"}' "$(base64 < image.jpg | tr -d '\n')" > edit.json
curl -sf -H "Modal-Key: $MODAL_KEY" -H "Modal-Secret: $MODAL_SECRET" --json @edit.json $URL/generate -o edit.png
```

Layouts: end the prompt with a JSON list of boxes, `[y_min, x_min, y_max, x_max]` on a 0 to 1000 canvas, origin top left:

```
{"prompt": "A blue mug on the left [{\"id\": \"mug\", \"bbox\": [200, 100, 800, 450], \"desc\": \"blue mug\"}]", "seed": 42}
```

## Cold starts and scaling

A worker stops 120 s after its last request, and the next request starts one: about 17 s with a snapshot, about four minutes without one (the first start after a deploy, or on a new kind of worker). The upsampler stays up. To keep a worker up:

```
uv run modal run worker.py --warm 1
```

A worker holds up to 16 requests and renders one picture at a time while the next one's encoding overlaps. Beyond 16, Modal queues and adds workers, up to `REPLICAS`. One upsampler fed three workers at 3.75 pictures per second without reaching its limit.

## Configuration

Environment variables read at deploy time, for example `MODEL=base REPLICAS=4 uv run modal deploy worker.py`. A redeploy with the same code and settings keeps the snapshot; changing either makes the next start a fresh one.

`upsampler.py` reads `GPU` (`B200`; H200 costs $1.70 per hour less and adds 0.4 s per expansion), `REGION` (unset; use the same value on both apps, an unpinned pair pays about 0.1 s more per picture) and `APP_NAME` (`flux3-upsampler`).

`worker.py`:

| variable | default | notes |
|---|---|---|
| `MODEL` | `distilled` | `distilled` or `base`; separate apps |
| `LORA_REPO` | unset | Base only: a Hugging Face repo with one safetensors adapter, `org/name[@revision]`. Turns off the fused FP8 kernels; not measured |
| `GPU` | `B200` | `B200`, `H200`, `H100` or `H100!`. H100 limits requests to 1536x1536 and two references |
| `WARM` | `0` | Workers kept up; `modal run worker.py --warm N` changes it without a redeploy |
| `REPLICAS` | `1` | Most workers, up to 16 |
| `IDLE_SECONDS` | `120` | How long an idle worker stays up |
| `SNAPSHOT` | `1` | `0` makes every start fresh |
| `HIGH_RES` | `0` | Distilled on B200 only: text to image up to 6256 per side, one request at a time |
| `UPSAMPLER_APP` | `flux3-upsampler` | The upsampler app, in the same environment |
| `REGION` | unset | Worker region; match the upsampler |
| `MAX_INFLIGHT` | `16` | Requests a worker holds before answering 429; lower it to add workers sooner |
| `IMAGE_SLOTS` | `2` | Requests at the image engine at once; `1` serializes encoding behind rendering |

## Measurements

Distilled FP8R, 1024x1024 JPEG, unique prompts, upsampler and workers in `us-east`, independent arrivals at the offered rate.

| fleet | offered per second | delivered per second | p50 / p95 | $ per 1,000 pictures |
|---|---|---|---|---|
| 1 worker | one client at a time | 0.43 | 2.2 / 3.0 s | |
| 1 worker | 1.5 | 1.27 | 2.8 / 4.1 s | 3.00 |
| 2 workers | 3.0 | 2.78 | 4.2 / 7.1 s | 2.06 |
| 3 workers | 4.0 | 3.75 | 3.1 / 5.0 s | 2.04 |

Three one-container replicas on the same three B200 deliver at most 1.8 pictures per second (p50 3.1 s, p95 5.2 s at 1.6 offered), because each replica's upsampler and image engine compete for the GPU. In a burst from zero at 1.5 requests per second, the first picture came after 17 s here and after 35 s with one container per replica. Cold starts, wake to first picture, medians of five probes: 17 s with a snapshot, 230 s without.

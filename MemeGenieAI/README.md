# Meme Genie AI Studio 1.2.1

## Cross-platform builds

Use Python 3.11 or newer. Windows: run `build_windows.bat`. Linux: run
`bash build_linux.sh`. macOS: run `bash build_macos.sh`. Each launcher creates
an isolated `.venv`, installs build dependencies and invokes the same `build.py`
and PyInstaller spec. Output is in `release/` (the older `dist/` is not updated).
Keep the complete `release/MemeGenieAI` folder together on Windows/Linux.
On macOS use `release/MemeGenieAI.app`. These builds are unsigned.

Build on each target operating system and architecture; PyInstaller does not
produce macOS/Linux binaries from Windows. `.github/workflows/build.yml` in the
parent directory tests and builds Windows, Linux and macOS when this source is
placed in a GitHub repository preserving the directory structure. That workflow
has not been run merely by adding the file locally.
CI packages Windows output as ZIP and Unix output as tar.gz using
`python package_build.py`, preserving executable permissions and macOS symlinks.
Download the archive from the workflow's artifacts and extract it before launch.

For manual development, create a virtual environment, install
`python -m pip install -r requirements-dev.txt`, then run `python -m pytest -q`,
`python memegenie_ai.py --smoke-test`, and `python build.py`. The smoke test opens
the real Qt window and exits automatically; it does not test image generation.
Launch normally with `run_windows.bat`, `bash run_linux.sh`, or `bash run_macos.sh`.

macOS stores models and engines in `~/Library/Application Support/MemeGenieAI`.
Linux honors `XDG_DATA_HOME`, Windows uses `LOCALAPPDATA`. Native engine automatic
installation is currently Windows x64 only. Linux and macOS users must install
a compatible native `sd-cli`, then select it with Browse; use Auto or CPU with
macOS engines. Models remain separate from application builds. GPU inference
must be tested with the user's engine, model and hardware on each platform.

### Local validation on October 19, 2026

22 tests and source/frozen Windows Qt startup checks passed with Python 3.12.
A packaging fix excludes an incompatible `icuuc.dll` discovered on the build
machine's PATH; Qt 6.11 uses Windows' ICU implementation.
The installed SD1.5 Q4 model passed SHA-256 verification, but generation is
currently unsuccessful on this machine: installed Vulkan engine commit
`e47426c` crashes with access violation `3221225477`; CPU inference fails
allocating its model parameter buffer. A separately downloaded official CPU
engine also failed. These results do not establish a GPU driver root cause.
See local `diagnostics-native/` reports. Cross-platform CI has not yet run.
Run `python diagnose_engine.py --verify --cpu-test --output-dir diagnostics`
to repeat checks; failures now produce a nonzero exit status.

**Created by Kaotick Jay / @kaotickj.** PyQt6 / Pillow meme editor and offline AI image studio.

## IMPORTANT: What is in this archive?

This is the **source + model installer + diagnostic tools** package, NOT a 3–4+ GB archive with preloaded model weights. It contains **two real model definitions**, their Hugging Face download URLs, independently published SHA-256 checksums, an installer that fetches the **actual weights**, model license retrieval, and an **offline-bundle builder that refuses to create a fake/offline model ZIP without those verified bytes**.

**The model weights and native `sd-cli` engine ARE NOT in this source ZIP.** They must be fetched once, using the installation features below, before image generation is operational. The purpose of the offline-bundle builder is to package all real downloaded weights with the app for your own offline installation or distribution, subject to licenses. No placeholder models are shipped.

The official model files are large. Downloading and testing the Vulkan implementation on a Windows RX 7600 was not possible from the build environment. Source syntax and download-manager/backend unit tests pass; **no genuine GPU generation was performed here**.

## First run, Windows 10/11

1. Extract the source ZIP and run `run_windows.bat`. This creates a virtual environment and installs Python dependencies (requires Python 3.11+). If you prefer, open a terminal and run `py -3 -m pip install -r requirements.txt` followed by `py -3 memegenie_ai.py`.
2. Open **AI Image Generation**, choose **Stable Diffusion 1.5 Q4 GGUF** in Model Catalog, click **Install Selected Model**, read and accept the license notice. The application downloads **3.05 GB** to `%LOCALAPPDATA%\MemeGenieAI\models\` and verifies SHA-256. A canceled download leaves `.part` behind so you can resume.
3. Click **Install Windows Vulkan Engine** to fetch an official `stable-diffusion.cpp` release from GitHub. The installer discovers the most recent Windows Vulkan x64 asset from GitHub's releases API and installs it under `%LOCALAPPDATA%\MemeGenieAI\engine\vulkan\`. If this fails, you can separately install an engine and use **Browse**. Model download and engine install require the internet **only at installation time**.
4. Start with **512×512**, **20 steps**, **Euler**, **CFG 7**, and **Vulkan**. If the Vulkan build is broken/unsupported on your machine, try a CPU build and select CPU. CPU generation will likely be much slower. Generation is strictly local and offline after installation.
5. Supply a prompt or choose a **Prompt Preset**. Click **Generate Image**. Every successful PNG is saved to your chosen output directory and automatically opened in the editor.
6. Switch to **Image Library** and double-click any prior generated image to edit it. Use **Export YouTube 1920×1080** or **Export Short 1080×1920** if needed; these are ordinary Pillow resizes/crops, *not* AI upscales.

For Linux, run `./run_linux.sh` and install your Vulkan-enabled `sd-cli` manually. The one-click engine installation currently only supports Windows x64. On Debian and similar environments, follow upstream's [Vulkan build guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/build.md); `cmake -DSD_VULKAN=ON` is required. Point the GUI at the resulting `sd-cli` binary.

## Included model catalog (real weight files acquired on request)

| ID | Model | Download | Published SHA-256 |
|---|---|---:|---|
| `sd15-q4` | [Stable Diffusion 1.5 Q4_0 GGUF, community quantization](https://huggingface.co/kostakoff/stable-diffusion-v1-5-GGUF) | ~3.05 GB | `24bcd54c1d1f0354a1cd19d07ad9a20771d43fde8318d505d71bcd84b078f20a` |
| `sd15-original` | [Stable Diffusion 1.5 safetensors, original checkpoint](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | ~4.27 GB | `6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa` |

**Recommended first model:** Q4_0 GGUF, for a smaller file and quantized compute footprint on the RX 7600. *This is a starting hypothesis, not a verified AMD hardware benchmark.* If your engine reports model incompatibility or black images, try the original safetensors; the Q4 conversion is community maintained. The original license applies to both models, and they have CreativeML OpenRAIL-M use restrictions. A checked model SHA-256 proves that bytes match the catalog, **not** that a third-party model is free of defects or that it will run on every GPU.

## Terminal setup, verification and offline packaging

The CLI is useful if the PyQt6 app cannot download large files or if you want a scripted/offline deployment:

```powershell
py -3 setup_models.py list
py -3 setup_models.py download --model sd15-q4
py -3 setup_models.py verify --model sd15-q4
py -3 setup_models.py license
py -3 setup_models.py engine --flavor vulkan
py -3 setup_models.py bundle --model sd15-q4 --output "C:\Users\Public\MemeGenieAI-Offline.zip"
```

The `bundle` command **only succeeds if the actual full model weight file is present and matches the published SHA-256**, the model license was downloaded, and the engine is installed. It creates a ZIP (several gigabytes) with app source, model weights, license notice and engine binaries. That ZIP is the offline-ready version. If the model/engine is absent, it fails loudly rather than producing a misleading package. Use `--no-engine` only for a deliberately incomplete custom-engine archive.

To create a Windows `.exe` from source, use `build_windows.bat` and then run the `dist\MemeGenieAI\MemeGenieAI.exe` output. The Windows build uses PyInstaller `--onedir`, not a brittle one-file binary with gigabytes of dynamic model weights. Model and engine install paths use a writable app-data folder, not the PyInstaller unpack location.

### Python requirements

Python 3.11+, PyQt6, Pillow, requests and BeautifulSoup. No PyTorch/ROCm Python dependency. Inference happens through the external native stable-diffusion.cpp engine.

### Security and licensing considerations

- **Official sources**: model weights are fetched only from HTTPS URLs in `model_catalog.json`, using pinned SHA-256 checksums. The Windows engine is resolved through the official GitHub releases API and is checked using the asset SHA-256 digest if the API provides one; otherwise HTTPS protects transit, but an independent digest check is unavailable. Upstream changes can break compatibility, so test before distributing.
- **Model licenses**: the weights are governed by **CreativeML OpenRAIL-M** and its use restrictions. `setup_models.py download` attempts to save the complete license text locally. `setup_models.py license` retries it. The offline packager **requires that license file**. Confirm downstream distribution obligations before sharing the resulting archive.
- **Engine license**: `stable-diffusion.cpp` is MIT licensed; the engine notice is under `docs/ENGINE_LICENSE.txt`. Third-party dependencies in binaries may carry additional notices.
- **Privacy**: Stable Diffusion image prompts are passed to `sd-cli` locally. Local caption models use local Ollama. Model and engine installers contact Hugging Face and GitHub when the user explicitly asks to install. The existing **Fetch Quote** action contacts `kdgwebsolutions.com` when clicked. Optional Luna caption/layout requests send the selected crop and brief to OpenAI only when selected; see below. Every image export retains mandatory GitHub attribution.
- **Source provenance**: original supplied script is preserved as `memegenie_original.py`. Do not import arbitrary `.ckpt` models from untrusted sources; prefer `.safetensors` or `.gguf`.

## Editor / generation capabilities

- Image import, live preview, caption overlay, fonts, colors, text shadows and alignment.
- Save original generated base image, edited PNG/JPEG with optional metadata, or exactly sized YouTube/Short exports.
- Text-to-image with prompt, negative prompt, sampling steps, CFG scale, seed, five sampler types, backend selector, and multiple dimension presets.
- Prompt presets for post-apocalyptic art, blues album artwork, gaming thumbnails, vertical character portraits and meme backgrounds.
- Image Library to reopen generated and exported PNG/JPEG/WebP files from the selected folder.
- Install/cancel/resume model weights, verify SHA-256, download native engine on Windows, and create large offline model bundle after verification.
- Generated files retain the model filename and generation settings in PNG metadata.

## Explicit limitations / next development tasks

- **Not yet validated end-to-end on Windows RX 7600 XT** and no downloadable model weights inside this small development archive. A true bundled-model ZIP must be built where large network downloads can run.
- **Image-to-image**, inpainting, outpainting, actual neural upscaling, LoRA, batch generation and multi-model FLUX/SDXL workflows require further development and compatible model files. The current UI supports only *text-to-image* against a single model file. The YouTube resize option does not synthesize detail.
- Installed model tracking is file-path based; verification is available in the GUI/CLI. Manually copied files are not automatically claimed to have been SHA-256 verified.
- Current Qt UI is source-validated but has not been interactively tested in a GUI-capable environment here.
- Library displays filenames, not thumbnail image previews yet. The online quote endpoint is inherited from the original program and might become unavailable.

## Source contents

```text
memegenie_ai.py           Main PyQt6 application with engine diagnostics
memegenie_original.py     Unchanged original supplied project
sd_backend.py             CLI arguments and validation
model_catalog.json        Pinned model URLs, license references, SHA-256
model_manager.py          Model download, resume, SHA verify, engine installer, offline ZIP
setup_models.py           CLI for install / verify / offline pack
diagnose_engine.py        Reproducible GPU/CPU troubleshooting tests
run_diagnostics_windows.bat  Diagnostic launcher
run_windows.bat           Easy Windows Python launcher
run_linux.sh              Easy Linux Python launcher
build_windows.bat         Build source PyInstaller onedir distribution
build_linux.sh           Build source PyInstaller onedir distribution
requirements.txt          Python dependencies
docs/ENGINE_LICENSE.txt   Upstream engine MIT notice
models/                   Empty in source archive until installed
engine/                   Empty in source archive until installed
tests/                    Pytest backend + model-manager tests
```

## A practical first test prompt

> Cinematic Chicago blues club stage at midnight, worn upright bass, saxophone in silhouette, amber spotlights, moody photo-realistic film grain, no text, wide composition

Initial settings: SD1.5 Q4 GGUF, 512×512, 20 steps, Euler, CFG 7, random seed. After you confirm the hardware path works, test wider formats and then add production-quality SDXL/FLUX models that fit your machine.

---

## v1.2.1: Engine diagnostics and Vulkan compatibility

This release adds detailed diagnostics after reports that `sd-cli` initialized the Q4_0 SD 1.5 model but exited during generation on an AMD RX 7600 XT.

- The UI saves **every engine output line** to `<image output directory>/meme_genie_ai_<timestamp>.log`, rather than relying on the 250-line scrollback from v1.2.0. The complete `sd-cli` argument list and exit code are retained. **Logs can contain prompts and local file paths.**
- Generation failure now presents the real process exit code, last 16 lines of output, and the saved log path. The AI tab has **Open diagnostic log folder** and **Copy engine command** buttons.
- **Vulkan compatibility** checkbox adds `--flash-attn off --conv-direct off` when calling current `stable-diffusion.cpp` versions. This disables optional accelerator paths to help isolate driver/shader trouble; it may be slower. It is not a universal solution.
- `run_diagnostics_windows.bat` executes reproducible 512×512 two-step tests of ordinary Vulkan and compatibility-mode Vulkan using a public, fixed prompt. Add `--cpu-test` to also run CPU inference. It writes full reports to `%LOCALAPPDATA%\MemeGenieAI\diagnostics\<timestamp>\summary.txt` and associated logs. It makes no network calls, does not alter models, and does not update installed engines.
- Run `run_diagnostics_windows.bat --verify --cpu-test` to additionally SHA-256 check your model and run the slower CPU test. If the engine/model are stored in custom paths, use `--engine "C:\full\path\sd-cli.exe" --model "C:\full\path\file.gguf"`.
- Current upstream documentation confirms `--backend vulkan0`, `--flash-attn off`, and `--conv-direct off` are supported, but your locally installed engine's actual version is authoritative. A `-v` flag is used for verbose diagnostics.

If `sd-cli` loads the backend and identifies the SD1.x checkpoint as in the reported log, there is **not yet evidence** of a model/backend format mismatch. A subsequent shader crash, GPU allocation error, failed sampling, or issue writing the output file must be distinguished from the full log or direct diagnostic runs. Reaching the auto-fit stage does not establish generation success. For model corruption check with **Verify SHA-256** or `--verify`. To isolate a conversion-specific defect, repeat the same test with the catalog's original `.safetensors` model, after obtaining it.

Technical references: https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/backend.md and https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/cli/README.md

## Optional OpenAI Luna captions and layouts

Choose **OpenAI · gpt-6-luna (online)** in the existing caption model selector,
then use **AI: Suggest memes** or **Recheck image** as usual. Local model choices
and the default remain unchanged. Luna creates editable captions, protected
subject regions and layouts for the selected background. Stable Diffusion image
generation and manual editing/export controls remain unchanged.

Provide `OPENAI_API_KEY` in the app process environment, or place only the key in
`~/.memegenie-ai/openai.key` outside the repository. Create its directory privately
(`mkdir -p ~/.memegenie-ai; chmod 700 ~/.memegenie-ai`) and enter the key with a
trusted editor; restrict the file with `chmod 600 ~/.memegenie-ai/openai.key`.
The file must be owned by you on Unix. On Windows, restrict its ACL to your own
account. `MEMEGENIE_OPENAI_KEY_FILE` can point to another private file. The
environment key takes precedence. Never put keys in source, launch scripts,
QSettings, logs, screenshots, or packaged files. The default private file works
with the desktop launcher without changing its environment/test profile.

Restart the development app to load the changes. The desktop entry
`~/Desktop/memegenie-ai-dev-test.desktop` runs
`~/.local/share/memegenie-ai-dev-test/launch.py`, using the virtual environment
and source at `~/git/meme-genie-ai-dev/MemeGenieAI`.

Only an explicitly selected Luna suggestion/recheck sends a JPEG of the current
crop (at most 768 pixels per side) and the brief to OpenAI. API charges apply.
Local mode never reads an OpenAI key or falls back to cloud. Luna makes one
request per click, with no automatic retries or fallback. Recheck sends a fresh
request. This integration does not use the local scene cache or its separate
caption-rewrite calls. Exact quoted copy, placement checks, subject review,
Apply/Cancel, and mandatory GitHub credit still use the existing editor code.

The adapter uses the existing `requests` dependency and Responses REST API with
strict JSON schema, `store: false`, output-token and response-size limits,
timeouts, and sanitized errors. `store: false` does not promise zero provider
retention. Refusals, incomplete results, missing keys, network failures and
model-access errors leave current editor content unchanged. No SDK installation
is needed; no SDK was installed in this app environment. Keys are excluded from
the explicit source bundle lists.

Verified against official documentation on 2026-11-15:
[models](https://developers.openai.com/api/docs/models),
[structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs),
[image input](https://developers.openai.com/api/docs/guides/images-vision), and
[Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create).
Tests use mocked HTTP and synthetic images only. No paid API request, user-image
upload, live caption-quality comparison, or account model-access verification
was performed.

## Optional OpenAI background images

Open **Tools → Experimental Image Generation**, then select
**OpenAI · gpt-image-2.6-flare (online)** in the new image provider selector.
The tab remains available through Tools, as before. Local Stable Diffusion is
still selected at every launch, with its models and settings intact. Image and
caption provider choices are independent; selecting this option does not change
Luna/local caption generation.

Use the existing prompt preset or write a prompt, set the resolution and click
**Generate Image**. The existing negative prompt is appended as an avoidance
instruction (the Image API has no separate negative-prompt parameter).
Local-only steps, CFG, seed, sampler, backend and compatibility fields are
inactive for OpenAI and retain their values when switching back. Local model
installation controls remain available.

The request generates exactly one PNG using `gpt-image-2.6-flare`, medium quality
and medium complexity. Square canvases request 1024×1024; landscape requests
1536×1024; portrait requests 1024×1536. The result is center-cropped/resized
locally to the selected resolution, so some edges can be cropped. No existing
background or personal image is sent; this path uses only the prompt text and
calls `/v1/images/generations`, not an image-edit endpoint.

It shares the Luna key configuration described above. Requests incur OpenAI API
charges and are never automatically retried or sent as a local fallback. The UI
links to current model pricing; cost depends on input/output usage. Cancellation
discards the result and prevents saving/applying it, but cannot guarantee server
cancellation or avoid charges. A request may take up to the network timeout to
return; closing waits for the worker to finish. Missing keys, account/model
access restrictions, rejected prompts, rate limits and network errors appear in
the existing status area or settings dialog without exposing response bodies or
credentials. No engine diagnostic file is created for OpenAI requests.

Successful generation opens the clean source in memory for editing and saves a
PNG with mandatory GitHub attribution in the chosen image folder. Save an
editable project to retain that clean source with your layers. Reopening the
credited gallery PNG imports the credit as part of its pixels; later exports
still add mandatory attribution. Saved generation metadata records the provider,
model, prompt, negative prompt, quality, complexity and API canvas, never a key.
A save failure leaves the current editor image unchanged.

Official references checked 2026-11-15:
[GPT Image 2.6 Flare](https://developers.openai.com/api/docs/models/gpt-image-2.6-flare),
[Image generation API](https://developers.openai.com/api/reference/resources/images/methods/generate),
[image guide](https://developers.openai.com/api/docs/guides/image-generation).
This implementation was tested with mocked HTTP and synthetic images only.
Account access, live visual quality and actual billing were not tested.

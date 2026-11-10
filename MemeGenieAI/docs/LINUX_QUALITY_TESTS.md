# Linux development quality checks

Development branch only. Native engine and editor/export functionality work; this is not a claim that generated artwork is publication-ready.

## Reproduce the quality setup

Launch **Meme Genie AI - Development Test** from the desktop. The launcher runs this checkout, not the older editor or the frozen package.

- Model Catalog: **Stable Diffusion 1.5 original checkpoint** (`sd15-original`). Confirm the model path ends in `v1-5-pruned-emaonly.safetensors`, not the Q4 GGUF.
- Compute backend: **CPU**.
- Resolution: **512 x 512**.
- Steps: **25**, CFG: **7**, sampler: **Euler**.
- Use a fixed seed for comparisons; these checks used **2142989708**.

The development launcher now preserves quality controls across launches. It has a separate settings profile. A one-time requested CPU-mode migration preserves the previous backend; later manual backend choices are respected. Existing full and Q4 model files have not been moved or deleted.

The model catalog previously failed to find sibling checkpoints beside a manually browsed model on another drive. It now checks that folder first. Choosing the full model should update the actual model-path field. Missing models are not downloaded automatically.

## Measured results

On this machine (Ryzen 5 5600X, six CPU threads), the full checkpoint took approximately 192 seconds for the original blues-band prompt and 196 seconds for an empty-club variant. Both completed normally with all compute modules assigned to CPU. Same-seed full-checkpoint CPU and Vulkan results had closely similar composition and defects.

The full checkpoint improved clarity relative to Q4 in one controlled same-seed comparison, but hands, instruments, and multi-person scene composition remained unreliable. Merely changing from 12 to 25 steps is not a complete fix. A prompt requesting an empty historical club still invented musicians; do not claim this prompt reliably excludes people.

The full-checkpoint Vulkan comparison peaked at **98 C junction/hotspot**. It completed below the test's 100 C cutoff, but that leaves insufficient margin for unattended sustained generation. CPU is the tested heat-avoidance option; its speed cost is substantial.

This engine does not expose a llama-style GPU-layer count. Per-module CPU placement and weight offloading exist, but leaving diffusion on GPU still leaves the hottest stage on GPU. VRAM budgets are not temperature controls.

## Validation scope

- 24 automated tests passed, including sibling-model catalog selection.
- Qt editor loading, caption rendering, PNG/JPEG saving, 1920x1080 and 1080x1920 exports, canceled exports, and source-image preservation passed isolated checks.
- Real Qt Generate control to native engine to automatic editor preview passed offscreen.
- Actual installed full-model selection persisted across reopening in an isolated Qt test.
- User's normal settings and old editor were not changed by tests.

Quality evidence is under `diagnostics-linux/quality-cpu/`; logs include exact prompts, seeds, settings, placement, and times. Models remain separate from the app package. The frozen package predates the model-selection fix; use the development launcher for this change. No merge or release has been performed.

## A usable constrained preset

**Bass instrument detail photograph** is now an additional preset, leaving existing presets intact. With the full checkpoint, CPU, 25 steps, CFG7, Euler and seed2142989708, it produced a substantially cleaner single-instrument detail in195 seconds. The instrument is cropped at the top; this is not a verified full-length product image. Strings/bridge still warrant review. It is useful as a background/detail starting point, not evidence that crowded band scenes are solved. The preset excludes people in the negative prompt.

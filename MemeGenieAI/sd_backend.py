"""Validated, GUI-independent sd-cli request generation for Meme Genie AI.

Targets a locally installed stable-diffusion.cpp sd-cli executable.
Images, prompts, and settings remain on the local machine.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationRequest:
    executable: str
    model: str
    output: str
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = 42
    sampler: str = "euler"
    backend: str = "auto"
    compatibility_mode: bool = False


SUPPORTED_SAMPLERS = frozenset({"euler", "euler_a", "heun", "dpm2", "dpm++2m"})
SUPPORTED_BACKENDS = frozenset({"auto", "cpu", "vulkan0"})


def build_sd_cli_arguments(req: GenerationRequest) -> list[str]:
    """Build argv (without executable). No shell interpolation is involved.

    For models requiring separate text encoders/VAEs (notably many FLUX models),
    this first prototype intentionally only supports single-file checkpoints.
    """
    if not Path(req.executable).is_file():
        raise ValueError("Select an existing sd-cli executable.")
    if not Path(req.model).is_file():
        raise ValueError("Select an existing local model checkpoint.")
    if Path(req.model).suffix.lower() not in {".safetensors", ".ckpt", ".gguf", ".pth"}:
        raise ValueError("Model must be a .safetensors, .ckpt, .gguf, or .pth file.")
    if not req.prompt.strip():
        raise ValueError("Enter an AI image prompt.")
    if not (256 <= req.width <= 2048 and req.width % 64 == 0):
        raise ValueError("Width must be 256–2048 in multiples of 64.")
    if not (256 <= req.height <= 2048 and req.height % 64 == 0):
        raise ValueError("Height must be 256–2048 in multiples of 64.")
    if not 1 <= req.steps <= 150:
        raise ValueError("Steps must be 1–150.")
    if not 0.0 <= req.cfg_scale <= 30.0:
        raise ValueError("CFG must be 0–30.")
    if not 0 <= req.seed <= 2147483647:
        raise ValueError("Seed must be 0–2,147,483,647.")
    if req.sampler not in SUPPORTED_SAMPLERS:
        raise ValueError("Unsupported sampler.")
    if req.backend not in SUPPORTED_BACKENDS:
        raise ValueError("Unsupported backend.")
    output = Path(req.output)
    if output.suffix.lower() != ".png" or not output.parent.is_dir():
        raise ValueError("Output must be a PNG in an existing directory.")

    args = [
        "-m", req.model,
        "-p", req.prompt.strip(),
        "-W", str(req.width),
        "-H", str(req.height),
        "--steps", str(req.steps),
        "--cfg-scale", str(req.cfg_scale),
        "--seed", str(req.seed),
        "--sampling-method", req.sampler,
        "-o", str(output),
    ]
    if req.negative_prompt.strip():
        args += ["-n", req.negative_prompt.strip()]
    if req.backend != "auto":
        args += ["--backend", req.backend]
    if req.compatibility_mode:
        # Disable optional compute optimizations to isolate GPU/driver shader bugs.
        # Documented by stable-diffusion.cpp; may be slower than automatic selection.
        args += ["--flash-attn", "off", "--conv-direct", "off"]
    return args

"""Run local sd-cli diagnostics without a UI, network requests, or model modifications.

The report contains local paths, the diagnostic prompt, and engine stdout/stderr.
Only share it if you're comfortable sharing those details.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import platform
import subprocess
import sys

from model_manager import app_data_dir, locate_engine, engine_dir, model_dir, models, resolve_model, verify_model
from sd_backend import GenerationRequest, build_sd_cli_arguments


def run_command(command: list[str], cwd: Path, log: Path, timeout: int) -> int | None:
    with log.open("w", encoding="utf-8", errors="replace") as out:
        out.write("Command: " + subprocess.list2cmdline(command) + "\n\n")
        out.flush()
        try:
            result = subprocess.run(command, cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT,
                                    timeout=timeout, check=False)
            out.write(f"\n[EXIT] return code = {result.returncode}\n")
            return result.returncode
        except subprocess.TimeoutExpired:
            out.write(f"\n[TIMEOUT] exceeded {timeout} seconds\n")
            return None
        except OSError as exc:
            out.write(f"\n[LAUNCH ERROR] {exc}\n")
            return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose the installed Meme Genie sd-cli engine")
    parser.add_argument("--engine", type=Path, help="Path to sd-cli.exe; normally auto detected")
    parser.add_argument("--model", type=Path, help="Path to SD 1.5 GGUF or safetensors; auto detected")
    parser.add_argument("--cpu-test", action="store_true", help="Also attempt slow CPU inference")
    parser.add_argument("--output-dir", type=Path, help="Directory for diagnostic reports")
    parser.add_argument("--verify", action="store_true", help="Verify known model's full SHA-256 checksum")
    parser.add_argument("--timeout", type=int, default=360, help="Seconds allowed per generation test")
    args = parser.parse_args(argv)

    engine = args.engine or locate_engine(engine_dir())
    model = args.model
    if model is None:
        model = next((p for m in models() if (p := m.installed_path(model_dir())).is_file()), None)

    if engine is None or not Path(engine).is_file():
        parser.error("Cannot find sd-cli. Use --engine to set its absolute path.")
    if model is None or not Path(model).is_file():
        parser.error("Cannot find a checkpoint. Use --model to set its absolute path.")
    if args.timeout < 15:
        parser.error("--timeout must be at least 15 seconds")

    engine = Path(engine).resolve()
    model = Path(model).resolve()
    folder = (args.output_dir or app_data_dir() / "diagnostics") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    summary = ["Meme Genie AI Studio engine diagnostic", f"Python: {sys.version.split()[0]}",
               f"OS: {platform.platform()}", f"Engine: {engine}", f"Model: {model}",
               f"Reports: {folder}", ""]

    if args.verify:
        found = next((m for m in models() if m.filename == model.name), None)
        if found is None:
            summary.append("Checksum: model not in bundled catalog; skipped")
        else:
            ok = verify_model(found, model.parent)
            summary.append("Checksum: " + ("PASS" if ok else "FAILED; do not trust this model file"))
            if not ok:
                print("\n".join(summary))
                (folder / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
                return 2

    for label, args_list in (("version", ["--version"]), ("devices", ["--list-devices"])):
        code = run_command([str(engine), *args_list], engine.parent, folder / (label + ".log"), timeout=30)
        summary.append(f"{label}: return code {code}; see {label}.log")

    # Deliberately fixed public diagnostic prompt, not the user's private prompts.
    base = dict(executable=str(engine), model=str(model), prompt="a red apple on a wooden table",
                negative_prompt="", width=512, height=512, steps=2, cfg_scale=7.0, seed=42,
                sampler="euler")
    cases = [("vulkan_standard", "vulkan0", False),
             ("vulkan_compat", "vulkan0", True)]
    if args.cpu_test:
        cases.append(("cpu", "cpu", False))
    all_passed = True
    for label, backend, compat in cases:
        output = folder / (label + ".png")
        request = GenerationRequest(**base, backend=backend, compatibility_mode=compat,
                                    output=str(output))
        cmd = [str(engine), *build_sd_cli_arguments(request), "-v"]
        print(f"Running {label}...", flush=True)
        code = run_command(cmd, engine.parent, folder / (label + ".log"), timeout=args.timeout)
        passed = code == 0 and output.is_file() and output.stat().st_size > 0
        all_passed = all_passed and passed
        summary.append(f"{label}: {'PASS' if passed else 'FAIL'} (return code {code}); "
                       f"image {'created' if output.is_file() else 'missing'}; see {label}.log")

    summary.extend(["", "Interpretation:",
                    "- Standard Vulkan fails, compatibility Vulkan succeeds: optional Vulkan compute optimization likely involved.",
                    "- Both Vulkan tests fail, CPU succeeds: GPU driver/runtime/backend more likely than model syntax.",
                    "- Vulkan and CPU both fail: inspect log tails; check model integrity, CLI revision, and model/runtime support.",
                    "- A pass on two-step tests does not guarantee longer or higher-resolution generation will pass.",
                    "- Keep any error output after 'auto-fit plan' for troubleshooting."])
    (folder / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary), flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""CLI installation, verification and assembly of a *real* offline model archive."""
import argparse
import sys
from pathlib import Path

from model_manager import (
    models, resolve_model, download_model, verify_model, install_engine,
    locate_engine, create_offline_bundle, model_dir, engine_dir, retrieve_model_license,
)


def show_progress(current, total, status):
    if total:
        print(f"\r{status}: {current / 1024**2:.1f}/{total / 1024**2:.1f} MiB", end="", flush=True)
    else:
        print(f"\r{status}: {current / 1024**2:.1f} MiB", end="", flush=True)


def main() -> int:
    cli = argparse.ArgumentParser(description="Meme Genie AI model and offline pack manager")
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for action in ("download", "verify", "bundle"):
        p = sub.add_parser(action)
        p.add_argument("--model", default="sd15-q4", choices=[m.id for m in models()])
        if action == "bundle":
            p.add_argument("--output", required=True)
            p.add_argument("--no-engine", action="store_true")
    sub.add_parser("license")
    p_engine = sub.add_parser("engine")
    p_engine.add_argument("--flavor", default="vulkan", choices=("vulkan", "cpu"))
    args = cli.parse_args()
    try:
        if args.command == "list":
            print("Model catalog: installed weights are never faked or stubbed")
            for model in models():
                present = model.installed_path().is_file()
                print(f"{model.id}: {model.display_name} [{ 'present, not yet verified' if present else 'not downloaded' }]\n  {model.model_card}")
            print("Model directory:", model_dir())
            print("Engine directory:", engine_dir())
        elif args.command == "download":
            info = resolve_model(args.model)
            print("Installing:", info.display_name)
            print("License:", info.license_url)
            path = download_model(info, progress=show_progress)
            print("\nVerified model:", path)
            try:
                print("License saved:", retrieve_model_license())
            except Exception as exc:
                print("Warning: license fetch failed; offline redistribution is blocked until fixed:", exc)
        elif args.command == "verify":
            info = resolve_model(args.model)
            ok = verify_model(info, progress=show_progress)
            print("\nModel SHA-256:", "VALID" if ok else "MISSING OR INVALID")
            return 0 if ok else 1
        elif args.command == "license":
            print("License saved:", retrieve_model_license())
        elif args.command == "engine":
            print("Downloading official engine release...")
            path = install_engine(args.flavor, progress=show_progress)
            print("\nEngine:", path)
        elif args.command == "bundle":
            print("Checking model SHA-256 and preparing true offline archive (multi-GB)...")
            path = create_offline_bundle(Path(args.output).expanduser(),
                                         require_engine=not args.no_engine,
                                         selected_ids=(args.model,), progress=show_progress)
            print("\nReal model weights + engine + application packaged into:", path)
        return 0
    except (OSError, ValueError, RuntimeError, InterruptedError) as exc:
        print("\nERROR:", exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

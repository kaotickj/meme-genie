from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from sd_backend import GenerationRequest, build_sd_cli_arguments


class TestSdBackend(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.exe = root / "sd-cli"
        self.exe.write_text("fake exe")
        self.model = root / "model.safetensors"
        self.model.write_text("fake model")
        self.out = root / "out.png"
        self.request = GenerationRequest(
            executable=str(self.exe), model=str(self.model), output=str(self.out),
            prompt="A wasteland skyline with ravens", seed=12345,
        )

    def test_valid_basic(self):
        args = build_sd_cli_arguments(self.request)
        self.assertEqual(args[args.index("-p") + 1], "A wasteland skyline with ravens")
        self.assertEqual(args[args.index("--seed") + 1], "12345")
        self.assertNotIn("--backend", args)
        self.assertNotIn("-n", args)

    def test_negative_prompt_and_vulkan(self):
        req = replace(self.request, negative_prompt="text, watermark", backend="vulkan0")
        args = build_sd_cli_arguments(req)
        self.assertEqual(args[args.index("-n") + 1], "text, watermark")
        self.assertEqual(args[args.index("--backend") + 1], "vulkan0")

    def test_bad_resolution(self):
        with self.assertRaises(ValueError):
            build_sd_cli_arguments(replace(self.request, width=777))

    def test_empty_prompt(self):
        with self.assertRaises(ValueError):
            build_sd_cli_arguments(replace(self.request, prompt="  "))

    def test_bad_model(self):
        with self.assertRaises(ValueError):
            build_sd_cli_arguments(replace(self.request, model=str(self.exe)))

    def test_executable_missing(self):
        with self.assertRaises(ValueError):
            build_sd_cli_arguments(replace(self.request, executable="/missing/none"))

    def test_bad_sampler(self):
        with self.assertRaises(ValueError):
            build_sd_cli_arguments(replace(self.request, sampler="insecure invalid"))

    def test_outside_dir(self):
        with self.assertRaises(ValueError):
            build_sd_cli_arguments(replace(self.request, output=str(Path(self.temp.name) / "missing" / "foo.png")))

    def test_compatibility_mode_disables_optional_acceleration(self):
        args = build_sd_cli_arguments(replace(self.request, compatibility_mode=True, backend="vulkan0"))
        self.assertEqual(args[args.index("--flash-attn") + 1], "off")
        self.assertEqual(args[args.index("--conv-direct") + 1], "off")
        self.assertEqual(args[args.index("--backend") + 1], "vulkan0")

    def test_standard_mode_omits_compatibility_flags(self):
        args = build_sd_cli_arguments(self.request)
        self.assertNotIn("--flash-attn", args)
        self.assertNotIn("--conv-direct", args)

    def test_backend_cpu(self):
        args = build_sd_cli_arguments(replace(self.request, backend="cpu"))
        self.assertEqual(args[args.index("--backend")+1], "cpu")


if __name__ == "__main__":
    unittest.main()

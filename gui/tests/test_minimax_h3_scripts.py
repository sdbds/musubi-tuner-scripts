import re
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestMiniMaxH3Scripts(unittest.TestCase):
    CACHE = ROOT / "2.11minimax_h3_cache_latent_and_text_encoder.ps1"
    TRAIN = ROOT / "3.11minimax_h3_train_lora.ps1"
    GENERATE = ROOT / "5.11minimax_h3_generate.ps1"

    def read_script(self, path: Path) -> str:
        self.assertTrue(path.is_file(), f"Script not found: {path}")
        return path.read_text(encoding="utf-8")

    def test_scripts_exist(self):
        for path in (self.CACHE, self.TRAIN, self.GENERATE):
            with self.subTest(script=path.name):
                self.assertTrue(path.is_file(), f"Script not found: {path}")

    def test_shared_defaults_match_released_components(self):
        for path in (self.CACHE, self.TRAIN, self.GENERATE):
            script = self.read_script(path)
            with self.subTest(script=path.name):
                self.assertIn('$task = "t2va"', script)
                self.assertIn(
                    '$video_vae = "./ckpts/vae/minimax_h3_video_vae_fp16.safetensors"',
                    script,
                )
                self.assertIn(
                    '$audio_vae = "./ckpts/vae/minimax_h3_audio_vae_fp32.safetensors"',
                    script,
                )
                self.assertIn(
                    '$text_encoder = "./ckpts/text_encoder/qwen3vl_32b_minimax_h3_bf16.safetensors"',
                    script,
                )

        for path in (self.CACHE, self.TRAIN):
            self.assertIn(
                '$dataset_config = "./toml/qinglong-video-datasets.toml"',
                self.read_script(path),
            )

    def test_cache_runs_both_h3_cache_entry_points(self):
        cache = self.read_script(self.CACHE)

        self.assertIn("minimax_h3_cache_latents.py", cache)
        self.assertIn("minimax_h3_cache_text_encoder_outputs.py", cache)
        for flag in (
            "--dataset_config=$dataset_config",
            "--task=$task",
            "--video_vae=$video_vae",
            "--audio_vae=$audio_vae",
            "--cache_seed=$cache_seed",
            "--text_encoder=$text_encoder",
            "--text_cache_dtype=$text_cache_dtype",
        ):
            self.assertIn(flag, cache)

    def test_cache_exposes_one_frame_and_unconditional_text_contract(self):
        cache = self.read_script(self.CACHE)

        for declaration in (
            "$one_frame = $False",
            '$uncond_output = ""',
            '$uncond_text = ""',
        ):
            self.assertIn(declaration, cache)

        one_frame_block = cache.split("if ($one_frame)", 1)[1].split(
            "if ($uncond_output", 1
        )[0]
        self.assertEqual(one_frame_block.count('.Add("--one_frame")'), 2)

        uncond_block = cache.split('if ($uncond_output -ne "")', 1)[1].split(
            'python "./musubi-tuner/$text_script"', 1
        )[0]
        self.assertIn('.Add("--uncond_output=$uncond_output")', uncond_block)
        self.assertIn('if ($uncond_text -ne "")', uncond_block)
        self.assertIn('.Add("--uncond_text=$uncond_text")', uncond_block)

    def test_training_uses_h3_lora_bf16_shifts_and_sampling_defaults(self):
        train = self.read_script(self.TRAIN)

        self.assertIn("minimax_h3_train_network.py", train)
        self.assertIn('$network_module = "networks.lora_minimax_h3"', train)
        self.assertIn('$mixed_precision = "bf16"', train)
        self.assertIn('$dit_dtype = "bfloat16"', train)
        self.assertIn("--h3_shift_video=$h3_shift_video", train)
        self.assertIn("--h3_shift_audio=$h3_shift_audio", train)
        self.assertIn("$h3_shift_video = 12", train)
        self.assertIn("$h3_shift_audio = 3", train)
        self.assertIn("$enable_sample = $True", train)
        self.assertIn("$sample_at_first = $True", train)
        self.assertIn(
            '$sample_prompts = "./toml/qinglong_minimaxh3.txt"',
            train,
        )

        sampling_block = train.split("if ($enable_sample)", 1)[1].split(
            "# Metadata", 1
        )[0]
        for flag in (
            "--sample_at_first",
            "--sample_prompts=$sample_prompts",
            "--video_vae=$video_vae",
            "--audio_vae=$audio_vae",
            "--text_encoder=$text_encoder",
        ):
            self.assertIn(flag, sampling_block)
        for dependency_flag in (
            "--video_vae=$video_vae",
            "--audio_vae=$audio_vae",
            "--text_encoder=$text_encoder",
        ):
            self.assertEqual(train.count(dependency_flag), 1, dependency_flag)

    def test_training_exposes_one_frame_and_guidance_contract(self):
        train = self.read_script(self.TRAIN)

        for declaration in (
            "$one_frame = $False",
            "$video_only = $False",
            "$audio_loss_weight = 1.0",
            "$h3_guidance_loss_scale = 0.0",
            '$h3_guidance_loss_scale_audio = ""',
            "$h3_guidance_loss_sigma_min = 0.0",
            '$h3_guidance_loss_uncond_cache = ""',
        ):
            self.assertIn(declaration, train)

        for flag in (
            '--one_frame',
            '--video_only',
            '--audio_loss_weight=$audio_loss_weight',
            '--h3_guidance_loss_scale=$h3_guidance_loss_scale',
            '--h3_guidance_loss_scale_audio=$h3_guidance_loss_scale_audio',
            '--h3_guidance_loss_sigma_min=$h3_guidance_loss_sigma_min',
            '--h3_guidance_loss_uncond_cache=$h3_guidance_loss_uncond_cache',
        ):
            self.assertIn(flag, train)

        audio_scale_block = train.split(
            'if ($h3_guidance_loss_scale_audio -ne "")', 1
        )[1].split("}", 1)[0]
        self.assertIn(
            '.Add("--h3_guidance_loss_scale_audio=$h3_guidance_loss_scale_audio")',
            audio_scale_block,
        )

    def test_generation_routes_task_specific_inputs_and_uses_output(self):
        generate = self.read_script(self.GENERATE)

        self.assertIn("minimax_h3_generate_video.py", generate)
        self.assertIn('$first_frame = ""', generate)
        self.assertIn('$last_frame = ""', generate)
        self.assertIn('$reference_jsonl = ""', generate)
        self.assertIn("$reference_index = 0", generate)
        self.assertIn('if ($task -ieq "fl2va")', generate)
        self.assertIn('elseif ($task -ieq "ref2va")', generate)
        self.assertIn("if (($frame_count - 5) % 17 -ne 0)", generate)
        self.assertIn("$duration_seconds = $frame_count / 24.0", generate)
        self.assertNotIn(
            "if (-not $allow_experimental_duration -and (($frame_count - 5) % 17 -ne 0))",
            generate,
        )
        for flag in (
            "--first_frame=$first_frame",
            "--last_frame=$last_frame",
            "--reference_jsonl=$reference_jsonl",
            "--reference_index=$reference_index",
            "--h3_shift_video=$h3_shift_video",
            "--h3_shift_audio=$h3_shift_audio",
            "--output=$output",
        ):
            self.assertIn(flag, generate)
        generation_call = generate.split('python "./musubi-tuner/$script"', 1)[1].split(
            "Assert-NativeCommandSucceeded", 1
        )[0]
        self.assertNotIn("--save_path=", generation_call)

    def test_every_python_call_is_guarded_by_native_command_helper(self):
        python_line = re.compile(r"^\s*python(?:\s|$)", re.MULTILINE)
        guard_line = re.compile(
            r"^\s*Assert-NativeCommandSucceeded\b", re.MULTILINE
        )

        for path in (self.CACHE, self.TRAIN, self.GENERATE):
            script = self.read_script(path)
            with self.subTest(script=path.name):
                self.assertIn("powershell/native_command.ps1", script)
                python_calls = len(python_line.findall(script))
                self.assertGreater(python_calls, 0)
                self.assertEqual(python_calls, len(guard_line.findall(script)))

    def test_scripts_parse_with_powershell_ast(self):
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if not pwsh:
            self.skipTest("PowerShell is unavailable")

        for path in (self.CACHE, self.TRAIN, self.GENERATE):
            self.assertTrue(path.is_file(), f"Script not found: {path}")
            command = (
                "$tokens=$null; $errors=$null; "
                f"[System.Management.Automation.Language.Parser]::ParseFile('{path}',"
                "[ref]$tokens,[ref]$errors) | Out-Null; "
                "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }"
            )
            with self.subTest(script=path.name):
                result = subprocess.run(
                    [pwsh, "-NoProfile", "-NonInteractive", "-Command", command],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

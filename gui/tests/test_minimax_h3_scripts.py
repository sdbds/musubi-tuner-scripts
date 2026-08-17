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
    BEST_OF_K_HELPER = ROOT / "powershell" / "minimax_h3_best_of_k.ps1"
    TRAIN_DEFAULTS_HELPER = ROOT / "powershell" / "minimax_h3_train_defaults.ps1"

    def read_script(self, path: Path) -> str:
        self.assertTrue(path.is_file(), f"Script not found: {path}")
        return path.read_text(encoding="utf-8")

    def test_scripts_exist(self):
        for path in (
            self.CACHE,
            self.TRAIN,
            self.GENERATE,
            self.BEST_OF_K_HELPER,
            self.TRAIN_DEFAULTS_HELPER,
        ):
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
        self.assertIn('$task -notin @("t2va", "fl2va")', cache)
        self.assertNotIn('$one_frame -and $task -ine "t2va"', cache)

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
        self.assertIn('$task -notin @("t2va", "fl2va")', train)
        self.assertNotIn('$one_frame -and $task -ine "t2va"', train)

        audio_scale_block = train.split(
            'if ($h3_guidance_loss_scale_audio -ne "")', 1
        )[1].split("}", 1)[0]
        self.assertIn(
            '.Add("--h3_guidance_loss_scale_audio=$h3_guidance_loss_scale_audio")',
            audio_scale_block,
        )

    def test_training_exposes_best_of_k_contract(self):
        train = self.read_script(self.TRAIN)

        for declaration in (
            "$h3_best_of_k = 1",
            '$h3_best_of_k_stream = "video"',
        ):
            self.assertIn(declaration, train)
        for source_fragment in (
            "powershell/minimax_h3_best_of_k.ps1",
            "Resolve-H3BestOfKCount",
            "Resolve-H3BestOfKStream",
            "Assert-NoH3BestOfKReservedArguments",
            "Assert-H3BestOfKArgumentInvariant",
        ):
            self.assertIn(source_fragment, train)
        for flag in (
            "--h3_best_of_k=$h3_best_of_k",
            "--h3_best_of_k_stream=$h3_best_of_k_stream",
        ):
            self.assertEqual(train.count(flag), 1, flag)
        self.assertIn(
            '$h3_best_of_k -gt 1 -and $h3_best_of_k_stream -eq "audio" -and $video_only',
            train,
        )
        self.assertLess(
            train.index("Resolve-H3BestOfKCount"),
            train.index('if ($env:OS -ilike "*windows*")'),
        )
        self.assertLess(
            train.index("Assert-H3BestOfKArgumentInvariant"),
            train.index('Write-Output "Extended arguments:"'),
        )

    def test_best_of_k_count_helper_enforces_strict_integer_contract(self):
        success_cases = (
            ("[int]1", "1"),
            ("[long]2", "2"),
            ("[string]'3'", "3"),
        )
        for expression, expected in success_cases:
            with self.subTest(expression=expression):
                result = self.run_best_of_k_helper(
                    f"Resolve-H3BestOfKCount ({expression})"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

        invalid_expressions = (
            "[double]1.0",
            "[double]1.5",
            "0",
            "-1",
            "$true",
            "''",
            "'1e0'",
            "'word'",
        )
        for expression in invalid_expressions:
            with self.subTest(expression=expression):
                result = self.run_best_of_k_helper(
                    f"Resolve-H3BestOfKCount ({expression})"
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("base-10 integer", result.stderr)

    def test_best_of_k_stream_helper_normalizes_and_rejects_values(self):
        for expression, expected in (("' VIDEO '", "video"), ("'audio'", "audio")):
            with self.subTest(expression=expression):
                result = self.run_best_of_k_helper(
                    f"Resolve-H3BestOfKStream ({expression})"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

        for expression in ("''", "'music'", "[int]1"):
            with self.subTest(expression=expression):
                result = self.run_best_of_k_helper(
                    f"Resolve-H3BestOfKStream ({expression})"
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("video or audio", result.stderr)

    def test_best_of_k_helper_rejects_reserved_raw_options_in_both_forms(self):
        cases = (
            "--h3_best_of_k 8",
            "--h3_best_of_k=8",
            "--h3_best_of_k_stream audio",
            "--h3_best_of_k_stream=audio",
            "--xm_best_of_k 8",
            "--xm_best_of_k=8",
        )
        for value in cases:
            escaped = value.replace("'", "''")
            with self.subTest(value=value):
                result = self.run_best_of_k_helper(
                    f"Assert-NoH3BestOfKReservedArguments '{escaped}'"
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("reserved", result.stderr)

    def test_best_of_k_helper_enforces_final_argument_invariant(self):
        for arguments in (
            "@()",
            "@('--h3_best_of_k=2', '--h3_best_of_k_stream=video')",
        ):
            with self.subTest(arguments=arguments):
                result = self.run_best_of_k_helper(
                    f"Assert-H3BestOfKArgumentInvariant {arguments}"
                )
                self.assertEqual(result.returncode, 0, result.stderr)

        invalid_argument_lists = (
            "@('--h3_best_of_k_stream=video')",
            "@('--h3_best_of_k=2', '--h3_best_of_k=3', '--h3_best_of_k_stream=video')",
            "@('--h3_best_of_k=2', '--h3_best_of_k_stream=video', '--xm_best_of_k=2')",
        )
        for arguments in invalid_argument_lists:
            with self.subTest(arguments=arguments):
                result = self.run_best_of_k_helper(
                    f"Assert-H3BestOfKArgumentInvariant {arguments}"
                )
                self.assertNotEqual(result.returncode, 0)

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

    def test_train_default_helper_compares_numeric_values_exactly(self):
        for expression in ("12", "[double]12.0", "'12.000'"):
            with self.subTest(expression=expression):
                result = self.run_train_defaults_helper(
                    f"Test-H3NumericDefault ({expression}) ([decimal]12)"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "True")

        nearby = self.run_train_defaults_helper(
            "Test-H3NumericDefault ([decimal]0.9991) ([decimal]0.999)"
        )
        self.assertEqual(nearby.returncode, 0, nearby.stderr)
        self.assertEqual(nearby.stdout.strip(), "False")

        for expression in ("$true", "''", "'word'", "'NaN'"):
            with self.subTest(expression=expression):
                result = self.run_train_defaults_helper(
                    f"Test-H3NumericDefault ({expression}) ([decimal]1)"
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("finite number", result.stderr)

    def test_training_guards_default_sensitive_arguments(self):
        train = self.read_script(self.TRAIN)
        guarded_fragments = (
            'if ($h3_best_of_k -ne 1 -or $h3_best_of_k_stream -cne "video")',
            'if ($network_module -cne "networks.lora_minimax_h3")',
            'Test-H3NumericDefault $h3_shift_video ([decimal]12',
            'Test-H3NumericDefault $h3_shift_audio ([decimal]3',
            'Test-H3NumericDefault $h3_visual_cond_clean ([decimal]0.999',
            'Test-H3NumericDefault $h3_audio_cond_clean ([decimal]1',
            'Test-H3NumericDefault $audio_loss_weight ([decimal]1',
            'Test-H3NumericDefault $h3_guidance_loss_scale ([decimal]0',
            'Test-H3NumericDefault $h3_guidance_loss_sigma_min ([decimal]0',
            'Test-H3NumericDefault $lr ([decimal]0.000002',
            'Test-H3NumericDefault $max_train_steps ([decimal]1600',
            'Test-H3NumericDefault $network_alpha ([decimal]1',
            'Test-H3NumericDefault $gradient_accumulation_steps ([decimal]1',
            'Test-H3NumericDefault $lr_scheduler_num_cycles ([decimal]1',
            'Test-H3NumericDefault $max_data_loader_n_workers ([decimal]8',
            'Test-H3NumericDefault $max_grad_norm ([decimal]1',
        )
        for fragment in guarded_fragments:
            self.assertIn(fragment, train)

        for removed in (
            "--h3_video_best_of_k",
            "--h3_audio_best_of_k",
            "--h3_image_best_of_k",
        ):
            self.assertNotIn(removed, train)

        invocation = train.split(
            "python -m accelerate.commands.launch", 1
        )[1].split("Assert-NativeCommandSucceeded", 1)[0]
        for option in (
            "--network_module=",
            "--timestep_sampling=",
            "--discrete_flow_shift=",
            "--weighting_scheme=",
            "--h3_shift_video=",
            "--h3_shift_audio=",
            "--h3_visual_cond_clean=",
            "--h3_audio_cond_clean=",
            "--learning_rate=",
        ):
            self.assertNotIn(option, invocation)

    def test_scripts_parse_with_powershell_ast(self):
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if not pwsh:
            self.skipTest("PowerShell is unavailable")

        for path in (
            self.CACHE,
            self.TRAIN,
            self.GENERATE,
            self.BEST_OF_K_HELPER,
            self.TRAIN_DEFAULTS_HELPER,
        ):
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

    def run_best_of_k_helper(self, body: str) -> subprocess.CompletedProcess[str]:
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if not pwsh:
            self.skipTest("PowerShell is unavailable")
        helper = str(self.BEST_OF_K_HELPER).replace("'", "''")
        return subprocess.run(
            [
                pwsh,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                f". '{helper}'; {body}",
            ],
            capture_output=True,
            encoding="utf-8",
        )

    def run_train_defaults_helper(self, body: str) -> subprocess.CompletedProcess[str]:
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if not pwsh:
            self.skipTest("PowerShell is unavailable")
        helper = str(self.TRAIN_DEFAULTS_HELPER).replace("'", "''")
        return subprocess.run(
            [
                pwsh,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                f". '{helper}'; {body}",
            ],
            capture_output=True,
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()

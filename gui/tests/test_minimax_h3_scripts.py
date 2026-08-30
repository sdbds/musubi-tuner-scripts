import json
import re
import shutil
import subprocess
import tempfile
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
        self.assertIn("$output_fps = 24", generate)
        self.assertIn("$stretch_keep_bands = 0", generate)
        self.assertIn("$duration_seconds = $frame_count / [double]$output_fps", generate)
        self.assertIn("($output_fps -lt 1) -or ($output_fps -gt 24)", generate)
        self.assertIn("($stretch_keep_bands -lt 0) -or ($stretch_keep_bands -gt 15)", generate)
        self.assertIn("$stretch_keep_bands -gt 0 -and $output_fps -eq 24", generate)
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
            "--output_fps=$output_fps",
            "--stretch_keep_bands=$stretch_keep_bands",
            "--output=$output",
        ):
            self.assertIn(flag, generate)
        generation_call = generate.split('python "./musubi-tuner/$script"', 1)[1].split(
            "Assert-NativeCommandSucceeded", 1
        )[0]
        self.assertNotIn("--save_path=", generation_call)

    def test_generation_rejects_non_integer_temporal_stretch_values_before_python(self):
        cases = (
            ({"output_fps": "[double]12.5"}, "output_fps"),
            (
                {
                    "output_fps": "12",
                    "stretch_keep_bands": "[double]3.5",
                },
                "stretch_keep_bands",
            ),
        )
        for overrides, field in cases:
            with self.subTest(overrides=overrides):
                result, launched = self.run_generate_script(overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(field, result.stderr)
                self.assertIn("base-10 integer", result.stderr)
                self.assertFalse(launched)

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

    def test_training_appends_raw_optimizer_arguments_after_structured_defaults(self):
        cases = (
            (
                "structured nondefault and raw default",
                {"optimizer_args": '"--learning_rate=0.000002"'},
                ["--learning_rate=1e-4", "--learning_rate=0.000002"],
            ),
            (
                "structured default and raw nondefault",
                {
                    "lr": '"0.000002"',
                    "optimizer_args": '"--learning_rate=0.000003"',
                },
                ["--learning_rate=0.000003"],
            ),
            (
                "duplicate nondefault raw overrides",
                {
                    "lr": '"0.000004"',
                    "optimizer_args": (
                        '"--learning_rate=0.000003;--learning_rate=0.000005"'
                    ),
                },
                [
                    "--learning_rate=0.000004",
                    "--learning_rate=0.000003",
                    "--learning_rate=0.000005",
                ],
            ),
        )
        for name, overrides, expected in cases:
            with self.subTest(case=name):
                result, activated, payload = self.run_train_script(overrides)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(activated)
                self.assertIsNotNone(payload)
                learning_rates = [
                    argument
                    for argument in payload["arguments"]
                    if argument.startswith("--learning_rate=")
                ]
                self.assertEqual(learning_rates, expected)
                self.assertEqual(payload["arguments"][-1], expected[-1])

    def test_training_rejects_invalid_numeric_values_before_activation(self):
        for overrides in (
            {"h3_shift_video": '"not-a-number"'},
            {"lr": '"not-a-number"'},
        ):
            with self.subTest(overrides=overrides):
                result, activated, _ = self.run_train_script(overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("finite number", result.stderr)
                self.assertFalse(activated)

    def test_training_activates_before_mocked_python_launch(self):
        result, activated, payload = self.run_train_script({})

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(activated)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["events"], ["activate", "python"])

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

    def run_train_script(
        self, overrides: dict[str, str]
    ) -> tuple[subprocess.CompletedProcess[str], bool, dict[str, object] | None]:
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if not pwsh:
            self.skipTest("PowerShell is unavailable")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train = self.read_script(self.TRAIN)
            for name, value in overrides.items():
                train, replacements = re.subn(
                    rf"(?m)^\${re.escape(name)}\s*=\s*.*$",
                    f"${name} = {value}",
                    train,
                )
                self.assertEqual(replacements, 1, name)

            train_path = root / self.TRAIN.name
            train_path.write_text(train, encoding="utf-8")
            helpers = root / "powershell"
            helpers.mkdir()
            for helper in (
                "native_command.ps1",
                "minimax_h3_best_of_k.ps1",
                "minimax_h3_train_defaults.ps1",
            ):
                shutil.copy2(ROOT / "powershell" / helper, helpers / helper)

            activation = root / "venv" / "bin"
            activation.mkdir(parents=True)
            (activation / "activate").write_text("", encoding="utf-8")
            marker = root / "activation.marker"
            escaped_marker = str(marker).replace("'", "''")
            (activation / "Activate.ps1").write_text(
                "\n".join(
                    (
                        "$global:H3TestEvents.Add('activate')",
                        f"Set-Content -LiteralPath '{escaped_marker}' -Value 'activated'",
                    )
                ),
                encoding="utf-8",
            )

            escaped_train = str(train_path).replace("'", "''")
            wrapper = root / "run.ps1"
            wrapper.write_text(
                "\n".join(
                    (
                        '$env:OS = ""',
                        "$global:H3TestArguments = @()",
                        "$global:H3TestEvents = [System.Collections.Generic.List[string]]::new()",
                        "function global:python {",
                        '    $global:H3TestEvents.Add("python")',
                        "    $global:H3TestArguments = @(foreach ($argument in $args) {",
                        "        if ($argument -is [System.Collections.IEnumerable] -and $argument -isnot [string]) {",
                        "            $argument",
                        "        }",
                        "        else {",
                        "            $argument",
                        "        }",
                        "    })",
                        "    $global:LASTEXITCODE = 0",
                        "}",
                        "function global:Read-Host { return \"\" }",
                        f"& '{escaped_train}'",
                        "$payload = @{",
                        "    events = @($global:H3TestEvents)",
                        "    arguments = @($global:H3TestArguments)",
                        "}",
                        'Write-Output ("H3_TEST_RESULT:" + ($payload | ConvertTo-Json -Compress))',
                    )
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    pwsh,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(wrapper),
                ],
                capture_output=True,
                encoding="utf-8",
            )
            payload = None
            for line in reversed(result.stdout.splitlines()):
                if line.startswith("H3_TEST_RESULT:"):
                    payload = json.loads(line.removeprefix("H3_TEST_RESULT:"))
                    break
            return result, marker.is_file(), payload

    def run_generate_script(
        self, overrides: dict[str, str]
    ) -> tuple[subprocess.CompletedProcess[str], bool]:
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if not pwsh:
            self.skipTest("PowerShell is unavailable")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            generate = self.read_script(self.GENERATE)
            for name, value in overrides.items():
                generate, replacements = re.subn(
                    rf"(?m)^\${re.escape(name)}\s*=\s*.*$",
                    f"${name} = {value}",
                    generate,
                    count=1,
                )
                self.assertEqual(replacements, 1, name)

            generate_path = root / self.GENERATE.name
            generate_path.write_text(generate, encoding="utf-8")
            helpers = root / "powershell"
            helpers.mkdir()
            shutil.copy2(ROOT / "powershell" / "native_command.ps1", helpers)

            marker = root / "python.marker"
            escaped_marker = str(marker).replace("'", "''")
            escaped_generate = str(generate_path).replace("'", "''")
            wrapper = root / "run.ps1"
            wrapper.write_text(
                "\n".join(
                    (
                        '$env:OS = ""',
                        "function global:python {",
                        f"    Set-Content -LiteralPath '{escaped_marker}' -Value 'launched'",
                        "    $global:LASTEXITCODE = 0",
                        "}",
                        'function global:Read-Host { return "" }',
                        f"& '{escaped_generate}'",
                    )
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    pwsh,
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(wrapper),
                ],
                capture_output=True,
                encoding="utf-8",
            )
            return result, marker.is_file()


if __name__ == "__main__":
    unittest.main()

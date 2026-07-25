import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestMageFlowScripts(unittest.TestCase):
    CACHE = ROOT / "2.10mage_flow_cache_latent_and_text_encoder.ps1"
    TRAIN = ROOT / "3.10mage_flow_train_lora.ps1"
    GENERATE = ROOT / "5.10mage_flow_generate.ps1"

    def test_scripts_expose_current_upstream_contract(self):
        cache = self.CACHE.read_text(encoding="utf-8")
        train = self.TRAIN.read_text(encoding="utf-8")
        generate = self.GENERATE.read_text(encoding="utf-8")

        self.assertEqual(cache.count('Add("--is_edit")'), 2)
        self.assertIn("--seed=$cache_seed", cache)
        self.assertIn("mage_flow_cache_latents.py", cache)
        self.assertIn("mage_flow_cache_text_encoder_outputs.py", cache)

        self.assertIn('$network_module = "musubi_tuner.networks.lora_mage_flow"', train)
        self.assertIn("--discrete_flow_shift=$discrete_flow_shift", train)
        self.assertIn("--weighting_scheme=$weighting_scheme", train)
        self.assertIn("--allow_mage_architecture_mismatch", train)
        self.assertIn("$enable_sample -and -not $sample_prompts", train)
        self.assertNotIn("$include_patterns", train)
        self.assertNotIn("$exclude_patterns", train)

        self.assertIn("mage_flow_generate_image.py", generate)
        self.assertIn("--output=$mage_output_path", generate)
        self.assertIn("--control_image=", generate)
        self.assertNotIn("--save_path=", generate)
        self.assertIn(
            "$loraMultipliers.Count -gt $loraWeights.Count",
            generate,
        )

        combined = cache + train + generate
        self.assertNotIn("--processor", combined)
        self.assertNotIn("--tokenizer", combined)

    def test_scripts_parse_with_powershell_ast(self):
        pwsh = shutil.which("pwsh") or shutil.which("powershell")
        if not pwsh:
            self.skipTest("PowerShell is unavailable")
        for path in (self.CACHE, self.TRAIN, self.GENERATE):
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

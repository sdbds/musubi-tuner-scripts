import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PWSH = shutil.which("pwsh") or shutil.which("powershell")


@unittest.skipUnless(PWSH, "PowerShell is required for installer behavior tests")
class TestImageInstallDownloads(unittest.TestCase):
    def run_menu(self, root, menu, choice, transport="success"):
        script_path = str(ROOT / "1.install-uv-qinglong.ps1").replace("'", "''")
        # Load only real functions and the selected top-level menu, never setup/install code.
        command = r"""
$ErrorActionPreference = 'Stop'
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('__SCRIPT__', [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
$ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $false) |
    ForEach-Object { Invoke-Expression $_.Extent.Text }
$script:choice = '__CHOICE__'
$script:transport = '__TRANSPORT__'
function Read-Host { param($Prompt) return $script:choice }
function hf {
    if ($args.Count -ne 5 -or $args[0] -ne 'download' -or $args[3] -ne '--local-dir') {
        throw "Unexpected hf arguments: $args"
    }
    if ($script:transport -eq 'fail') { $global:LASTEXITCODE = 23; return }
    $global:LASTEXITCODE = 0
    if ($script:transport -eq 'missing') { return }
    if ($script:transport -eq 'forbid') { throw 'Existing components must not be downloaded again' }
    $path = Join-Path $args[4] $args[2]
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $path) | Out-Null
    Set-Content -LiteralPath $path -Value "$($args[1])/$($args[2])" -NoNewline
    Add-Content -LiteralPath 'downloads.jsonl' -Value (ConvertTo-Json -Compress @($args[1], $args[2]))
}
$statements = @($ast.EndBlock.Statements)
$start = -1
for ($i = 0; $i -lt $statements.Count; $i++) {
    $node = $statements[$i]
    if ($node -is [System.Management.Automation.Language.AssignmentStatementAst] -and
        $node.Left -is [System.Management.Automation.Language.VariableExpressionAst] -and
        $node.Left.VariablePath.UserPath -eq '__MENU__') { $start = $i; break }
}
if ($start -lt 0) { throw 'Requested installer menu is missing' }
for ($i = $start; $i -le $start + 2; $i++) { Invoke-Expression $statements[$i].Extent.Text }
"""
        command = (
            command.replace("__SCRIPT__", script_path)
            .replace("__MENU__", menu)
            .replace("__CHOICE__", choice)
            .replace("__TRANSPORT__", transport)
        )
        return subprocess.run(
            [PWSH, "-NoProfile", "-NonInteractive", "-Command", command],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )

    def assert_downloads(self, menu, choice, expected):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_menu(root, menu, choice)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            files = {p.relative_to(root / "ckpts").as_posix(): p.read_text() for p in (root / "ckpts").rglob("*.safetensors")}
            self.assertEqual(files, expected)
            downloads = [json.loads(line) for line in (root / "downloads.jsonl").read_text().splitlines()]
            self.assertEqual(len(downloads), len(expected), "Shared components should only download once")
            result = self.run_menu(root, menu, choice, "forbid")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(
                files, {p.relative_to(root / "ckpts").as_posix(): p.read_text() for p in (root / "ckpts").rglob("*.safetensors")}
            )

    def test_qwen_image_21_choices_download_matching_bf16_components(self):
        shared = {
            "text_encoder/qwen3vl_8b_bf16.safetensors": "Comfy-Org/Qwen-Image-2.1/text_encoders/qwen3vl_8b_bf16.safetensors",
            "vae/qwen_image_2.1_vae_bf16.safetensors": "Comfy-Org/Qwen-Image-2.1/vae/qwen_image_2.1_vae_bf16.safetensors",
        }
        standard = {
            "diffusion_models/qwen_image_2.1_bf16.safetensors": (
                "Comfy-Org/Qwen-Image-2.1/diffusion_models/qwen_image_2.1_bf16.safetensors"
            )
        }
        turbo = {
            "diffusion_models/qwen_image_2.1_turbo_bf16.safetensors": (
                "Comfy-Org/Qwen-Image-2.1/diffusion_models/qwen_image_2.1_turbo_bf16.safetensors"
            )
        }
        for choice, diffusion in (("1", standard), ("2", turbo), ("3", standard | turbo)):
            with self.subTest(choice=choice):
                self.assert_downloads("download_qwen_image_21", choice, shared | diffusion)

    def test_moss_image_choices_download_single_bf16_text_encoder(self):
        shared = {
            "text_encoder/qwen3.5_9b_bf16.safetensors": "Comfy-Org/Qwen3.5/text_encoders/qwen3.5_9b_bf16.safetensors",
            "vae/flux2-vae.safetensors": "Comfy-Org/MOSS-Image/vae/flux2-vae.safetensors",
        }
        base = {
            "diffusion_models/moss_image_base_bf16.safetensors": (
                "Comfy-Org/MOSS-Image/diffusion_models/moss_image_base_bf16.safetensors"
            )
        }
        sft = {
            "diffusion_models/moss_image_sft_bf16.safetensors": (
                "Comfy-Org/MOSS-Image/diffusion_models/moss_image_sft_bf16.safetensors"
            )
        }
        for choice, diffusion in (("1", base), ("2", sft), ("3", base | sft)):
            with self.subTest(choice=choice):
                self.assert_downloads("download_moss_image", choice, shared | diffusion)

    def test_image_menus_skip_by_default_or_explicit_choice(self):
        for menu in ("download_qwen_image_21", "download_moss_image"):
            for choice in ("", "n", "N", "invalid"):
                with self.subTest(menu=menu, choice=choice), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    result = self.run_menu(root, menu, choice, "forbid")
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertFalse((root / "ckpts").exists())

    def test_image_menus_propagate_transport_failure_and_missing_output(self):
        for menu in ("download_qwen_image_21", "download_moss_image"):
            for transport, code in (("fail", 23), ("missing", 1)):
                with self.subTest(menu=menu, transport=transport), tempfile.TemporaryDirectory() as directory:
                    result = self.run_menu(Path(directory), menu, "1", transport)
                    self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                    self.assertIn("Install failed", result.stdout)


if __name__ == "__main__":
    unittest.main()

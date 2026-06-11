import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestInstallScriptDownloads(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.install_script = (ROOT / "1.install-uv-qinglong.ps1").read_text(encoding="utf-8")

    def test_lens_download_prompt_matches_gui_default_paths(self):
        script = self.install_script

        self.assertIn("function DownloadLensModel", script)
        self.assertIn("$download_lens = Read-Host", script)
        self.assertIn('hf download $RepoId $FilePath --local-dir $LocalDir', script)

        for expected in (
            '@{ RepoId = "Comfy-Org/Lens"; FilePath = "diffusion_models/lens_bf16.safetensors" }',
            '@{ RepoId = "Comfy-Org/Lens"; FilePath = "text_encoders/gpt_oss_20b_nvfp4.safetensors"; TargetPath = "text_encoder/gpt_oss_20b_nvfp4.safetensors" }',
            '@{ RepoId = "Comfy-Org/Lens"; FilePath = "vae/flux2-vae.safetensors" }',
        ):
            self.assertIn(expected, script)

        for omitted in (
            '@{ RepoId = "microsoft/Lens"; FilePath = "text_encoder/config.json" }',
            '@{ RepoId = "microsoft/Lens"; FilePath = "text_encoder/generation_config.json" }',
            '@{ RepoId = "microsoft/Lens"; FilePath = "tokenizer/chat_template.jinja" }',
            '@{ RepoId = "microsoft/Lens"; FilePath = "tokenizer/tokenizer.json" }',
            '@{ RepoId = "microsoft/Lens"; FilePath = "tokenizer/tokenizer_config.json" }',
        ):
            self.assertNotIn(omitted, script)

        self.assertIn('$lensRoot = "./ckpts"', script)
        self.assertNotIn('$lensRoot = "./ckpts/lens"', script)

    def test_ideogram4_download_prompt_uses_shared_component_layout(self):
        script = self.install_script

        self.assertIn("function DownloadIdeogram4Model", script)
        self.assertIn("$download_ideogram4 = Read-Host", script)
        self.assertIn('$ideogram4Root = "./ckpts"', script)
        self.assertNotIn('$ideogram4Root = "./ckpts/ideogram4"', script)
        self.assertIn("DownloadIdeogram4Qwen3Vl8BBf16TextEncoder", script)
        self.assertIn("Comfy-Org/Qwen3-VL", script)
        self.assertIn("text_encoders/qwen3vl_8b_bf16.safetensors", script)
        self.assertIn('-TargetPath "text_encoder/qwen3vl_8b_bf16.safetensors"', script)
        self.assertNotIn("qwen3vl_8b_fp8_scaled.safetensors", script)

        ideogram_block = script.split("function DownloadIdeogram4Model", 1)[1].split("$download_lens", 1)[0]
        self.assertIn("DownloadIdeogram4Qwen3Vl8BBf16TextEncoder", ideogram_block)
        self.assertNotIn("DownloadFlux2KleinQwen3TextEncoder8B", ideogram_block)
        self.assertNotIn("qwen_3_8b.safetensors", ideogram_block)

        for expected in (
            '@{ RepoId = "Comfy-Org/Ideogram-4"; FilePath = "diffusion_models/ideogram4_fp8_scaled.safetensors" }',
            '@{ RepoId = "Comfy-Org/Ideogram-4"; FilePath = "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors" }',
            '@{ RepoId = "Comfy-Org/Ideogram-4"; FilePath = "vae/flux2-vae.safetensors" }',
        ):
            self.assertIn(expected, script)


if __name__ == "__main__":
    unittest.main()

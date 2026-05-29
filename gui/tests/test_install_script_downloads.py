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
            '@{ RepoId = "Comfy-Org/Lens"; FilePath = "text_encoders/gpt_oss_20b_nvfp4.safetensors" }',
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

        self.assertIn('$lensRoot = "./ckpts/lens"', script)


if __name__ == "__main__":
    unittest.main()

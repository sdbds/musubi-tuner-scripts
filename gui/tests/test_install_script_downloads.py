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
            '@{ RepoId = "Comfy-Org/Ideogram-4"; FilePath = "vae/flux2-vae.safetensors" }',
        ):
            self.assertIn(expected, script)
        self.assertNotIn("ideogram4_unconditional_fp8_scaled.safetensors", script)

    def test_mage_flow_download_prompt_exposes_only_supported_bf16_components(self):
        script = self.install_script
        self.assertIn("function DownloadMageFlowModel", script)
        self.assertIn("$download_mage_flow = Read-Host", script)
        self.assertIn('$mageFlowRoot = "./ckpts"', script)

        for expected in (
            "diffusion_models/mage_flow_bf16.safetensors",
            "diffusion_models/mage_flow_turbo_bf16.safetensors",
            "diffusion_models/mage_flow_edit_bf16.safetensors",
            "diffusion_models/mage_flow_edit_turbo_bf16.safetensors",
            "vae/mage_flow_vae_bf16.safetensors",
        ):
            self.assertIn(expected, script)

        mage_block = script.split("function DownloadMageFlowModel", 1)[1].split(
            "function DownloadMiniMaxH3Model", 1
        )[0]
        self.assertIn('-RepoId "Comfy-Org/Mage-Flow"', mage_block)
        self.assertIn("DownloadQwenVl4BReweightTextEncoder", mage_block)
        self.assertNotIn("qwen3vl_4b_bf16.safetensors", mage_block)
        self.assertNotIn("int8_convrot", mage_block.lower())
        self.assertNotIn("processor", mage_block.lower())
        self.assertNotIn("tokenizer", mage_block.lower())

    def test_minimax_h3_download_menu_selects_base_and_reuses_shared_components(self):
        script = self.install_script

        self.assertIn("function DownloadMiniMaxH3Model", script)
        self.assertIn("$download_minimax_h3 = Read-Host", script)
        self.assertIn('$miniMaxH3Root = "./ckpts"', script)

        function_block = script.split("function DownloadMiniMaxH3Model", 1)[1].split(
            "function DownloadLensModel", 1
        )[0]
        self.assertIn('-RepoId "Comfy-Org/MiniMax-H3"', function_block)
        self.assertIn("foreach ($filePath in $DiffusionFiles)", function_block)
        self.assertIn("[hashtable]$TextEncoder", function_block)
        self.assertIn("if ($null -ne $TextEncoder)", function_block)
        self.assertIn("-RepoId $TextEncoder.RepoId", function_block)
        self.assertIn("-FilePath $TextEncoder.FilePath", function_block)
        self.assertIn("-TargetPath $TextEncoder.TargetPath", function_block)

        shared_components = (
            "vae/minimax_h3_video_vae_fp16.safetensors",
            "vae/minimax_h3_audio_vae_fp32.safetensors",
        )
        for component in shared_components:
            self.assertEqual(function_block.count(component), 1, component)

        menu_block = script.split("$download_minimax_h3 = Read-Host", 1)[1].split(
            "$download_lens", 1
        )[0]
        self.assertIn("FL2VA/T2VA", menu_block)
        self.assertIn("Ref2VA", menu_block)
        self.assertIn("BF16", menu_block)
        self.assertIn("INT8 ConvRot", menu_block)
        self.assertIn("Download all BF16 and INT8 models", menu_block)
        self.assertIn("Skip download", menu_block)
        self.assertIn(
            '"1" { @("diffusion_models/minimax_h3_fl2va_bf16.safetensors") }',
            menu_block,
        )
        self.assertIn(
            '"2" { @("diffusion_models/minimax_h3_ref2va_bf16.safetensors") }',
            menu_block,
        )
        self.assertIn(
            '"4" { @("diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors") }',
            menu_block,
        )
        self.assertIn(
            '"5" { @("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors") }',
            menu_block,
        )
        self.assertIn(
            '$download_minimax_h3_text_encoder = Read-Host',
            menu_block,
        )
        self.assertIn("[1/2/3/n] (默认为 2)", menu_block)
        self.assertIn("[1/2/3/n] (default 2)", menu_block)
        self.assertEqual(menu_block.count("$download_minimax_h3_text_encoder = Read-Host"), 1)

        selected_dit_guard = "if ($miniMaxH3DiffusionFiles.Count -gt 0)"
        self.assertLess(menu_block.index(selected_dit_guard), menu_block.index("$download_minimax_h3_text_encoder"))

        text_encoder_switch = menu_block.split("$miniMaxH3TextEncoder = switch", 1)[1].split(
            "DownloadMiniMaxH3Model", 1
        )[0]
        official_bf16 = (
            '@{ RepoId = "Comfy-Org/MiniMax-H3"; '
            'FilePath = "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors"; '
            'TargetPath = "text_encoder/qwen3vl_32b_minimax_h3_bf16.safetensors" }'
        )
        official_int8 = (
            '@{ RepoId = "Comfy-Org/MiniMax-H3"; '
            'FilePath = "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"; '
            'TargetPath = "text_encoder/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" }'
        )
        heretic_int8 = (
            '@{ RepoId = "ethanfel/Qwen3-VL-32B-Ultra-Heretic-H3-ComfyUI-INT8-ConvRot"; '
            'FilePath = "qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors"; '
            'TargetPath = "text_encoder/qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors" }'
        )
        self.assertIn(f'"1" {{ {official_bf16} }}', text_encoder_switch)
        self.assertIn(f'"2" {{ {official_int8} }}', text_encoder_switch)
        self.assertIn(f'"3" {{ {heretic_int8} }}', text_encoder_switch)
        self.assertIn('"n" { $null }', text_encoder_switch)
        self.assertIn(f"default {{ {official_int8} }}", text_encoder_switch)
        self.assertIn(
            "DownloadMiniMaxH3Model -DiffusionFiles $miniMaxH3DiffusionFiles -TextEncoder $miniMaxH3TextEncoder",
            menu_block,
        )
        self.assertNotIn("qwen3vl_32b_h3_generation_tail_50_63_int8_convrot.safetensors", script)

        for diffusion_model in (
            "diffusion_models/minimax_h3_fl2va_bf16.safetensors",
            "diffusion_models/minimax_h3_ref2va_bf16.safetensors",
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        ):
            self.assertGreaterEqual(menu_block.count(diffusion_model), 2, diffusion_model)


if __name__ == "__main__":
    unittest.main()

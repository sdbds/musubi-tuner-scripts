# Mage-Flow Scripts And GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first-class Mage-Flow T2I/Edit workflow covering supported BF16 model downloads, cache, LoRA training, generation, GUI configuration, presets, and documentation.

**Architecture:** Keep cache and training inside the existing shared GUI command-building path, with Mage-specific validation and an explicit `is_edit` flag. Put mode/variant recommendations in the model catalog, and branch generation to a bespoke builder because Mage-Flow's CLI uses a file output, explicit width/height, and repeated ordered control images. Keep the three PowerShell entry points independently usable and derive GUI presets from their editable variables.

**Tech Stack:** PowerShell 7/Windows PowerShell, Python 3.10+, NiceGUI, `unittest`/pytest, Hugging Face CLI, musubi-tuner CLI modules, Playwright.

## Global Constraints

- Target the checked-in `musubi-tuner` submodule at commit `3962a5a` on branch `qinglong`.
- Use `Comfy-Org/Mage-Flow` and expose only the four BF16 DiT files plus the shared BF16 VAE and Qwen3-VL text encoder.
- Do not expose ComfyUI INT8 ConvRot checkpoints, full-model fine-tuning, Base checkpoints, native packing, or unsupported attention backends.
- Do not emit `--processor` or `--tokenizer`; current upstream resolves pinned processor assets automatically.
- Keep T2I/Edit identity explicit through `is_edit` in cache, training, generation, scripts, GUI state, and presets.
- Edit generation requires one to three ordered reference images; T2I generation rejects every control image.
- The only supported attention modes are SDPA and optional FlashAttention 2.
- Mage-Flow training uses the fixed network module `musubi_tuner.networks.lora_mage_flow`.
- Mage-Flow training defaults are BF16, `timestep_sampling=shift`, `discrete_flow_shift=6`, and `weighting_scheme=none`.
- Mage-Flow generation recommendations are T2I Standard 20/5, T2I Turbo 4/1, Edit Standard 30/5, and Edit Turbo 4/1 for steps/CFG.
- Generation must use `--output`, never the generic `--save_path`.
- Automated verification must not download large checkpoints or claim real-weight parity.
- Follow TDD for behavior changes, use the existing repository style, and do not modify or stage unrelated user files.

---

### Task 1: Add The Mage-Flow Catalog And Recommendation Profiles

**Files:**
- Modify: `gui/utils/model_catalog.py:32-680`
- Modify: `gui/utils/model_catalog.py:687-820`
- Modify: `gui/utils/i18n.py:93-110`
- Modify: `gui/utils/i18n.py:723-740`
- Modify: `gui/utils/i18n.py:1348-1365`
- Modify: `gui/utils/i18n.py:1973-1990`
- Modify: `gui/tests/test_model_catalog.py:30-250`

**Interfaces:**
- Produces: `MAGE_FLOW_ARCH: str = "Mage-Flow"`.
- Produces: `get_mage_flow_profile(is_edit: bool, variant: str) -> Dict[str, Any]`.
- Produces: profiles containing `dit_path`, `steps`, `cfg_scale`, `width`, `height`, and `max_size`.
- Consumed by: command builders and all three wizard pages in later tasks.

- [ ] **Step 1: Write failing catalog and profile tests**

Add these methods to `TestModelCatalog`:

```python
def test_mage_flow_catalog_exposes_current_entry_points_and_component_paths(self):
    mage = self.catalog.get_architecture("Mage-Flow")
    self.assertEqual(mage["id"], "mage_flow")
    self.assertEqual(mage["versions"], ["standard", "turbo"])
    self.assertEqual(mage["cache_module"], "musubi_tuner.mage_flow_cache_latents")
    self.assertEqual(mage["cache_te_module"], "musubi_tuner.mage_flow_cache_text_encoder_outputs")
    self.assertEqual(mage["train_module"], "musubi_tuner.mage_flow_train_network")
    self.assertEqual(mage["generate_module"], "musubi_tuner.mage_flow_generate_image")
    self.assertEqual(mage["pages"]["cache"]["required_paths"], ["vae", "text_encoder"])
    self.assertEqual(mage["pages"]["train"]["required_paths"], ["dit"])
    self.assertEqual(mage["pages"]["generate"]["required_paths"], ["dit", "vae", "text_encoder"])

    cache = self.catalog.get_path_defaults("Mage-Flow", "cache", version="standard")
    self.assertEqual(cache["vae_path"], "./ckpts/vae/mage_flow_vae_bf16.safetensors")
    self.assertEqual(cache["text_encoder_path"], "./ckpts/text_encoder/qwen3vl_4b_bf16.safetensors")
    self.assertNotIn("processor_path", cache)
    self.assertNotIn("tokenizer_path", cache)


def test_mage_flow_profiles_cover_mode_and_variant_recommendations(self):
    expected = {
        (False, "standard"): ("mage_flow_bf16.safetensors", 20, 5.0, 1024, 1024, None),
        (False, "turbo"): ("mage_flow_turbo_bf16.safetensors", 4, 1.0, 1024, 1024, None),
        (True, "standard"): ("mage_flow_edit_bf16.safetensors", 30, 5.0, None, None, 1024),
        (True, "turbo"): ("mage_flow_edit_turbo_bf16.safetensors", 4, 1.0, None, None, 1024),
    }
    for key, values in expected.items():
        with self.subTest(profile=key):
            profile = self.catalog.get_mage_flow_profile(*key)
            filename, steps, cfg, width, height, max_size = values
            self.assertTrue(profile["dit_path"].endswith(filename))
            self.assertEqual(profile["steps"], steps)
            self.assertEqual(profile["cfg_scale"], cfg)
            self.assertEqual(profile["width"], width)
            self.assertEqual(profile["height"], height)
            self.assertEqual(profile["max_size"], max_size)

    with self.assertRaises(ValueError):
        self.catalog.get_mage_flow_profile(False, "base")
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_model_catalog.py -k mage_flow -q
```

Expected: FAIL because `Mage-Flow` and `get_mage_flow_profile` do not exist.

- [ ] **Step 3: Implement catalog profiles and page capabilities**

Add these constants before `MODEL_CATALOG`:

```python
MAGE_FLOW_ARCH = "Mage-Flow"
MAGE_FLOW_PROFILES: Dict[tuple[bool, str], Dict[str, Any]] = {
    (False, "standard"): {
        "dit_path": "./ckpts/diffusion_models/mage_flow_bf16.safetensors",
        "steps": 20,
        "cfg_scale": 5.0,
        "width": 1024,
        "height": 1024,
        "max_size": None,
    },
    (False, "turbo"): {
        "dit_path": "./ckpts/diffusion_models/mage_flow_turbo_bf16.safetensors",
        "steps": 4,
        "cfg_scale": 1.0,
        "width": 1024,
        "height": 1024,
        "max_size": None,
    },
    (True, "standard"): {
        "dit_path": "./ckpts/diffusion_models/mage_flow_edit_bf16.safetensors",
        "steps": 30,
        "cfg_scale": 5.0,
        "width": None,
        "height": None,
        "max_size": 1024,
    },
    (True, "turbo"): {
        "dit_path": "./ckpts/diffusion_models/mage_flow_edit_turbo_bf16.safetensors",
        "steps": 4,
        "cfg_scale": 1.0,
        "width": None,
        "height": None,
        "max_size": 1024,
    },
}
```

Add a `Mage-Flow` catalog entry with:

```python
MAGE_FLOW_ARCH: {
    "id": "mage_flow",
    "cache_module": "musubi_tuner.mage_flow_cache_latents",
    "cache_te_module": "musubi_tuner.mage_flow_cache_text_encoder_outputs",
    "train_module": "musubi_tuner.mage_flow_train_network",
    "generate_module": "musubi_tuner.mage_flow_generate_image",
    "versions": ["standard", "turbo"],
    "defaults": {
        "cache": {"version": "standard"},
        "train": {"version": "standard", "train_mode": "lora"},
        "generate": {"version": "standard"},
    },
    "path_defaults": {
        "cache": {
            "common": {
                "vae_path": "./ckpts/vae/mage_flow_vae_bf16.safetensors",
                "text_encoder_path": "./ckpts/text_encoder/qwen3vl_4b_bf16.safetensors",
            },
        },
        "train": {
            "common": {
                "vae_path": "./ckpts/vae/mage_flow_vae_bf16.safetensors",
                "text_encoder_path": "./ckpts/text_encoder/qwen3vl_4b_bf16.safetensors",
            },
            "versions": {
                "standard": {"dit_path": MAGE_FLOW_PROFILES[(False, "standard")]["dit_path"]},
                "turbo": {"dit_path": MAGE_FLOW_PROFILES[(False, "turbo")]["dit_path"]},
            },
        },
        "generate": {
            "common": {
                "vae_path": "./ckpts/vae/mage_flow_vae_bf16.safetensors",
                "text_encoder_path": "./ckpts/text_encoder/qwen3vl_4b_bf16.safetensors",
            },
            "versions": {
                "standard": {"dit_path": MAGE_FLOW_PROFILES[(False, "standard")]["dit_path"]},
                "turbo": {"dit_path": MAGE_FLOW_PROFILES[(False, "turbo")]["dit_path"]},
            },
        },
    },
    "supports_text_encoder": True,
    "supports_fp8_text_encoder": False,
    "supports_fp8_scaled": True,
    "requires_vae": True,
    "default_timestep_sampling": "shift",
    "default_weighting_scheme": "none",
    "default_guidance_scale": 1.0,
    "is_video": False,
    "icon": "MF",
    "color": "#0f766e",
    "pages": {
        "cache": {
            "versions": ["standard"],
            "supports_task_selector": False,
            "required_paths": ["vae", "text_encoder"],
            "flags": ["is_edit", "cache_seed"],
        },
        "train": {
            "supports_task_selector": False,
            "required_paths": ["dit"],
            "flags": ["is_edit", "fp8_base", "fp8_scaled", "allow_mage_architecture_mismatch"],
        },
        "generate": {
            "supports_task_selector": False,
            "required_paths": ["dit", "vae", "text_encoder"],
            "flags": ["is_edit", "control_images", "renormalize_cfg"],
        },
    },
},
```

Add the public helper after `get_architecture_names`:

```python
def get_mage_flow_profile(is_edit: bool, variant: str) -> Dict[str, Any]:
    normalized_variant = str(variant or "standard").strip().lower()
    key = (bool(is_edit), normalized_variant)
    if key not in MAGE_FLOW_PROFILES:
        raise ValueError(f"Unsupported Mage-Flow variant: {variant}")
    return deepcopy(MAGE_FLOW_PROFILES[key])
```

Add `mage_flow` labels to the four `model_architecture_list` dictionaries in
`gui/utils/i18n.py`.

- [ ] **Step 4: Run catalog tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_model_catalog.py -q
```

Expected: PASS, including module-resolution and four-language coverage tests.

- [ ] **Step 5: Commit the catalog foundation**

```powershell
git add gui/utils/model_catalog.py gui/utils/i18n.py gui/tests/test_model_catalog.py
git commit -m "feat: add Mage-Flow model catalog profiles"
```

---

### Task 2: Add Supported Mage-Flow Downloads To The Installer

**Files:**
- Modify: `1.install-uv-qinglong.ps1:36-151`
- Modify: `1.install-uv-qinglong.ps1:727-750`
- Modify: `gui/tests/test_install_script_downloads.py:1-80`

**Interfaces:**
- Consumes: existing `DownloadModelComponent`.
- Produces: `DownloadMageFlowModel -DiffusionFiles [string[]]`.
- Produces local files matching the catalog paths from Task 1.

- [ ] **Step 1: Write the failing installer contract test**

Add:

```python
def test_mage_flow_download_prompt_exposes_only_supported_bf16_components(self):
    script = self.install_script
    self.assertIn("function DownloadMageFlowModel", script)
    self.assertIn("$download_mage_flow = Read-Host", script)
    self.assertIn('$mageFlowRoot = "./ckpts"', script)

    for expected in (
        'diffusion_models/mage_flow_bf16.safetensors',
        'diffusion_models/mage_flow_turbo_bf16.safetensors',
        'diffusion_models/mage_flow_edit_bf16.safetensors',
        'diffusion_models/mage_flow_edit_turbo_bf16.safetensors',
        'vae/mage_flow_vae_bf16.safetensors',
        'text_encoders/qwen3vl_4b_bf16.safetensors',
        'text_encoder/qwen3vl_4b_bf16.safetensors',
    ):
        self.assertIn(expected, script)

    mage_block = script.split("function DownloadMageFlowModel", 1)[1].split("function DownloadLensModel", 1)[0]
    self.assertIn('-RepoId "Comfy-Org/Mage-Flow"', mage_block)
    self.assertNotIn("int8_convrot", mage_block.lower())
    self.assertNotIn("processor", mage_block.lower())
    self.assertNotIn("tokenizer", mage_block.lower())
```

- [ ] **Step 2: Run the installer test and confirm it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_install_script_downloads.py -k mage_flow -q
```

Expected: FAIL because the installer has no Mage-Flow section.

- [ ] **Step 3: Implement the download function and prompt**

Place this function before `DownloadLensModel`:

```powershell
function DownloadMageFlowModel {
    param (
        [string[]]$DiffusionFiles
    )

    $mageFlowRoot = "./ckpts"
    New-Item -ItemType Directory -Force -Path $mageFlowRoot | Out-Null

    foreach ($filePath in $DiffusionFiles) {
        DownloadModelComponent `
            -RepoId "Comfy-Org/Mage-Flow" `
            -FilePath $filePath `
            -LocalDir $mageFlowRoot `
            -ErrorInfo "Download Comfy-Org/Mage-Flow/$filePath failed|下载 Mage-Flow $filePath 失败。"
    }

    DownloadModelComponent `
        -RepoId "Comfy-Org/Mage-Flow" `
        -FilePath "vae/mage_flow_vae_bf16.safetensors" `
        -LocalDir $mageFlowRoot `
        -ErrorInfo "Download Mage-Flow VAE failed|下载 Mage-Flow VAE 失败。"

    DownloadModelComponent `
        -RepoId "Comfy-Org/Mage-Flow" `
        -FilePath "text_encoders/qwen3vl_4b_bf16.safetensors" `
        -TargetPath "text_encoder/qwen3vl_4b_bf16.safetensors" `
        -LocalDir $mageFlowRoot `
        -ErrorInfo "Download Mage-Flow Qwen3-VL text encoder failed|下载 Mage-Flow Qwen3-VL 文本编码器失败。"
}
```

Add an installer prompt with choices `1` through `5` and `n`:

```powershell
$download_mage_flow = Read-Host "请选择要下载的 Mage-Flow BF16 模型 [1/2/3/4/5/n] (默认为 n)
1: T2I Standard
2: T2I Turbo
3: Edit Standard
4: Edit Turbo
5: 全部下载
n: 不下载
Select Mage-Flow BF16 models [1/2/3/4/5/n] (default n)
1: T2I Standard
2: T2I Turbo
3: Edit Standard
4: Edit Turbo
5: Download all
n: Skip download"

$mageFlowDiffusionFiles = switch ($download_mage_flow) {
    "1" { @("diffusion_models/mage_flow_bf16.safetensors") }
    "2" { @("diffusion_models/mage_flow_turbo_bf16.safetensors") }
    "3" { @("diffusion_models/mage_flow_edit_bf16.safetensors") }
    "4" { @("diffusion_models/mage_flow_edit_turbo_bf16.safetensors") }
    "5" {
        @(
            "diffusion_models/mage_flow_bf16.safetensors",
            "diffusion_models/mage_flow_turbo_bf16.safetensors",
            "diffusion_models/mage_flow_edit_bf16.safetensors",
            "diffusion_models/mage_flow_edit_turbo_bf16.safetensors"
        )
    }
    default { @() }
}
if ($mageFlowDiffusionFiles.Count -gt 0) {
    DownloadMageFlowModel -DiffusionFiles $mageFlowDiffusionFiles
}
```

- [ ] **Step 4: Run installer and failure-propagation tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_install_script_downloads.py gui/tests/test_powershell_failure_propagation.py -q
```

Expected: PASS without downloading a model because the tests only inspect text.

- [ ] **Step 5: Commit installer support**

```powershell
git add 1.install-uv-qinglong.ps1 gui/tests/test_install_script_downloads.py
git commit -m "feat: add Mage-Flow BF16 downloads"
```

---

### Task 3: Add Native Mage-Flow PowerShell Workflows

**Files:**
- Create: `2.10mage_flow_cache_latent_and_text_encoder.ps1`
- Create: `3.10mage_flow_train_lora.ps1`
- Create: `5.10mage_flow_generate.ps1`
- Create: `gui/tests/test_mage_flow_scripts.py`
- Modify: `gui/tests/test_multiscript_param_consistency.py:42-75`

**Interfaces:**
- Produces editable script variables parsed later by `script_preset_catalog`.
- Invokes `mage_flow_cache_latents.py`, `mage_flow_cache_text_encoder_outputs.py`, `mage_flow_train_network.py`, and `mage_flow_generate_image.py`.
- Guarantees each native Python call is followed by `Assert-NativeCommandSucceeded`.

- [ ] **Step 1: Write failing script contract and AST tests**

Create `gui/tests/test_mage_flow_scripts.py`:

```python
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

        self.assertIn("mage_flow_generate_image.py", generate)
        self.assertIn("--output=$mage_output_path", generate)
        self.assertIn("--control_image=", generate)
        self.assertNotIn("--save_path=", generate)

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
```

Update `test_generate_family_has_core_execution_flags` so the expected output
flag is architecture-aware:

```python
if name == "5.10mage_flow_generate.ps1":
    self.assertIn("--output=$mage_output_path", text)
    self.assertNotIn("--save_path=$save_path", text)
else:
    self.assertIn("--save_path=$save_path", text)
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_mage_flow_scripts.py gui/tests/test_multiscript_param_consistency.py -q
```

Expected: FAIL because the three scripts do not exist.

- [ ] **Step 3: Implement the dual cache script**

Use editable variables `dataset_config`, `vae`, `vae_dtype`, `batch_size`,
`device`, `num_workers`, `skip_existing`, `cache_seed`, `text_encoder`,
`text_encoder_dtype`, `text_encoder_batch_size`, `text_encoder_device`,
`text_encoder_num_workers`, `text_encoder_skip_existing`, and `is_edit` above
the standard marker.

The command section must follow this exact shape:

```powershell
$latent_args = [System.Collections.ArrayList]::new()
$text_args = [System.Collections.ArrayList]::new()

if ($is_edit) {
    [void]$latent_args.Add("--is_edit")
    [void]$text_args.Add("--is_edit")
}
[void]$latent_args.Add("--seed=$cache_seed")
[void]$text_args.Add("--text_encoder=$text_encoder")
[void]$text_args.Add("--text_encoder_dtype=$text_encoder_dtype")

python "./musubi-tuner/mage_flow_cache_latents.py" `
    --dataset_config=$dataset_config `
    --vae=$vae `
    --vae_dtype=$vae_dtype $latent_args
Assert-NativeCommandSucceeded "Command failed: 2.10mage_flow_cache_latent_and_text_encoder.ps1"

python "./musubi-tuner/mage_flow_cache_text_encoder_outputs.py" `
    --dataset_config=$dataset_config $text_args
Assert-NativeCommandSucceeded "Command failed: 2.10mage_flow_cache_latent_and_text_encoder.ps1"
```

Add optional batch/device/worker/skip flags to the matching argument list. Do
not add `--seed` to text caching.

- [ ] **Step 4: Implement the LoRA-only training script**

Use the repository's existing activation, Accelerate, native-command, logging,
optimizer, save/resume, sample, DDP, and block-swap conventions. Set:

```powershell
$is_edit = $False
$model_variant = "standard"
$dit = "./ckpts/diffusion_models/mage_flow_bf16.safetensors"
$vae = "./ckpts/vae/mage_flow_vae_bf16.safetensors"
$text_encoder = "./ckpts/text_encoder/qwen3vl_4b_bf16.safetensors"
$attn_mode = "sdpa"
$mixed_precision = "bf16"
$vae_dtype = "bfloat16"
$timestep_sampling = "shift"
$discrete_flow_shift = 6.0
$weighting_scheme = "none"
$fp8_base = $False
$fp8_scaled = $False
$blocks_to_swap = 0
$compile_fullgraph = $False
$allow_mage_architecture_mismatch = $False
```

Validate before launch:

```powershell
if ($mixed_precision -ine "bf16") { throw "Mage-Flow training requires bf16 mixed precision." }
if ($fp8_base -and -not $fp8_scaled) { throw "Mage-Flow fp8_base requires fp8_scaled." }
if (($blocks_to_swap -lt 0) -or ($blocks_to_swap -gt 10)) { throw "Mage-Flow blocks_to_swap must be 0 through 10." }
if ($compile_fullgraph) { throw "Mage-Flow does not support compile_fullgraph." }
if ($attn_mode -notin @("sdpa", "flash", "flash2")) { throw "Mage-Flow supports SDPA or FlashAttention 2 only." }
if ($dim_from_weights -and -not $network_weights) { throw "dim_from_weights requires network_weights." }
```

Build arguments with:

```powershell
$network_module = "musubi_tuner.networks.lora_mage_flow"
$script = "mage_flow_generate_image.py"
$ext_args = [System.Collections.ArrayList]::new()
if ($is_edit) { [void]$ext_args.Add("--is_edit") }
if ($fp8_scaled) { [void]$ext_args.Add("--fp8_scaled") }
if ($allow_mage_architecture_mismatch) { [void]$ext_args.Add("--allow_mage_architecture_mismatch") }
if ($attn_mode -in @("flash", "flash2")) {
    [void]$ext_args.Add("--flash_attn")
} else {
    [void]$ext_args.Add("--sdpa")
}

python -m accelerate.commands.launch $launch_args "./musubi-tuner/mage_flow_train_network.py" `
    --dataset_config=$dataset_config `
    --dit=$dit `
    --vae=$vae `
    --text_encoder=$text_encoder `
    --network_module=$network_module `
    --mixed_precision=$mixed_precision `
    --timestep_sampling=$timestep_sampling `
    --discrete_flow_shift=$discrete_flow_shift `
    --weighting_scheme=$weighting_scheme `
    --learning_rate=$lr $ext_args
Assert-NativeCommandSucceeded "Command failed: 3.10mage_flow_train_lora.ps1"
```

- [ ] **Step 5: Implement the bespoke generation script**

Use `mage_`-prefixed editable values for fields that do not match the generic
GUI contract:

```powershell
$is_edit = $False
$model_variant = "standard"
$mage_output_path = "./output_dir/mage_flow.png"
$mage_control_images = ""
$mage_width = 1024
$mage_height = 1024
$mage_max_size = $null
$mage_steps = 20
$mage_cfg_scale = 5.0
$mage_flow_shift = 6.0
$mage_seed = 42
$mage_device = ""
$mage_dtype = "bfloat16"
$mage_attn_mode = "sdpa"
$mage_renormalize_cfg = $False
$mage_allow_architecture_mismatch = $False
$mage_lora_weights = ""
$mage_lora_multipliers = ""
```

Split ordered paths only on newlines or semicolons, validate the mode and
dimensions, then construct accepted CLI arguments:

```powershell
$controlImages = @(
    $mage_control_images -split "[`r`n;]+" |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)
if ($is_edit -and (($controlImages.Count -lt 1) -or ($controlImages.Count -gt 3))) {
    throw "Mage-Flow Edit requires one to three ordered control images."
}
if (-not $is_edit -and $controlImages.Count -gt 0) {
    throw "Mage-Flow T2I does not accept control images."
}
if (($null -eq $mage_width) -xor ($null -eq $mage_height)) {
    throw "Mage-Flow width and height must be supplied together."
}
if (-not $is_edit -and $null -ne $mage_max_size) {
    throw "Mage-Flow max_size is Edit-only."
}

$ext_args = [System.Collections.ArrayList]::new()
if ($is_edit) { [void]$ext_args.Add("--is_edit") }
foreach ($controlImage in $controlImages) {
    [void]$ext_args.Add("--control_image=$controlImage")
}

python "./musubi-tuner/$script" `
    --dit=$dit `
    --vae=$vae `
    --text_encoder=$text_encoder `
    --prompt=$prompt `
    --negative_prompt=$negative_prompt `
    --output=$mage_output_path `
    --steps=$mage_steps `
    --cfg_scale=$mage_cfg_scale `
    --flow_shift=$mage_flow_shift `
    --seed=$mage_seed `
    --dtype=$mage_dtype `
    --attn_mode=$mage_attn_mode $ext_args
Assert-NativeCommandSucceeded "Command failed: 5.10mage_flow_generate.ps1"
```

Add width/height, Edit max size, device, CFG renormalization, adapter mismatch,
LoRA `nargs` lists, and LoRA multiplier-count validation only when populated.

- [ ] **Step 6: Run script, AST, family, and failure-propagation tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_mage_flow_scripts.py gui/tests/test_multiscript_param_consistency.py gui/tests/test_powershell_failure_propagation.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the PowerShell workflows**

```powershell
git add 2.10mage_flow_cache_latent_and_text_encoder.ps1 3.10mage_flow_train_lora.ps1 5.10mage_flow_generate.ps1 gui/tests/test_mage_flow_scripts.py gui/tests/test_multiscript_param_consistency.py
git commit -m "feat: add Mage-Flow PowerShell workflows"
```

---

### Task 4: Build And Validate Cache And Training Jobs

**Files:**
- Modify: `gui/utils/command_builder.py:13-335`
- Modify: `gui/utils/command_builder.py:730-878`
- Modify: `gui/utils/command_builder.py:1204-1460`
- Modify: `gui/utils/command_builder.py:1580-1925`
- Create: `gui/tests/test_mage_flow_command_builder.py`

**Interfaces:**
- Consumes: `model_catalog.get_mage_flow_profile`.
- Produces: `_with_mage_flow_defaults(state, page_key) -> dict[str, Any]`.
- Produces: `_validate_mage_flow_train_state(state, train_mode) -> None`.
- Preserves: `build_cache_jobs` and `build_train_job` public signatures.

- [ ] **Step 1: Write failing cache and training builder tests**

Create the test file with shared config and these core tests:

```python
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = ROOT / "gui"
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from utils.command_builder import CommandBuildError, build_cache_jobs, build_train_job  # noqa: E402


PROJECT_CONFIG = {
    "dataset": {
        "general": {"resolution": [512, 512]},
        "datasets": [{"image_directory": "images", "caption_extension": ".txt", "batch_size": 1}],
    },
    "interop": {"dataset_extra": {"root": {}, "general": {}, "datasets": [{}]}},
}


class TestMageFlowCommandBuilder(unittest.TestCase):
    def test_edit_cache_passes_same_identity_to_both_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = build_cache_jobs(
                {
                    "arch": "Mage-Flow",
                    "is_edit": True,
                    "vae_path": "ckpts/vae/mage.safetensors",
                    "text_encoder_path": "ckpts/te/qwen.safetensors",
                    "cache_seed": 17,
                },
                tmp,
                PROJECT_CONFIG,
            )
        self.assertEqual([job.script_key for job in jobs], [
            "musubi_tuner.mage_flow_cache_latents",
            "musubi_tuner.mage_flow_cache_text_encoder_outputs",
        ])
        self.assertIn("--is_edit", jobs[0].args)
        self.assertIn("--is_edit", jobs[1].args)
        self.assertIn("--seed=17", jobs[0].args)
        self.assertNotIn("--seed=17", jobs[1].args)
        self.assertFalse(any("--processor" in arg or "--tokenizer" in arg for job in jobs for arg in job.args))

    def test_t2i_cache_omits_edit_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = build_cache_jobs({"arch": "Mage-Flow", "is_edit": False}, tmp, PROJECT_CONFIG)
        self.assertNotIn("--is_edit", jobs[0].args)
        self.assertNotIn("--is_edit", jobs[1].args)

    def test_edit_train_uses_fixed_lora_contract_and_optional_sampling_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = build_train_job(
                {
                    "arch": "Mage-Flow",
                    "version": "standard",
                    "is_edit": True,
                    "mixed_precision": "bf16",
                    "attn_mode": "flash2",
                    "fp8_base": True,
                    "fp8_scaled": True,
                    "vae_path": "ckpts/vae/mage.safetensors",
                    "text_encoder_path": "ckpts/te/qwen.safetensors",
                    "enable_sample": True,
                    "sample_prompts": "prompts.txt",
                },
                tmp,
                PROJECT_CONFIG,
            )
        self.assertTrue(job.script_key.endswith(str(Path("musubi_tuner") / "mage_flow_train_network.py")))
        self.assertIn("--dit=./ckpts/diffusion_models/mage_flow_edit_bf16.safetensors", job.args)
        self.assertIn("--vae=ckpts/vae/mage.safetensors", job.args)
        self.assertIn("--text_encoder=ckpts/te/qwen.safetensors", job.args)
        self.assertIn("--network_module=musubi_tuner.networks.lora_mage_flow", job.args)
        self.assertIn("--is_edit", job.args)
        self.assertIn("--fp8_base", job.args)
        self.assertIn("--fp8_scaled", job.args)
        self.assertIn("--flash_attn", job.args)

    def test_train_rejects_mage_flow_unsupported_combinations(self):
        invalid_states = (
            {"mixed_precision": "fp16"},
            {"fp8_base": True, "fp8_scaled": False},
            {"blocks_to_swap": 11},
            {"compile_fullgraph": True},
            {"attn_mode": "sageattn"},
            {"enable_lycoris": True},
            {"enable_blocks": True, "include_patterns": ".*q_proj.*"},
            {"dim_from_weights": True, "network_weights": ""},
            {"enable_sample": True, "sample_prompts": "prompts.txt", "vae_path": "", "text_encoder_path": ""},
        )
        with tempfile.TemporaryDirectory() as tmp:
            for extra in invalid_states:
                state = {"arch": "Mage-Flow", "mixed_precision": "bf16", "attn_mode": "sdpa", **extra}
                with self.subTest(extra=extra), self.assertRaises(CommandBuildError):
                    build_train_job(state, tmp, PROJECT_CONFIG)
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_mage_flow_command_builder.py -k "cache or train" -q
```

Expected: FAIL because Mage-specific mappings and validation are absent.

- [ ] **Step 3: Add Mage cache/train mappings and default resolution**

Add:

```python
MAGE_FLOW_ARCH = "Mage-Flow"
MAGE_FLOW_DEFAULT_OUTPUT_IMAGE = "./output_dir/mage_flow.png"
```

Extend mappings:

```python
NETWORK_MODULE_BY_ARCH[MAGE_FLOW_ARCH] = "musubi_tuner.networks.lora_mage_flow"
CACHE_LATENT_SCALARS["cache_seed"] = "--seed"
CACHE_LATENT_BOOLS["is_edit"] = "--is_edit"
CACHE_TEXT_BOOLS["is_edit"] = "--is_edit"
TRAIN_BOOLS["is_edit"] = "--is_edit"
TRAIN_BOOLS["allow_mage_architecture_mismatch"] = "--allow_mage_architecture_mismatch"

CACHE_LATENT_ARCH_SCALAR_KEYS[MAGE_FLOW_ARCH] = {"cache_seed"}
CACHE_LATENT_ARCH_BOOL_KEYS[MAGE_FLOW_ARCH] = {"is_edit"}
CACHE_TEXT_ARCH_BOOL_KEYS[MAGE_FLOW_ARCH] = {"is_edit"}
TRAIN_ARCH_BOOL_KEYS[MAGE_FLOW_ARCH] = {"fp8_scaled", "is_edit", "allow_mage_architecture_mismatch"}
TRAIN_DISABLED_BOOL_KEYS_BY_ARCH[MAGE_FLOW_ARCH] = {"split_attn", "img_in_txt_in_offloading"}
```

Add the state resolver:

```python
def _with_mage_flow_defaults(state: Mapping[str, Any], page_key: str) -> dict[str, Any]:
    resolved = dict(state)
    variant = str(resolved.get("version") or "standard").strip().lower()
    is_edit = _truthy(resolved.get("is_edit"))
    try:
        profile = model_catalog.get_mage_flow_profile(is_edit, variant)
    except ValueError as exc:
        raise CommandBuildError(str(exc)) from exc
    for key, value in profile.items():
        if value is not None and not _has_value(resolved.get(key)):
            resolved[key] = value
    resolved["version"] = variant
    return resolved
```

Call it immediately after `_resolve_architecture` in training when
`arch_name == MAGE_FLOW_ARCH`. Cache component defaults continue to resolve
through `model_catalog.get_path_defaults`, and cache does not use a DiT profile.

- [ ] **Step 4: Add optional sampling component handling and strict validation**

Before common training arguments are emitted:

```python
if arch_name == MAGE_FLOW_ARCH:
    _validate_mage_flow_train_state(state, train_mode)
```

After required training paths, add optional components:

```python
if arch_name == MAGE_FLOW_ARCH:
    for path_key in ("vae", "text_encoder"):
        candidates = MODEL_PATH_STATE_KEYS[path_key]
        if _has_value(_first_value(state, candidates)):
            _add_model_path(args, state, arch_name, "train", path_key)
```

Implement validation:

```python
def _validate_mage_flow_train_state(state: Mapping[str, Any], train_mode: str) -> None:
    if train_mode != "lora":
        raise CommandBuildError("Mage-Flow supports LoRA training only.")
    if _truthy(state.get("enable_lycoris")):
        raise CommandBuildError("Mage-Flow does not support LyCORIS training.")
    if _truthy(state.get("enable_blocks")) or _has_value(state.get("include_patterns")) or _has_value(state.get("exclude_patterns")):
        raise CommandBuildError("Mage-Flow uses a fixed LoRA target set; include/exclude patterns are unsupported.")
    if _normalize_train_mixed_precision(state.get("mixed_precision")) != "bf16":
        raise CommandBuildError("Mage-Flow training requires mixed_precision=bf16.")
    if _truthy(state.get("fp8_base")) and not _truthy(state.get("fp8_scaled")):
        raise CommandBuildError("Mage-Flow fp8_base requires fp8_scaled.")
    blocks = _as_int(state.get("blocks_to_swap"), 0)
    if not 0 <= blocks <= 10:
        raise CommandBuildError("Mage-Flow blocks_to_swap must be from 0 through 10.")
    if _truthy(state.get("compile_fullgraph")):
        raise CommandBuildError("Mage-Flow does not support compile_fullgraph; use compile without fullgraph.")
    attention = str(state.get("attn_mode") or "sdpa").strip().lower()
    if attention not in {"sdpa", "torch", "flash", "flash2", "flash_attn"}:
        raise CommandBuildError("Mage-Flow supports SDPA and FlashAttention 2 only.")
    if _truthy(state.get("dim_from_weights")) and not _has_value(state.get("network_weights")):
        raise CommandBuildError("Mage-Flow dim_from_weights requires network_weights.")
    if _truthy(state.get("enable_sample")):
        if not _has_value(state.get("sample_prompts")):
            raise CommandBuildError("Mage-Flow sampling requires sample_prompts.")
        if not _has_value(state.get("vae_path")) or not _has_value(state.get("text_encoder_path")):
            raise CommandBuildError("Mage-Flow sampling requires both VAE and text encoder paths.")
```

Special-case `_add_train_attention_args` so an empty Mage attention value emits
`--sdpa`, while `flash`, `flash2`, and `flash_attn` emit `--flash_attn`.

- [ ] **Step 5: Run builder and existing regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_mage_flow_command_builder.py gui/tests/test_command_builder.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit cache/training job support**

```powershell
git add gui/utils/command_builder.py gui/tests/test_mage_flow_command_builder.py
git commit -m "feat: build Mage-Flow cache and train jobs"
```

---

### Task 5: Add The Bespoke Mage-Flow Generation Builder

**Files:**
- Modify: `gui/utils/command_builder.py:880-1040`
- Modify: `gui/utils/command_builder.py:1204-1240`
- Modify: `gui/tests/test_mage_flow_command_builder.py`

**Interfaces:**
- Produces: `_build_mage_flow_generate_job(state, arch, arch_name, project_dir) -> CommandJob`.
- Consumes Mage GUI keys: `mage_output_path`, `mage_control_images`, `mage_width`,
  `mage_height`, `mage_max_size`, `mage_steps`, `mage_cfg_scale`,
  `mage_flow_shift`, `mage_seed`, `mage_device`, `mage_dtype`,
  `mage_attn_mode`, `mage_renormalize_cfg`,
  `mage_allow_architecture_mismatch`, `mage_lora_weights`, and
  `mage_lora_multipliers`.
- Preserves generic generation behavior for every other architecture.

- [ ] **Step 1: Add failing generation tests**

Append:

```python
def test_generate_uses_each_recommended_profile_without_generic_flags(self):
    expected = {
        (False, "standard"): ("mage_flow_bf16.safetensors", "20", "5.0"),
        (False, "turbo"): ("mage_flow_turbo_bf16.safetensors", "4", "1.0"),
        (True, "standard"): ("mage_flow_edit_bf16.safetensors", "30", "5.0"),
        (True, "turbo"): ("mage_flow_edit_turbo_bf16.safetensors", "4", "1.0"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for (is_edit, variant), values in expected.items():
            controls = "source.png" if is_edit else ""
            job = build_generate_job(
                {
                    "arch": "Mage-Flow",
                    "version": variant,
                    "is_edit": is_edit,
                    "prompt": "replace the sky" if is_edit else "a glass greenhouse",
                    "mage_control_images": controls,
                },
                tmp,
            )
            filename, steps, cfg = values
            with self.subTest(is_edit=is_edit, variant=variant):
                self.assertTrue(any(arg.endswith(filename) for arg in job.args))
                self.assertIn(f"--steps={steps}", job.args)
                self.assertIn(f"--cfg_scale={cfg}", job.args)
                self.assertIn("--output=./output_dir/mage_flow.png", job.args)
                self.assertFalse(any(arg.startswith("--save_path") for arg in job.args))
                self.assertFalse(any(arg.startswith("--infer_steps") for arg in job.args))
                self.assertFalse(any(arg.startswith("--output_type") for arg in job.args))
                self.assertFalse(any(arg.startswith("--processor") for arg in job.args))


def test_edit_generate_preserves_ordered_repeated_control_images(self):
    with tempfile.TemporaryDirectory() as tmp:
        job = build_generate_job(
            {
                "arch": "Mage-Flow",
                "version": "standard",
                "is_edit": True,
                "prompt": "restyle the subject",
                "mage_control_images": "source.png\nstyle.png;pose.png",
                "mage_lora_weights": "one.safetensors\ntwo.safetensors",
                "mage_lora_multipliers": "0.8\n1.1",
                "mage_renormalize_cfg": True,
            },
            tmp,
        )
    controls = [arg.split("=", 1)[1] for arg in job.args if arg.startswith("--control_image=")]
    self.assertEqual(controls, ["source.png", "style.png", "pose.png"])
    lora_index = job.args.index("--lora_weight")
    self.assertEqual(job.args[lora_index + 1:lora_index + 3], ["one.safetensors", "two.safetensors"])
    multiplier_index = job.args.index("--lora_multiplier")
    self.assertEqual(job.args[multiplier_index + 1:multiplier_index + 3], ["0.8", "1.1"])
    self.assertIn("--renormalize_cfg", job.args)


def test_generate_rejects_invalid_mage_flow_inputs(self):
    invalid_states = (
        {"is_edit": True, "mage_control_images": ""},
        {"is_edit": True, "mage_control_images": "1.png\n2.png\n3.png\n4.png"},
        {"is_edit": False, "mage_control_images": "source.png"},
        {"is_edit": False, "mage_width": 1024, "mage_height": ""},
        {"is_edit": False, "mage_max_size": 1024},
        {"is_edit": False, "mage_steps": 0},
        {"is_edit": False, "mage_flow_shift": 0},
        {"is_edit": False, "mage_attn_mode": "sageattn"},
        {"is_edit": False, "mage_lora_weights": "one.safetensors", "mage_lora_multipliers": "1.0\n0.5"},
        {"is_edit": False, "from_file": "prompts.txt"},
        {"is_edit": False, "prompt": ""},
    )
    with tempfile.TemporaryDirectory() as tmp:
        for extra in invalid_states:
            state = {"arch": "Mage-Flow", "version": "standard", "prompt": "test", **extra}
            with self.subTest(extra=extra), self.assertRaises(CommandBuildError):
                build_generate_job(state, tmp)
```

- [ ] **Step 2: Run generation tests and confirm they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_mage_flow_command_builder.py -k generate -q
```

Expected: FAIL because generation still enters the generic builder.

- [ ] **Step 3: Branch early and implement the dedicated builder**

At the top of `build_generate_job`:

```python
if arch_name == MAGE_FLOW_ARCH:
    return _build_mage_flow_generate_job(state, arch, arch_name, project_dir)
```

Implement the builder with explicit accepted arguments:

```python
def _build_mage_flow_generate_job(
    state: Mapping[str, Any],
    arch: Mapping[str, Any],
    arch_name: str,
    project_dir: str | Path,
) -> CommandJob:
    resolved = _with_mage_flow_defaults(state, "generate")
    if not _has_value(resolved.get("prompt")):
        raise CommandBuildError("Mage-Flow generate requires a direct prompt.")
    if _has_value(resolved.get("from_file")) or _has_value(resolved.get("latent_path")):
        raise CommandBuildError("Mage-Flow does not support prompt files or latent decode.")

    is_edit = _truthy(resolved.get("is_edit"))
    controls = _split_path_list(str(resolved.get("mage_control_images") or ""))
    if is_edit and not 1 <= len(controls) <= 3:
        raise CommandBuildError("Mage-Flow Edit requires one to three ordered control images.")
    if not is_edit and controls:
        raise CommandBuildError("Mage-Flow T2I does not accept control images.")

    width = resolved.get("mage_width", resolved.get("width"))
    height = resolved.get("mage_height", resolved.get("height"))
    if _has_value(width) != _has_value(height):
        raise CommandBuildError("Mage-Flow width and height must be supplied together.")
    max_size = resolved.get("mage_max_size", resolved.get("max_size"))
    if not is_edit and _has_value(max_size):
        raise CommandBuildError("Mage-Flow max_size is Edit-only.")
    for label, value in (("width", width), ("height", height), ("max_size", max_size)):
        if _has_value(value) and _as_int(value, 0) <= 0:
            raise CommandBuildError(f"Mage-Flow {label} must be positive.")

    steps = resolved.get("mage_steps", resolved.get("steps"))
    cfg_scale = resolved.get("mage_cfg_scale", resolved.get("cfg_scale"))
    flow_shift = resolved.get("mage_flow_shift", 6.0)
    if _as_int(steps, 0) <= 0:
        raise CommandBuildError("Mage-Flow steps must be positive.")
    if _as_float(flow_shift, 0.0) <= 0:
        raise CommandBuildError("Mage-Flow flow_shift must be positive.")

    attention = str(resolved.get("mage_attn_mode") or "sdpa").strip().lower()
    if attention not in {"sdpa", "flash2"}:
        raise CommandBuildError("Mage-Flow attention must be sdpa or flash2.")

    args: list[str] = []
    for path_key in ("dit", "vae", "text_encoder"):
        _add_model_path(args, resolved, arch_name, "generate", path_key)
    _add_scalar(args, "--prompt", resolved.get("prompt"))
    _add_scalar(args, "--negative_prompt", resolved.get("negative_prompt", " "))
    output = _default_generate_file(
        resolved.get("mage_output_path"),
        MAGE_FLOW_DEFAULT_OUTPUT_IMAGE,
        "mage_flow.png",
    )
    _add_scalar(args, "--output", output)
    if is_edit:
        args.append("--is_edit")
    for control in controls:
        args.append(f"--control_image={control}")
    if _has_value(width):
        _add_scalar(args, "--width", width)
        _add_scalar(args, "--height", height)
    if _has_value(max_size):
        _add_scalar(args, "--max_size", max_size)
    _add_scalar(args, "--steps", steps)
    _add_scalar(args, "--cfg_scale", cfg_scale)
    _add_scalar(args, "--flow_shift", flow_shift)
    _add_scalar(args, "--seed", resolved.get("mage_seed", 42))
    _add_scalar(args, "--device", resolved.get("mage_device"))
    _add_scalar(args, "--dtype", resolved.get("mage_dtype", "bfloat16"))
    _add_scalar(args, "--attn_mode", attention)

    if _truthy(resolved.get("mage_renormalize_cfg")):
        args.append("--renormalize_cfg")
    if _truthy(resolved.get("mage_allow_architecture_mismatch")):
        args.append("--allow_mage_architecture_mismatch")

    weights = _split_path_list(str(resolved.get("mage_lora_weights") or ""))
    multipliers = _split_multi_value(str(resolved.get("mage_lora_multipliers") or ""))
    if len(multipliers) > len(weights):
        raise CommandBuildError("Mage-Flow LoRA multipliers cannot outnumber LoRA weights.")
    if weights:
        args.append("--lora_weight")
        args.extend(weights)
    if multipliers:
        args.append("--lora_multiplier")
        args.extend(multipliers)

    return CommandJob(
        name=f"{arch_name} Generate",
        script_key=str(arch["generate_module"]),
        args=args,
        runner_kwargs={},
    )
```

Use `_split_path_list` for paths so spaces inside a path are preserved, and
use `_split_multi_value` only for numeric multipliers.

- [ ] **Step 4: Run focused and complete command-builder tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_mage_flow_command_builder.py gui/tests/test_command_builder.py -q
```

Expected: PASS with no regressions in generic or Krea-2 generation.

- [ ] **Step 5: Commit bespoke generation**

```powershell
git add gui/utils/command_builder.py gui/tests/test_mage_flow_command_builder.py
git commit -m "feat: build Mage-Flow generation jobs"
```

---

### Task 6: Add Mage-Flow Cache And Training Controls

**Files:**
- Modify: `gui/wizard/step2_cache.py:105-330`
- Modify: `gui/wizard/step2_cache.py:520-620`
- Modify: `gui/wizard/step3_train.py:80-455`
- Modify: `gui/wizard/step3_train.py:690-900`
- Modify: `gui/wizard/step3_train.py:1059-1235`
- Create: `gui/tests/test_mage_flow_gui_contract.py`

**Interfaces:**
- Consumes: `model_catalog.get_mage_flow_profile`.
- Produces form state keys `is_edit`, `version`, `cache_seed`, and current Mage
  model component paths.
- Calls existing public `build_cache_jobs` and `build_train_job` unchanged.

- [ ] **Step 1: Write failing GUI wiring tests**

Create:

```python
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestMageFlowGuiContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = (ROOT / "gui/wizard/step2_cache.py").read_text(encoding="utf-8")
        cls.train = (ROOT / "gui/wizard/step3_train.py").read_text(encoding="utf-8")
        cls.generate = (ROOT / "gui/wizard/step4_generate.py").read_text(encoding="utf-8")

    def test_cache_exposes_explicit_mode_text_encoder_and_seed(self):
        self.assertIn('elif arch_name == "Mage-Flow"', self.cache)
        self.assertIn('"is_edit"', self.cache)
        self.assertIn('"cache_seed"', self.cache)
        self.assertIn("qwen3vl_4b_bf16.safetensors", self.cache)
        self.assertNotIn("mage_processor", self.cache)

    def test_train_limits_mage_flow_to_supported_controls(self):
        self.assertIn("def _sync_mage_flow_train_ui", self.train)
        self.assertIn('["sdpa", "flash2"]', self.train)
        self.assertIn("get_mage_flow_profile", self.train)
        self.assertIn("compile_fullgraph", self.train)
        self.assertIn("mage_flow", self.train.lower())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run GUI wiring tests and confirm they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_mage_flow_gui_contract.py -k "cache or train" -q
```

Expected: FAIL because the wizard pages have no Mage branches.

- [ ] **Step 3: Add cache controls**

Add a `Mage-Flow` text encoder branch in `_render_dynamic_model_paths`:

```python
elif arch_name == "Mage-Flow":
    self._set_control(
        "text_encoder_path",
        create_path_selector(
            label="Qwen3-VL 4B BF16 Text Encoder",
            selection_type="file",
            file_filter="*.safetensors",
            placeholder="./ckpts/text_encoder/qwen3vl_4b_bf16.safetensors",
        ),
        scope="model_paths",
    )
```

Add a Mage card in `_render_dynamic_arch_specific`:

```python
elif arch_name == "Mage-Flow":
    with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
        with ui.row().classes("w-full gap-4 items-end flex-wrap"):
            self._set_control(
                "is_edit",
                ui.toggle({False: "T2I", True: "Edit"}, value=bool(self.config.get("is_edit", False)))
                .props("no-caps")
                .classes("mage-flow-mode-toggle"),
                scope="arch_specific",
            )
            self.config.setdefault("cache_seed", 0)
            editable_slider(
                "cache_seed",
                self.config,
                "cache_seed",
                min_val=0,
                max_val=9999999999,
                step=1,
                decimals=0,
                label_default="Latent Seed",
            )
```

Ensure preset application and architecture changes preserve `is_edit` and
reapply catalog VAE/text-encoder defaults.

- [ ] **Step 4: Add training mode, defaults, and visibility synchronization**

Add the same Qwen3-VL path and T2I/Edit segmented control to the Mage dynamic
training model card. Store references for the target-block card, block-swap
slider, and fullgraph toggle. Define the normal option set once:

```python
TRAIN_ATTN_MODES = ["flash", "xformers", "sdpa", "sageattn", "flash3"]
```

Wire the mode control to:

```python
def _on_mage_flow_train_mode_change(self, value: Any) -> None:
    self.config["is_edit"] = bool(value)
    self._apply_mage_flow_train_defaults("Mage-Flow")
```

Implement:

```python
def _apply_mage_flow_train_defaults(self, arch_name: str) -> None:
    if arch_name != "Mage-Flow":
        return
    is_edit = bool(self.config.get("is_edit", False))
    variant = self._current_model_version(arch_name) or "standard"
    profile = model_catalog.get_mage_flow_profile(is_edit, variant)
    defaults = {
        "timestep_sampling": "shift",
        "discrete_flow_shift": 6.0,
        "weighting_scheme": "none",
        "mixed_precision": "bf16",
        "vae_dtype": "bfloat16",
        "split_attn": False,
        "compile_fullgraph": False,
        "enable_lycoris": False,
        "enable_blocks": False,
    }
    self.config.update(defaults)
    self._write_bound_control_values(defaults)
    self._write_control_value(self.dit_path, profile["dit_path"])
    if hasattr(self, "attn_mode"):
        self.attn_mode.options = ["sdpa", "flash2"]
        self._write_control_value(self.attn_mode, "sdpa")
        self.attn_mode.update()
```

Implement `_sync_mage_flow_train_ui` to:

```python
def _sync_mage_flow_train_ui(self) -> None:
    is_mage = (self._selected_arch or "FLUX.2") == "Mage-Flow"
    if self._tab_lycoris is not None:
        self._tab_lycoris.visible = not is_mage and (
            self.train_mode is None or self.train_mode.value == "lora"
        )
    if self._target_blocks_card is not None:
        self._target_blocks_card.visible = not is_mage
    if self._compile_fullgraph_control is not None:
        self._compile_fullgraph_control.visible = not is_mage
    if is_mage:
        self.config["compile_fullgraph"] = False
        self.config["enable_lycoris"] = False
        self.config["enable_blocks"] = False
        self.config["blocks_to_swap"] = min(int(self.config.get("blocks_to_swap", 0)), 10)
        if self._blocks_to_swap_slider is not None:
            self._blocks_to_swap_slider.props("max=10")
    elif self._blocks_to_swap_slider is not None:
        self._blocks_to_swap_slider.props("max=40")
```

When leaving Mage-Flow, set `self.attn_mode.options = TRAIN_ATTN_MODES`, restore
`flash` if the current value is not valid, and call `update()`. Call both Mage
helpers from `_on_arch_change`, mode changes, `_sync_train_mode_ui`, and
`_apply_config`.

- [ ] **Step 5: Run GUI contract and builder tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_mage_flow_gui_contract.py gui/tests/test_mage_flow_command_builder.py -k "cache or train" -q
```

Expected: PASS.

- [ ] **Step 6: Commit cache/training GUI controls**

```powershell
git add gui/wizard/step2_cache.py gui/wizard/step3_train.py gui/tests/test_mage_flow_gui_contract.py
git commit -m "feat: add Mage-Flow cache and train controls"
```

---

### Task 7: Add The Dedicated Mage-Flow Generation UI

**Files:**
- Modify: `gui/wizard/step4_generate.py:19-110`
- Modify: `gui/wizard/step4_generate.py:117-370`
- Modify: `gui/wizard/step4_generate.py:483-1020`
- Modify: `gui/tests/test_mage_flow_gui_contract.py`

**Interfaces:**
- Produces all `mage_` generation keys consumed by Task 5.
- Keeps only Basic, Model, Prompt, and Architecture tabs visible for Mage-Flow.
- Preserves the full generic tab set for every other architecture.

- [ ] **Step 1: Add failing generation UI contract tests**

Append:

```python
def test_generate_exposes_bespoke_mage_flow_fields(self):
    for field in (
        "mage_output_path",
        "mage_control_images",
        "mage_width",
        "mage_height",
        "mage_max_size",
        "mage_steps",
        "mage_cfg_scale",
        "mage_flow_shift",
        "mage_seed",
        "mage_device",
        "mage_dtype",
        "mage_attn_mode",
        "mage_renormalize_cfg",
        "mage_allow_architecture_mismatch",
        "mage_lora_weights",
        "mage_lora_multipliers",
    ):
        self.assertIn(field, self.generate)
    self.assertIn("def _apply_mage_flow_generate_profile", self.generate)
    self.assertIn("def _sync_mage_flow_generate_ui", self.generate)
    self.assertIn('selection_type="file"', self.generate)
```

- [ ] **Step 2: Run the generation UI test and confirm it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_mage_flow_gui_contract.py -k generate -q
```

Expected: FAIL because none of the bespoke fields exist.

- [ ] **Step 3: Track tabs and hide unsupported generic surfaces**

Store the tab objects in `__init__` and `render`, and store the prompt-file
batch card. Add:

```python
def _sync_mage_flow_generate_ui(self) -> None:
    is_mage = (self._selected_arch or "FLUX.2") == "Mage-Flow"
    for tab in (
        self._tab_lora,
        self._tab_generation,
        self._tab_inference,
        self._tab_compile,
    ):
        if tab is not None:
            tab.visible = not is_mage
    if self._prompt_file_card is not None:
        self._prompt_file_card.visible = not is_mage
```

Call this on architecture change and after preset application. Do not delete
the generic controls; the bespoke builder ignores them and they must reappear
unchanged when another architecture is selected.

- [ ] **Step 4: Render Mage model and architecture controls**

Add the Mage Qwen3-VL path branch in `_render_dynamic_te_paths`. Add these names
to `_dynamic_field_names` so they are removed when switching architectures.

Render a single un-nested Mage card with responsive rows:

```python
elif arch_name == "Mage-Flow":
    with ui.card().classes(get_classes("card") + " w-full q-pa-md"):
        with ui.row().classes("w-full gap-4 items-end flex-wrap"):
            self._set_control(
                "is_edit",
                ui.toggle(
                    {False: "T2I", True: "Edit"},
                    value=bool(self.config.get("is_edit", False)),
                    on_change=lambda e: self._on_mage_flow_mode_change(e.value),
                ).props("no-caps").classes("mage-flow-mode-toggle"),
                scope="arch_specific",
            )
            self._set_control(
                "mage_dtype",
                ui.select(
                    ["bfloat16", "float16", "float32"],
                    value="bfloat16",
                    label="Dtype",
                ).classes("flex-1"),
                scope="arch_specific",
            )
            self._set_control(
                "mage_attn_mode",
                ui.select(["sdpa", "flash2"], value="sdpa", label="Attention"),
                scope="arch_specific",
            )
            self._set_control(
                "mage_device",
                ui.select(["", "cuda", "cpu"], value="", label="Device"),
                scope="arch_specific",
            )

        with ui.row().classes("w-full gap-4 q-mt-md flex-wrap"):
            self._set_control("mage_width", ui.number("Width", min=16, step=16), scope="arch_specific")
            self._set_control("mage_height", ui.number("Height", min=16, step=16), scope="arch_specific")
            self._set_control("mage_max_size", ui.number("Edit Max Size", min=16, step=16), scope="arch_specific")

        with ui.row().classes("w-full gap-4 q-mt-md flex-wrap"):
            self._set_control("mage_steps", ui.number("Steps", min=1, step=1), scope="arch_specific")
            self._set_control("mage_cfg_scale", ui.number("CFG", min=0, step=0.1), scope="arch_specific")
            self._set_control("mage_flow_shift", ui.number("Flow Shift", min=0.1, step=0.1), scope="arch_specific")
            self._set_control("mage_seed", ui.number("Seed", min=0, step=1), scope="arch_specific")

        self._set_control(
            "mage_control_images",
            ui.textarea("Ordered Edit References", placeholder="One path per line").props("autogrow"),
            scope="arch_specific",
        )
        self._set_control(
            "mage_output_path",
            create_path_selector(
                label="Output Image",
                default_path="./output_dir/mage_flow.png",
                selection_type="file",
                file_filter="*.png *.jpg *.jpeg *.webp",
            ),
            scope="arch_specific",
        )
        self._set_control(
            "mage_lora_weights",
            ui.textarea("LoRA Weights", placeholder="One path per line").props("autogrow"),
            scope="arch_specific",
        )
        self._set_control(
            "mage_lora_multipliers",
            ui.textarea("LoRA Multipliers", placeholder="One value per line").props("autogrow"),
            scope="arch_specific",
        )
        self.config.setdefault("mage_renormalize_cfg", False)
        toggle_switch("mage_renormalize_cfg", self.config, "mage_renormalize_cfg", label_default="Renormalize CFG")
        self.config.setdefault("mage_allow_architecture_mismatch", False)
        toggle_switch(
            "mage_allow_architecture_mismatch",
            self.config,
            "mage_allow_architecture_mismatch",
            label_default="Allow Architecture Mismatch",
        )
```

Use tooltips on the two unfamiliar toggles rather than visible explanatory
paragraphs.

- [ ] **Step 5: Apply profile defaults on mode or variant changes**

Implement:

```python
def _on_mage_flow_mode_change(self, value: Any) -> None:
    self.config["is_edit"] = bool(value)
    self._apply_mage_flow_generate_profile()


def _apply_mage_flow_generate_profile(self) -> None:
    if (self._selected_arch or "") != "Mage-Flow":
        return
    variant = self._current_model_version("Mage-Flow") or "standard"
    profile = model_catalog.get_mage_flow_profile(bool(self.config.get("is_edit", False)), variant)
    values = {
        "dit_path": profile["dit_path"],
        "mage_steps": profile["steps"],
        "mage_cfg_scale": profile["cfg_scale"],
        "mage_width": profile["width"],
        "mage_height": profile["height"],
        "mage_max_size": profile["max_size"],
        "mage_flow_shift": 6.0,
        "mage_seed": 42,
    }
    for name, value in values.items():
        control = getattr(self, name, None)
        if control is not None:
            self._write_control_value(control, value)
```

Call it after dynamic fields are rendered, after the model catalog defaults,
after mode changes, after Standard/Turbo changes, and after a Mage preset is
applied. Preserve preset values by applying the profile before
`_apply_form_state(config)` completes its final field writes.

- [ ] **Step 6: Run GUI and generation builder tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_mage_flow_gui_contract.py gui/tests/test_mage_flow_command_builder.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the generation UI**

```powershell
git add gui/wizard/step4_generate.py gui/tests/test_mage_flow_gui_contract.py
git commit -m "feat: add Mage-Flow generation controls"
```

---

### Task 8: Register Scripts And Add Built-In Presets

**Files:**
- Modify: `gui/utils/script_preset_catalog.py:15-325`
- Modify: `gui/utils/script_preset_catalog.py:380-410`
- Modify: `gui/utils/script_coverage_manifest.py:6-45`
- Create: `gui/presets/cache/mage_flow_t2i.toml`
- Create: `gui/presets/cache/mage_flow_edit.toml`
- Create: `gui/presets/train/mage_flow_t2i.toml`
- Create: `gui/presets/train/mage_flow_edit.toml`
- Create: `gui/presets/generate/mage_flow_t2i_standard.toml`
- Create: `gui/presets/generate/mage_flow_t2i_turbo.toml`
- Create: `gui/presets/generate/mage_flow_edit_standard.toml`
- Create: `gui/presets/generate/mage_flow_edit_turbo.toml`
- Modify: `gui/tests/test_script_preset_catalog_sources.py`
- Modify: `gui/tests/test_script_coverage_manifest.py`
- Modify: `gui/tests/test_preset_scope_and_defaults.py`

**Interfaces:**
- Consumes the editable variables from Task 3.
- Produces eight named built-in presets and classifies all three scripts as
  `NATIVE_GUI`.

- [ ] **Step 1: Write failing preset and manifest tests**

Add:

```python
def test_mage_flow_scripts_are_native_gui(self):
    for script in (
        "2.10mage_flow_cache_latent_and_text_encoder.ps1",
        "3.10mage_flow_train_lora.ps1",
        "5.10mage_flow_generate.ps1",
    ):
        self.assertIn(script, self.manifest.NATIVE_GUI)
```

Add to `TestPresetScopeAndDefaults`:

```python
def test_mage_flow_presets_cover_cache_train_and_four_generate_profiles(self):
    manager = self.config_manager_module.ConfigManager()
    self.assertEqual(manager.load_config("cache", "mage_flow_t2i")["is_edit"], False)
    self.assertEqual(manager.load_config("cache", "mage_flow_edit")["is_edit"], True)
    self.assertEqual(manager.load_config("train", "mage_flow_t2i")["is_edit"], False)
    self.assertEqual(manager.load_config("train", "mage_flow_edit")["is_edit"], True)

    expected = {
        "mage_flow_t2i_standard": (False, "standard", 20, 5.0),
        "mage_flow_t2i_turbo": (False, "turbo", 4, 1.0),
        "mage_flow_edit_standard": (True, "standard", 30, 5.0),
        "mage_flow_edit_turbo": (True, "turbo", 4, 1.0),
    }
    for name, values in expected.items():
        preset = manager.load_config("generate", name)
        is_edit, variant, steps, cfg = values
        with self.subTest(preset=name):
            self.assertEqual(preset["arch"], "Mage-Flow")
            self.assertEqual(preset["is_edit"], is_edit)
            self.assertEqual(preset["version"], variant)
            self.assertEqual(preset["mage_steps"], steps)
            self.assertEqual(preset["mage_cfg_scale"], cfg)
            self.assertEqual(preset["mage_output_path"], "./output_dir/mage_flow.png")
            self.assertNotIn("processor_path", preset)
            self.assertNotIn("tokenizer_path", preset)
```

- [ ] **Step 2: Run preset and manifest tests and confirm they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_script_preset_catalog_sources.py gui/tests/test_script_coverage_manifest.py gui/tests/test_preset_scope_and_defaults.py -k mage_flow -q
```

Expected: FAIL because the catalog, manifest entries, and TOMLs are absent.

- [ ] **Step 3: Register source scripts and key mappings**

Allow source entries to carry typed overrides:

```python
PRESET_SOURCES: Dict[str, Dict[str, Dict[str, Any]]] = {
```

Add source entries sharing the three scripts. Each non-default profile uses an
`overrides` dictionary. Merge it after script translation:

```python
preset = _translate_to_preset(scope, slug, entry["arch"], parsed, entry["script"])
preset.update(copy.deepcopy(entry.get("overrides", {})))
built[slug] = preset
```

Use these exact entries:

```python
# PRESET_SOURCES["cache"]
"mage_flow_t2i": {
    "arch": "Mage-Flow",
    "script": "2.10mage_flow_cache_latent_and_text_encoder.ps1",
    "overrides": {"is_edit": False},
},
"mage_flow_edit": {
    "arch": "Mage-Flow",
    "script": "2.10mage_flow_cache_latent_and_text_encoder.ps1",
    "overrides": {"is_edit": True},
},

# PRESET_SOURCES["train"]
"mage_flow_t2i": {
    "arch": "Mage-Flow",
    "script": "3.10mage_flow_train_lora.ps1",
    "overrides": {
        "is_edit": False,
        "version": "standard",
        "dit_path": "./ckpts/diffusion_models/mage_flow_bf16.safetensors",
    },
},
"mage_flow_edit": {
    "arch": "Mage-Flow",
    "script": "3.10mage_flow_train_lora.ps1",
    "overrides": {
        "is_edit": True,
        "version": "standard",
        "dit_path": "./ckpts/diffusion_models/mage_flow_edit_bf16.safetensors",
        "output_name": "mage_flow_edit_lora_qinglong",
    },
},

# PRESET_SOURCES["generate"]
"mage_flow_t2i_standard": {
    "arch": "Mage-Flow",
    "script": "5.10mage_flow_generate.ps1",
    "overrides": {"is_edit": False, "version": "standard"},
},
"mage_flow_t2i_turbo": {
    "arch": "Mage-Flow",
    "script": "5.10mage_flow_generate.ps1",
    "overrides": {
        "is_edit": False,
        "version": "turbo",
        "dit_path": "./ckpts/diffusion_models/mage_flow_turbo_bf16.safetensors",
        "mage_steps": 4,
        "mage_cfg_scale": 1.0,
    },
},
"mage_flow_edit_standard": {
    "arch": "Mage-Flow",
    "script": "5.10mage_flow_generate.ps1",
    "overrides": {
        "is_edit": True,
        "version": "standard",
        "dit_path": "./ckpts/diffusion_models/mage_flow_edit_bf16.safetensors",
        "mage_control_images": "",
        "mage_width": None,
        "mage_height": None,
        "mage_max_size": 1024,
        "mage_steps": 30,
        "mage_cfg_scale": 5.0,
    },
},
"mage_flow_edit_turbo": {
    "arch": "Mage-Flow",
    "script": "5.10mage_flow_generate.ps1",
    "overrides": {
        "is_edit": True,
        "version": "turbo",
        "dit_path": "./ckpts/diffusion_models/mage_flow_edit_turbo_bf16.safetensors",
        "mage_control_images": "",
        "mage_width": None,
        "mage_height": None,
        "mage_max_size": 1024,
        "mage_steps": 4,
        "mage_cfg_scale": 1.0,
    },
},
```

Add key mappings:

```python
# cache
"is_edit": "is_edit",
"cache_seed": "cache_seed",
"text_encoder_dtype": "text_encoder_dtype",

# train
"is_edit": "is_edit",
"model_variant": "version",
"allow_mage_architecture_mismatch": "allow_mage_architecture_mismatch",

# generate
"is_edit": "is_edit",
"model_variant": "version",
"mage_output_path": "mage_output_path",
"mage_control_images": "mage_control_images",
"mage_width": "mage_width",
"mage_height": "mage_height",
"mage_max_size": "mage_max_size",
"mage_steps": "mage_steps",
"mage_cfg_scale": "mage_cfg_scale",
"mage_flow_shift": "mage_flow_shift",
"mage_seed": "mage_seed",
"mage_device": "mage_device",
"mage_dtype": "mage_dtype",
"mage_attn_mode": "mage_attn_mode",
"mage_renormalize_cfg": "mage_renormalize_cfg",
"mage_allow_architecture_mismatch": "mage_allow_architecture_mismatch",
"mage_lora_weights": "mage_lora_weights",
"mage_lora_multipliers": "mage_lora_multipliers",
```

Add Mage `text_encoder -> text_encoder_path` overrides for cache, train, and
generate source slugs.

- [ ] **Step 4: Create the eight TOML presets**

Every preset must contain `_label`, `arch`, `_source_script`, the shared
component paths, explicit `is_edit`, and relevant mode/variant values.

The T2I Standard generation file must contain:

```toml
_label = "Mage-Flow T2I Standard"
arch = "Mage-Flow"
_source_script = "5.10mage_flow_generate.ps1"
version = "standard"
is_edit = false
dit_path = "./ckpts/diffusion_models/mage_flow_bf16.safetensors"
vae_path = "./ckpts/vae/mage_flow_vae_bf16.safetensors"
text_encoder_path = "./ckpts/text_encoder/qwen3vl_4b_bf16.safetensors"
prompt = "A glass greenhouse above a quiet city"
negative_prompt = " "
mage_output_path = "./output_dir/mage_flow.png"
mage_control_images = ""
mage_width = 1024
mage_height = 1024
mage_steps = 20
mage_cfg_scale = 5.0
mage_flow_shift = 6.0
mage_seed = 42
mage_device = ""
mage_dtype = "bfloat16"
mage_attn_mode = "sdpa"
mage_renormalize_cfg = false
mage_allow_architecture_mismatch = false
mage_lora_weights = ""
mage_lora_multipliers = ""
```

For Edit presets, omit `mage_width`/`mage_height`, set `mage_max_size = 1024`,
set `is_edit = true`, and leave `mage_control_images = ""` so validation forces
the user to choose real inputs. Set each file's DiT, version, steps, and CFG
from the four-profile table.

The two training presets use `version = "standard"`, fixed BF16/shift-6/none
defaults, SDPA, `blocks_to_swap = 0`, `enable_lycoris = false`, and the matching
T2I/Edit Standard DiT path.

Use this minimal T2I cache preset, changing only `_label` and `is_edit` for the
Edit cache preset:

```toml
_label = "Mage-Flow T2I Cache"
arch = "Mage-Flow"
_source_script = "2.10mage_flow_cache_latent_and_text_encoder.ps1"
version = "standard"
is_edit = false
vae_path = "./ckpts/vae/mage_flow_vae_bf16.safetensors"
text_encoder_path = "./ckpts/text_encoder/qwen3vl_4b_bf16.safetensors"
vae_dtype = "bfloat16"
text_encoder_dtype = "bfloat16"
batch_size = 1
te_batch_size = 1
cache_seed = 0
```

Use this minimal T2I training preset, changing `_label`, `is_edit`, `dit_path`,
and `output_name` to the explicit Edit values from the source entry for the
Edit training preset:

```toml
_label = "Mage-Flow T2I Standard LoRA"
arch = "Mage-Flow"
_source_script = "3.10mage_flow_train_lora.ps1"
version = "standard"
train_mode = "lora"
is_edit = false
dit_path = "./ckpts/diffusion_models/mage_flow_bf16.safetensors"
vae_path = "./ckpts/vae/mage_flow_vae_bf16.safetensors"
text_encoder_path = "./ckpts/text_encoder/qwen3vl_4b_bf16.safetensors"
mixed_precision = "bf16"
vae_dtype = "bfloat16"
timestep_sampling = "shift"
discrete_flow_shift = 6.0
weighting_scheme = "none"
attn_mode = "sdpa"
fp8_base = false
fp8_scaled = false
blocks_to_swap = 0
compile = false
compile_fullgraph = false
enable_lycoris = false
enable_blocks = false
allow_mage_architecture_mismatch = false
output_name = "mage_flow_lora_qinglong"
output_dir = "./output_dir"
```

- [ ] **Step 5: Add all scripts to `NATIVE_GUI` and run tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_script_preset_catalog_sources.py gui/tests/test_script_coverage_manifest.py gui/tests/test_preset_scope_and_defaults.py -q
```

Expected: PASS, including classification of every root PowerShell script.

- [ ] **Step 6: Commit presets and metadata**

```powershell
git add gui/utils/script_preset_catalog.py gui/utils/script_coverage_manifest.py gui/presets/cache/mage_flow_t2i.toml gui/presets/cache/mage_flow_edit.toml gui/presets/train/mage_flow_t2i.toml gui/presets/train/mage_flow_edit.toml gui/presets/generate/mage_flow_t2i_standard.toml gui/presets/generate/mage_flow_t2i_turbo.toml gui/presets/generate/mage_flow_edit_standard.toml gui/presets/generate/mage_flow_edit_turbo.toml gui/tests/test_script_preset_catalog_sources.py gui/tests/test_script_coverage_manifest.py gui/tests/test_preset_scope_and_defaults.py
git commit -m "feat: add Mage-Flow GUI presets"
```

---

### Task 9: Document The Supported Mage-Flow Surface

**Files:**
- Modify: `README.md:93-100`
- Modify: `gui/README.md:10-140`
- Modify: `gui/README.en.md:10-140`
- Modify: `gui/README.ja.md:10-140`
- Modify: `gui/README.ko.md:10-140`
- Modify: `gui/PARAMETERS.md:1-520`

**Interfaces:**
- Documents the same file paths, entry points, modes, defaults, and limitations
  enforced by Tasks 1 through 8.

- [ ] **Step 1: Add documentation assertions before editing**

Run this check and confirm it exits nonzero:

```powershell
@'
from pathlib import Path
paths = [
    Path("README.md"),
    Path("gui/README.md"),
    Path("gui/README.en.md"),
    Path("gui/README.ja.md"),
    Path("gui/README.ko.md"),
    Path("gui/PARAMETERS.md"),
]
missing = [str(path) for path in paths if "Mage-Flow" not in path.read_text(encoding="utf-8")]
raise SystemExit("Missing Mage-Flow documentation: " + ", ".join(missing) if missing else 0)
'@ | .\.venv\Scripts\python.exe -
```

Expected: nonzero with every listed file reported.

- [ ] **Step 2: Update model lists and entry-point tables**

Add `Mage-Flow` to the root and four GUI model lists. Add this row to each GUI
architecture table, translated around the table but retaining exact module
names:

```markdown
| Mage-Flow | mage_flow_cache_latents / mage_flow_cache_text_encoder_outputs | mage_flow_train_network | mage_flow_generate_image |
```

- [ ] **Step 3: Add parameter and limitation guidance**

Add a focused Mage-Flow section to `gui/PARAMETERS.md` covering:

```markdown
#### Mage-Flow

| Parameter | Type | Meaning | Constraint |
| --- | --- | --- | --- |
| is_edit | segmented | T2I or Edit identity | Must match cache, train, and generate |
| version | select | Standard or Turbo | Selects the BF16 DiT recommendation |
| cache_seed | integer | Stable latent sampling seed | Latent cache only |
| mage_control_images | multiline paths | Ordered Edit references | Edit: 1-3; T2I: none |
| mage_steps | integer | Euler sampling steps | Standard T2I 20, Standard Edit 30, Turbo 4 |
| mage_cfg_scale | float | CFG scale | Standard 5.0, Turbo 1.0 |
| mage_width / mage_height | integers | Explicit output dimensions | Supply together |
| mage_max_size | integer | Reference-aspect maximum size | Edit only |
| mage_attn_mode | select | Attention backend | sdpa or flash2 |
| mage_output_path | file path | Generated image file | Uses --output |
```

State that processor assets auto-resolve from the pinned Microsoft repository,
INT8 ConvRot/full fine-tuning are unsupported, and real-weight parity remains
experimental. Link:

- `https://huggingface.co/Comfy-Org/Mage-Flow`
- `https://github.com/sdbds/musubi-tuner/blob/qinglong/docs/mage_flow.md`

- [ ] **Step 4: Re-run documentation and translation coverage checks**

Run:

```powershell
@'
from pathlib import Path
paths = [
    Path("README.md"),
    Path("gui/README.md"),
    Path("gui/README.en.md"),
    Path("gui/README.ja.md"),
    Path("gui/README.ko.md"),
    Path("gui/PARAMETERS.md"),
]
for path in paths:
    text = path.read_text(encoding="utf-8")
    assert "Mage-Flow" in text, path
assert "INT8 ConvRot" in Path("gui/PARAMETERS.md").read_text(encoding="utf-8")
assert "--output" in Path("gui/PARAMETERS.md").read_text(encoding="utf-8")
'@ | .\.venv\Scripts\python.exe -
.\.venv\Scripts\python.exe -m pytest gui/tests/test_model_catalog.py -k translations -q
```

Expected: both commands PASS.

- [ ] **Step 5: Commit documentation**

```powershell
git add README.md gui/README.md gui/README.en.md gui/README.ja.md gui/README.ko.md gui/PARAMETERS.md
git commit -m "docs: document Mage-Flow workflows"
```

---

### Task 10: Run Full Automated And Visual Verification

**Files:**
- Verify only; do not create persistent model artifacts or screenshots in the repository.

**Interfaces:**
- Consumes all prior tasks.
- Produces the final evidence used for completion reporting.

- [ ] **Step 1: Run PowerShell syntax and contract tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests/test_mage_flow_scripts.py gui/tests/test_multiscript_param_consistency.py gui/tests/test_powershell_failure_propagation.py gui/tests/test_install_script_downloads.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the complete GUI test suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest gui/tests -q
```

Expected: PASS with no pre-existing architecture regressions.

- [ ] **Step 3: Run lightweight upstream Mage-Flow tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest musubi-tuner/tests/test_mage_flow_attention.py musubi-tuner/tests/test_mage_flow_cache.py musubi-tuner/tests/test_mage_flow_checkpoints.py musubi-tuner/tests/test_mage_flow_contracts.py musubi-tuner/tests/test_mage_flow_edit.py musubi-tuner/tests/test_mage_flow_entrypoints.py musubi-tuner/tests/test_mage_flow_lora.py musubi-tuner/tests/test_mage_flow_model.py musubi-tuner/tests/test_mage_flow_runtime.py musubi-tuner/tests/test_mage_flow_text_encoder.py musubi-tuner/tests/test_mage_flow_training.py -q
```

Expected: PASS without downloading released weights.

- [ ] **Step 4: Start the GUI and perform desktop visual smoke checks**

Start a hidden local server:

```powershell
$guiProcess = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "gui\launch.py","--port","8890","--no-browser" -WindowStyle Hidden -PassThru
```

Using the `playwright` skill, inspect `/cache`, `/train`, and `/generate` at
`1440x1000`. Select `Mage-Flow` on each route and capture temporary screenshots.
Verify:

```javascript
await page.setViewportSize({ width: 1440, height: 1000 });
for (const route of ["/cache", "/train", "/generate"]) {
  await page.goto(`http://127.0.0.1:8890${route}`);
  await page.getByText("FLUX.2", { exact: true }).first().click();
  await page.getByText("Mage-Flow", { exact: true }).last().click();
  await page.waitForTimeout(250);
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  if (overflow) throw new Error(`horizontal overflow on ${route}`);
  await page.screenshot({ path: `${process.env.TEMP}/mage-flow-${route.slice(1)}-desktop.png`, fullPage: true });
}
```

Check that mode and variant are visible, no processor/tokenizer field exists,
and unsupported training/generation tabs disappear.

- [ ] **Step 5: Perform mobile visual smoke checks**

At `390x844`, repeat `/cache`, `/train`, and `/generate`:

```javascript
await page.setViewportSize({ width: 390, height: 844 });
for (const route of ["/cache", "/train", "/generate"]) {
  await page.goto(`http://127.0.0.1:8890${route}`);
  await page.getByText("FLUX.2", { exact: true }).first().click();
  await page.getByText("Mage-Flow", { exact: true }).last().click();
  await page.waitForTimeout(250);
  const metrics = await page.evaluate(() => ({
    viewport: window.innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
    clippedText: [...document.querySelectorAll("button, label")]
      .filter((element) => {
        const rect = element.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0 && element.scrollWidth > element.clientWidth + 1;
      })
      .map((element) => element.textContent?.trim())
      .filter(Boolean),
  }));
  if (metrics.scrollWidth > metrics.viewport || metrics.clippedText.length) {
    throw new Error(JSON.stringify({ route, metrics }));
  }
  await page.screenshot({ path: `${process.env.TEMP}/mage-flow-${route.slice(1)}-mobile.png`, fullPage: true });
}
```

Stop the server:

```powershell
Stop-Process -Id $guiProcess.Id
```

- [ ] **Step 6: Verify the final diff and residual risk**

Run:

```powershell
git diff --check
git status --short
git log --oneline --max-count=12
```

Expected: no whitespace errors; only intended Mage-Flow changes plus the
user's pre-existing unrelated working-tree files. Before reporting completion,
invoke `superpowers:verification-before-completion`. Report explicitly that
real checkpoint downloads, real-weight inference, and real-weight training were
not executed.

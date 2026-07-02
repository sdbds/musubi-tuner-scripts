[中文](./README.md) | [English](./README.en.md)

# Musubi Tuner GUI

A NiceGUI-based graphical interface for managing the complete musubi-tuner workflow.

## Features

- 🎨 **Full workflow coverage**: Dataset tagging → Caching → Training → Inference
- 🤖 **Multi-architecture support**: FLUX.2, Wan2.1, HunyuanVideo, FramePack, Long-CAT, Z-Image, Qwen Image, HV 1.5, Lens, Ideogram-4, HiDream O1, FLUX Kontext, Krea-2, and more
- 💾 **Preset management**: Save and load commonly used configurations
- 📝 **Real-time logs**: View command output and progress
- 🌐 **Cross-platform**: Windows/Linux supported, can run locally or in the cloud
- ⚡ **Direct invocation**: Calls Python scripts directly, no PowerShell dependency
- 🌓 **Theme toggle**: Dark/light theme with automatic preference saving
- 🌐 **Internationalization (i18n)**: Supports Chinese, English, Japanese, and Korean
- 🧪 **Advanced training**: Supports SOAR auxiliary training, D-OPSD distillation, and more

## Installation

```bash
# Enter the project directory
cd musubi-tuner-scripts

# Install project and GUI dependencies
uv sync --extra cu130 --extra gui --extra lycoris --extra attention --index-strategy unsafe-best-match

# Ensure all musubi-tuner dependencies are installed (torch, accelerate, etc.)
```

## Usage

### Option 1: Root launch script

```powershell
# Run from the project root
./1.6.GUI.ps1

# Specify port
./1.6.GUI.ps1 -Port 8888

# Cloud mode (allow external access)
./1.6.GUI.ps1 -Cloud

# Native window mode
./1.6.GUI.ps1 -Native

# Do not auto-open browser
./1.6.GUI.ps1 -NoBrowser
```

### Option 2: Run Python directly

```bash
# Run from the project root
python gui/launch.py

# Cloud mode (allow external access)
python gui/launch.py --cloud

# Specify port
python gui/launch.py --port 8888

# Native window mode
python gui/launch.py --native

# Do not auto-open browser
python gui/launch.py --no-browser
```

### Option 3: Run from the gui directory

```bash
cd gui
python launch.py
```

## Workflow

1. **Dataset Tagging** (`/tagging`)
   - Supports Qwen-VL and other tagging models
   - Batch image processing
   - Custom prompt prefixes/suffixes

2. **Caching** (`/cache`)
   - Select model architecture
   - Configure model paths
   - Pre-compute Latent and Text Encoder outputs
   - Directly calls `python -m musubi_tuner.xxx_cache_latents`

3. **LoRA Training** (`/train`)
   - Multi-tab parameter organization
   - Basic settings, model paths, training parameters, network structure, optimizer, advanced options
   - Real-time training log viewing
   - Save/load preset support
   - Directly calls `python -m accelerate.commands.launch musubi_tuner.xxx_train_network`

4. **Inference / Generation** (`/generate`)
   - Use trained LoRA weights
   - Adjust generation parameters
   - Support for reference image editing
   - Directly calls `python -m musubi_tuner.xxx_generate`

## Invocation Method

The GUI directly calls Python modules without relying on PowerShell scripts:

```bash
# Cache Latents
python -m musubi_tuner.flux_2_cache_latents --dataset_config=... --vae=...

# Cache Text Encoder
python -m musubi_tuner.flux_2_cache_text_encoder_outputs --dataset_config=... --text_encoder=...

# Training (using accelerate)
python -m accelerate.commands.launch --mixed_precision=bf16 musubi_tuner.flux_2_train_network --dit=... --vae=...

# Inference
python -m musubi_tuner.flux_2_generate_image --dit=... --prompt=...
```

## Supported Model Architectures

| Architecture | Cache Module | Training Module | Generation Module |
|------|---------|---------|---------|
| FLUX.2 | flux_2_cache_latents | flux_2_train_network | flux_2_generate_image |
| FLUX Kontext | flux_kontext_cache_latents | flux_kontext_train_network | flux_kontext_generate_image |
| Wan2.1 | wan_cache_latents | wan_train_network | wan_generate_video |
| HunyuanVideo | cache_latents | hv_train_network | hv_generate_video |
| FramePack | fpack_cache_latents | fpack_train_network | fpack_generate_video |
| Long-CAT | longcat_cache_latents | longcat_train_network | - |
| Z-Image | zimage_cache_latents | zimage_train_network | zimage_generate_image |
| HV 1.5 | hv_1_5_cache_latents | hv_1_5_train_network | hv_1_5_generate_video |
| Qwen Image | qwen_image_cache_latents | qwen_image_train_network | qwen_image_generate |
| Lens | lens_cache_latents | lens_train_network | lens_generate_image |
| Ideogram-4 | ideogram4_cache_latents | ideogram4_train_network | ideogram4_generate_image |
| HiDream O1 | hidream_o1_cache_pixel | hidream_o1_train_network | hidream_o1_generate_image |
| Krea-2 | krea2_cache_latents | krea2_train_network | krea2_generate_image |

## Presets

The `gui/presets/` directory is organized into subdirectories by stage, each containing TOML preset files:

### `presets/cache/` - Cache Presets

flux2, flux_kontext, framepack, hidream_o1, hunyuan_video, hv_1_5, ideogram4, krea2, lens, long_cat, qwen_image, wan2_1, zimage, zimage_dopsd

### `presets/train/` - Training Presets

flux2, flux_kontext, framepack, hidream_o1, hidream_o1_dev, hunyuan_video, hv_1_5, ideogram4, krea2, lens, lens_finetune, lens_finetune_low_vram, lens_low_vram, long_cat, qwen_image, qwen_image_finetune, wan2_1, zimage, zimage_dopsd, zimage_dopsd_finetune, zimage_finetune

### `presets/generate/` - Generation Presets

flux2, flux_kontext, framepack, hidream_o1, hidream_o1_dev_edit_flow, hidream_o1_dev_flash, hunyuan_video, hv_1_5, ideogram4, krea2, lens, long_cat, qwen_image, wan2_1, zimage

### `presets/user/` - User Custom Presets

Custom presets saved through the GUI are stored in this directory.

## Project Structure

```
gui/
├── main.py              # Main entry point
├── launch.py            # Launch script
├── README.md            # Chinese documentation
├── README.en.md         # English documentation (this file)
├── PARAMETERS.md        # Parameter mapping documentation
├── UPDATES.md           # Update notes
├── theme.py             # Theme system (integrates sd-scripts styles)
├── STYLES_REUSE.md      # Style reuse guide
├── components/          # Reusable components
│   ├── path_selector.py    # Path selector
│   ├── log_viewer.py       # Log viewer
│   ├── preset_manager.py   # Preset manager
│   ├── model_selector.py   # Model selector
│   └── side_tools.py       # Side toolbar
├── wizard/             # Wizard steps
│   ├── step0_setup.py      # Environment check
│   ├── step1_tagging.py    # Dataset tagging
│   ├── step2_cache.py      # Caching
│   ├── step3_train.py      # Training
│   ├── step4_generate.py   # Inference / generation
│   ├── step7_settings.py   # Settings page
│   └── console_page.py     # Console page
├── utils/              # Utilities
│   ├── config_manager.py   # Configuration management
│   ├── process_runner.py   # Process runner (direct Python calls)
│   ├── model_catalog.py    # Model architecture catalog
│   ├── port_utils.py       # Port resolution
│   └── i18n.py             # Internationalization (reused from sd-scripts)
├── presets/            # Preset configurations
│   ├── cache/              # Cache presets (*.toml)
│   ├── train/              # Training presets (*.toml)
│   ├── generate/           # Generation presets (*.toml)
│   └── user/               # User custom presets
└── examples/           # Usage examples
    └── reuse_styles_example.py
```

## Notes

1. **Working directory**: The GUI runs scripts from the project root by default; ensure paths are set correctly
2. **Dependencies**: All musubi-tuner dependencies (torch, accelerate, etc.) must be installed
3. **VRAM**: Depending on the model and settings, significant VRAM may be required
4. **Presets**: Presets only save parameters, not model paths; adjust according to your setup
5. **Python modules**: Ensure the `musubi-tuner` directory is in the Python path, or installed as a package

## FAQ

**Q: How to add support for a new model architecture?**
A: Edit `gui/utils/model_catalog.py` and add the new architecture to the `MODEL_CATALOG` dictionary.

**Q: How to customize training script parameters?**
A: Modify the parameter construction logic in the corresponding step page code (e.g., `step3_train.py`).

**Q: How to use on a cloud server?**
A: Launch with the `--cloud` parameter, then access via browser at `http://<server-ip>:7788` (default port)

**Q: How to save my custom configuration?**
A: Click the "Save as Preset" button on the training page and enter a name.

**Q: Module not found error?**
A: Ensure you're running from the project root and the `musubi-tuner` directory exists. You can try:
```python
import sys
sys.path.insert(0, '.')
```

## Theme & Internationalization

### Dark/Light Theme

The GUI supports both dark and light themes. Click the sun/moon icon in the top-right corner to toggle. Theme preference is saved in the browser's `localStorage` and automatically loaded on next visit.

### Modern Theme (default)
- Dark background, modern cards and buttons
- Deep green + gold natural color scheme

### Green Gold Theme (from sd-scripts)
- Bright background, traditional green-gold color scheme
- Reused from `sd-scripts/gui/styles.py`

### Internationalization (i18n)

Supports four languages (Chinese, English, Japanese, Korean). Click the language dropdown in the top-right corner to switch. The page will automatically refresh to apply the new language.

For more details, see `STYLES_REUSE.md`, `UPDATES.md`, and `examples/reuse_styles_example.py`.

## License

Same as the main musubi-tuner project.

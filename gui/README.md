# Musubi Tuner GUI

基于 NiceGUI 的图形化界面，用于管理 musubi-tuner 的完整工作流程。

## 功能特点

- 🎨 **全流程覆盖**: 数据集打标 → 缓存 → 训练 → 推理
- 🤖 **多架构支持**: FLUX.2、Wan2.1、HunyuanVideo、FramePack、Long-CAT、Z-Image 等
- 💾 **预设管理**: 保存和加载常用配置
- 📝 **实时日志**: 查看命令输出和进度
- 🌐 **跨平台**: Windows/Linux 都支持，可本地运行或云端部署
- ⚡ **直接调用**: 直接调用 Python 脚本，不依赖 PowerShell

## 安装

```bash
# 进入项目目录
cd musubi-tuner-scripts

# 安装项目和 GUI 依赖
uv sync --extra cu130 --extra gui --extra lycoris --extra attention --index-strategy unsafe-best-match

# 确保已安装 musubi-tuner 的所有依赖（torch、accelerate 等）
```

## 使用方法

### 方式 1: 直接运行

```bash
# 在项目根目录运行
python gui/launch.py

# 云模式（允许外部访问）
python gui/launch.py --cloud

# 指定端口
python gui/launch.py --port 8888

# 原生窗口模式
python gui/launch.py --native
```

### 方式 2: 从 gui 目录运行

```bash
cd gui
python launch.py
```

## 工作流程

1. **数据集打标** (`/tagging`)
   - 支持 Qwen-VL 等打标模型
   - 批量处理图片
   - 自定义提示词前后缀

2. **缓存处理** (`/cache`)
   - 选择模型架构
   - 配置模型路径
   - 预计算 Latent 和 Text Encoder 输出
   - 直接调用 `python -m musubi_tuner.xxx_cache_latents`

3. **训练 LoRA** (`/train`)
   - 多标签页组织参数
   - 基础设置、模型路径、训练参数、网络结构、优化器、高级选项
   - 实时查看训练日志
   - 支持保存/加载预设
   - 直接调用 `python -m accelerate.commands.launch musubi_tuner.xxx_train_network`

4. **推理生成** (`/generate`)
   - 使用训练好的 LoRA
   - 调整生成参数
   - 支持参考图像编辑
   - 直接调用 `python -m musubi_tuner.xxx_generate`

## 调用方式说明

GUI 直接调用 Python 模块，不依赖 PowerShell 脚本：

```bash
# 缓存 Latent
python -m musubi_tuner.flux_2_cache_latents --dataset_config=... --vae=...

# 缓存 Text Encoder
python -m musubi_tuner.flux_2_cache_text_encoder_outputs --dataset_config=... --text_encoder=...

# 训练（使用 accelerate）
python -m accelerate.commands.launch --mixed_precision=bf16 musubi_tuner.flux_2_train_network --dit=... --vae=...

# 推理
python -m musubi_tuner.flux_2_generate_image --dit=... --prompt=...
```

## 支持的模型架构

| 架构 | 缓存模块 | 训练模块 | 生成模块 |
|------|---------|---------|---------|
| FLUX.2 | flux_2_cache_latents | flux_2_train_network | flux_2_generate_image |
| Wan2.1 | wan_cache_latents | wan_train_network | wan_generate_video |
| HunyuanVideo | hy_cache_latents | hy_train_network | hy_generate_video |
| FramePack | framepack_cache_latents | framepack_train_network | framepack_generate_video |
| Long-CAT | long_cat_cache_latent_and_text_encoder | long_cat_train_network | long_cat_generate |
| Z-Image | zimage_cache_latents | zimage_train_network | zimage_generate_image |
| HV 1.5 | hv_1_5_cache_latents | hv_1_5_train_network | hv_1_5_generate_video |
| Qwen Image | qwen_image_cache_latents | qwen_image_train_network | qwen_image_generate |

## 预设配置

`gui/presets/` 目录包含以下预设：

- `flux2_klein_base_4b_lora.json` - FLUX.2 Klein Base 4B 训练配置
- `flux2_dev_lora.json` - FLUX.2 Dev 训练配置
- `wan_14b_lora.json` - Wan 2.1 14B 训练配置
- `hunyuan_video_lora.json` - HunyuanVideo 训练配置
- `zimage_lora.json` - Z-Image 训练配置

## 项目结构

```
gui/
├── main.py              # 主入口
├── launch.py            # 启动脚本
├── README.md           # 本文档
├── theme.py            # 主题系统（整合 sd-scripts 样式）
├── STYLES_REUSE.md     # 样式复用说明
├── components/         # 可复用组件
│   ├── path_selector.py    # 路径选择器
│   ├── log_viewer.py       # 日志查看器
│   ├── preset_manager.py   # 预设管理
│   └── model_selector.py   # 模型选择器
├── wizard/             # 向导步骤
│   ├── step0_setup.py      # 环境检查
│   ├── step1_tagging.py    # 数据集打标
│   ├── step2_cache.py      # 缓存处理
│   ├── step3_train.py      # 训练
│   └── step4_generate.py   # 推理生成
├── utils/              # 工具
│   ├── config_manager.py   # 配置管理
│   ├── process_runner.py   # 进程运行（直接调用 Python）
│   └── i18n.py             # 国际化（复用 sd-scripts）
├── presets/            # 预设配置
│   └── *.json
└── examples/           # 使用示例
    └── reuse_styles_example.py
```

## 注意事项

1. **工作目录**: GUI 默认在项目根目录运行脚本，确保路径设置正确
2. **依赖**: 需要安装 musubi-tuner 的所有依赖（torch、accelerate 等）
3. **显存**: 根据模型和设置，可能需要较大的显存
4. **预设**: 预设只保存参数，不保存模型路径，需要根据实际情况调整
5. **Python 模块**: 确保 `musubi-tuner` 目录在 Python 路径中，或者已安装为包

## 常见问题

**Q: 如何添加新的模型架构支持？**
A: 编辑 `gui/components/model_selector.py`，在 `MODEL_ARCHITECTURES` 字典中添加新架构。

**Q: 如何自定义训练脚本参数？**
A: 在对应步骤的页面代码中（如 `step3_train.py`），修改参数构建逻辑。

**Q: 云服务器上如何使用？**
A: 使用 `--cloud` 参数启动，然后通过浏览器访问 `http://服务器IP:8080`

**Q: 如何保存我的自定义配置？**
A: 在训练页面点击"保存为预设"按钮，输入名称即可保存当前配置。

**Q: 提示找不到模块？**
A: 确保在项目根目录运行，且 `musubi-tuner` 目录存在。可以尝试：
```python
import sys
sys.path.insert(0, '.')
```

## 主题系统

GUI 支持两种主题样式：

### Modern Theme (默认)
- 深色背景，现代化卡片和按钮
- 深绿+金色自然主题配色

```python
from theme import apply_theme
apply_theme()  # 或 apply_theme('modern')
```

### Green Gold Theme (来自 sd-scripts)
- 明亮背景，传统绿金配色
- 复用自 `sd-scripts/gui/styles.py`

```python
from theme import apply_green_gold_styles
apply_green_gold_styles()
```

### 国际化 (i18n)

支持多语言（中/英/日/韩）：

```python
from utils.i18n import t, set_language

# 获取翻译
label = t('nav_home')  # 返回 "首页" (根据当前语言)

# 切换语言
set_language('en')  # 切换到英语
```

更多用法参见 `STYLES_REUSE.md` 和 `examples/reuse_styles_example.py`。

## 许可证

与 musubi-tuner 主项目保持一致。

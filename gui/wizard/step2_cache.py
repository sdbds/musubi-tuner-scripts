"""步骤 2: 缓存 Latent 和 Text Encoder 输出 - 完整参数"""
from nicegui import ui
from pathlib import Path
from typing import Dict, Any
from theme import get_classes, COLORS
from components.path_selector import create_path_selector
from components.model_selector import create_model_selector, get_arch_info
from components.preset_manager import create_preset_manager
from components.advanced_inputs import toggle_switch, editable_slider
from components.execution_panel import ExecutionPanel
from utils.config_manager import config_manager
from utils.command_builder import CommandBuildError, build_cache_jobs
from utils.dataset_config import summarize_dataset_state
from utils.form_state import FormStateMixin
from utils.i18n import t
from utils import model_catalog
from utils.process_runner import ProcessStatus


class CacheStep(FormStateMixin):
    """缓存页面 - 完整参数"""

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.project_dir = Path(__file__).resolve().parents[2]
        self.model_selector = None
        self.exec_panel = None
        self.arch_info = None
        self._selected_arch = None
        self._selected_version = None
        # Dynamic UI containers for model-specific sections
        self._model_path_container = None
        self._model_specific_container = None
        self._vae_model_card = None
        self._init_form_state()

    def render(self):
        """渲染页面"""
        with ui.column().classes(get_classes('page_container') + ' gap-4'):
            # 页面标题
            with ui.row().classes('w-full items-center gap-3 q-mb-sm'):
                ui.icon('save', size='32px')
                with ui.column().classes('gap-0'):
                    ui.label(t('cache_processing')).classes('text-h4 text-weight-bold').style('color: var(--color-text);')
                    ui.label(t('cache_desc', 'Pre-compute Latent and Text Encoder outputs')).classes('text-body2').style('color: var(--color-text-secondary);')

            # 预设管理
            create_preset_manager(
                get_config=self._get_config,
                apply_config=self._apply_config,
                scope="cache",
            )

            with ui.tabs().classes('w-full') as tabs:
                tab_basic = ui.tab(t('basic_settings'), icon='settings')
                tab_model = ui.tab(t('model_settings'), icon='folder')
                tab_latent = ui.tab(t('latent_cache', 'Latent Cache'), icon='image')
                tab_te = ui.tab(t('text_encoder'), icon='text_fields')
                tab_arch_specific = ui.tab(t('arch_specific', 'Architecture Specific'), icon='tune')
                tab_debug = ui.tab(t('debug', 'Debug'), icon='bug_report')

            with ui.tab_panels(tabs, value=tab_basic).classes('w-full'):
                with ui.tab_panel(tab_basic):
                    self.model_selector = create_model_selector(
                        on_change=self._on_arch_change,
                        default_arch="FLUX.2",
                        page_key="cache",
                    )

                with ui.tab_panel(tab_model):
                    self._render_model_paths()

                with ui.tab_panel(tab_latent):
                    self._render_latent_settings()

                with ui.tab_panel(tab_te):
                    self._render_te_settings()

                with ui.tab_panel(tab_arch_specific):
                    self._render_arch_specific()

                with ui.tab_panel(tab_debug):
                    self._render_debug_settings()

            # 执行面板 (含 Start/Stop 按钮 + 日志)
            self.exec_panel = ExecutionPanel(
                start_label=t('start_cache'),
                height='400px',
                on_start=self._start_cache,
            )

        self._on_arch_change("FLUX.2", get_arch_info("FLUX.2"))

    def _render_model_paths(self):
        """渲染模型路径设置"""
        dataset_summary = summarize_dataset_state(
            config_manager.load_project_config(self.project_dir),
            self.project_dir,
        )

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('dataset_reference', 'Dataset Reference')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            ui.label(
                t('cache_dataset_reference_desc', 'Dataset state is owned by Step 1. Cache only reads the saved project dataset.')
            ).classes('text-body2').style('color: var(--color-text-secondary);')
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                ui.label(f'{t("datasets", "Datasets")}: {dataset_summary["dataset_count"]}').classes('text-body2')
                ui.label(f'{t("resolution")}: {dataset_summary["resolution"]}').classes('text-body2')
            ui.label(f'dataset_config.toml: {dataset_summary["source_path"]}').classes('text-caption q-mt-sm').style(
                'color: var(--color-text-secondary); word-break: break-all;'
            )
            ui.button(
                t('open_dataset_page', 'Open Dataset Page'),
                on_click=lambda: ui.navigate.to('/tagging'),
                icon='open_in_new',
            ).classes('modern-btn-secondary q-mt-md')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md') as self._vae_model_card:
            ui.label(t('vae_model')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.vae_path = create_path_selector(
                label=t('vae_path'),
                selection_type='file',
                file_filter='*.safetensors *.pt *.pth',
                placeholder=t('select_vae', 'Select VAE model file')
            )

        # 动态模型路径区域（根据架构变化）
        self._model_path_container = ui.column().classes('w-full gap-3 q-mt-md')
        with self._model_path_container:
            self._render_dynamic_model_paths("FLUX.2")

    def _render_dynamic_model_paths(self, arch_name: str):
        """根据架构渲染动态模型路径"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            title = 'HiDream O1 Checkpoint' if arch_name == "HiDream O1" else t('text_encoder')
            ui.label(title).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')

            if arch_name == "HiDream O1":
                self._set_control("dit_path", create_path_selector(
                    label='HiDream O1 checkpoint (optional for text embedding cache)',
                    selection_type='file_or_dir',
                    file_filter='*.safetensors *.pt *.pth',
                    placeholder='./ckpts/hidream-o1-image/hidream_o1_image_bf16.safetensors'
                ), scope="model_paths")
            elif arch_name == "FLUX.2":
                self._set_control("text_encoder_path", create_path_selector(
                    label='Text Encoder (Mistral 3 / Qwen 3)',
                    selection_type='file',
                    placeholder='选择文本编码器模型'
                ), scope="model_paths")
            elif arch_name == "Lens":
                self._set_control("text_encoder_path", create_path_selector(
                    label=t('lens_text_encoder', 'Lens Text Encoder'),
                    selection_type='file',
                    file_filter='*.safetensors *.pt *.pth',
                    placeholder='./ckpts/lens/text_encoders/gpt_oss_20b_nvfp4.safetensors'
                ), scope="model_paths")
            elif arch_name == "FLUX Kontext":
                self._set_control("te1_path", create_path_selector(
                    label='Text Encoder 1 (T5-XXL)',
                    selection_type='file',
                    placeholder='如: t5xxl_fp16.safetensors'
                ), scope="model_paths")
                self._set_control("te2_path", create_path_selector(
                    label='Text Encoder 2 (CLIP-L)',
                    selection_type='file',
                    placeholder='如: clip_l.safetensors'
                ), scope="model_paths")
                self._set_control("image_encoder_path", create_path_selector(
                    label='Image Encoder',
                    selection_type='file',
                    placeholder='如: sigclip_vision_patch14_384.safetensors'
                ), scope="model_paths")
            elif arch_name == "HunyuanVideo":
                self._set_control("te1_path", create_path_selector(
                    label='Text Encoder 1 (LLaVA LLaMA3)',
                    selection_type='file',
                    placeholder='如: llava_llama3_fp16.safetensors'
                ), scope="model_paths")
                self._set_control("te2_path", create_path_selector(
                    label='Text Encoder 2 (CLIP-L)',
                    selection_type='file',
                    placeholder='如: clip_l.safetensors'
                ), scope="model_paths")
                self._set_control("t5_path", create_path_selector(
                    label='T5 模型路径',
                    selection_type='file',
                    placeholder='如: models_t5_umt5-xxl-enc-bf16.pth'
                ), scope="model_paths")
                self._set_control("clip_path", create_path_selector(
                    label='CLIP 模型路径',
                    selection_type='file',
                    placeholder='如: models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
                ), scope="model_paths")
            elif arch_name == "Wan2.1":
                self._set_control("te1_path", create_path_selector(
                    label='Text Encoder 1 (LLaVA LLaMA3)',
                    selection_type='file',
                    placeholder='如: llava_llama3_fp16.safetensors'
                ), scope="model_paths")
                self._set_control("te2_path", create_path_selector(
                    label='Text Encoder 2 (CLIP-L)',
                    selection_type='file',
                    placeholder='如: clip_l.safetensors'
                ), scope="model_paths")
                self._set_control("t5_path", create_path_selector(
                    label='T5 模型路径',
                    selection_type='file',
                    placeholder='如: models_t5_umt5-xxl-enc-bf16.pth'
                ), scope="model_paths")
                self._set_control("clip_path", create_path_selector(
                    label='CLIP 模型路径 (I2V 必需)',
                    selection_type='file',
                    placeholder='如: open-clip-xlm-roberta-large-vit-huge-14.pth'
                ), scope="model_paths")
                self._set_control("image_encoder_path", create_path_selector(
                    label='Image Encoder (I2V 用)',
                    selection_type='file',
                    placeholder='如: sigclip_vision_patch14_384.safetensors'
                ), scope="model_paths")
            elif arch_name in ("Qwen Image", "Long-CAT"):
                self._set_control("text_encoder_path", create_path_selector(
                    label='Text Encoder (Qwen2.5-VL)',
                    selection_type='file',
                    placeholder='如: qwen_2.5_vl_7b.safetensors'
                ), scope="model_paths")
            elif arch_name == "Z-Image":
                self._set_control("text_encoder_path", create_path_selector(
                    label='Text Encoder (Qwen3)',
                    selection_type='file',
                    placeholder='如: qwen_3_4b.safetensors'
                ), scope="model_paths")
                self._set_control("image_encoder_path", create_path_selector(
                    label='Image Encoder (SOAR / I2V)',
                    selection_type='file',
                    placeholder='SigLIP2 image encoder'
                ), scope="model_paths")
            elif arch_name == "HV 1.5":
                self._set_control("text_encoder_path", create_path_selector(
                    label='Text Encoder (Qwen2.5-VL)',
                    selection_type='file',
                    placeholder='如: qwen_2.5_vl_7b.safetensors'
                ), scope="model_paths")
                self._set_control("byt5_path", create_path_selector(
                    label='BYT5 模型路径',
                    selection_type='file',
                    placeholder='如: byt5_model.safetensors'
                ), scope="model_paths")
                self._set_control("image_encoder_path", create_path_selector(
                    label='Image Encoder (I2V 用)',
                    selection_type='file',
                    placeholder='如: sigclip_vision_patch14_384.safetensors'
                ), scope="model_paths")
            elif arch_name == "FramePack":
                self._set_control("te1_path", create_path_selector(
                    label='Text Encoder 1 (LLaVA LLaMA3)',
                    selection_type='file',
                    placeholder='如: llava_llama3_fp16.safetensors'
                ), scope="model_paths")
                self._set_control("te2_path", create_path_selector(
                    label='Text Encoder 2 (CLIP-L)',
                    selection_type='file',
                    placeholder='如: clip_l.safetensors'
                ), scope="model_paths")
                self._set_control("image_encoder_path", create_path_selector(
                    label='Image Encoder',
                    selection_type='file',
                    placeholder='如: sigclip_vision_patch14_384.safetensors'
                ), scope="model_paths")

    def _render_latent_settings(self):
        """渲染 Latent 缓存设置"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('latent_cache_basic')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                self.config.setdefault('batch_size', 1)
                editable_slider(t('batch_size'), self.config, 'batch_size', min_val=1, max_val=64, step=1, decimals=0)
                self.vae_dtype = ui.select(
                    ['', 'float32', 'float16', 'bfloat16'],
                    label='VAE 数据类型',
                    value=''
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.device = ui.select(['', 'cuda', 'cpu'], label='设备', value='').classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')

            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.config.setdefault('num_workers', 0)
                editable_slider(t('num_workers'), self.config, 'num_workers', min_val=0, max_val=16, step=1, decimals=0).tooltip(t('num_workers_tooltip', '0 = auto (CPU cores - 1)'))
                self.config.setdefault('skip_existing', False)
                toggle_switch(t('skip_existing'), self.config, 'skip_existing')

    def _render_te_settings(self):
        """渲染 Text Encoder 设置"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('te_cache_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                self.config.setdefault('te_batch_size', 16)
                editable_slider(t('te_batch_size'), self.config, 'te_batch_size', min_val=1, max_val=64, step=1, decimals=0)
                self.te_device = ui.select(['', 'cuda', 'cpu'], label='TE 设备', value='').classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.config.setdefault('te_num_workers', 0)
                editable_slider(t('te_num_workers'), self.config, 'te_num_workers', min_val=0, max_val=16, step=1, decimals=0).tooltip(t('num_workers_tooltip', '0 = auto (CPU cores - 1)'))

            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.config.setdefault('te_skip_existing', False)
                toggle_switch(t('te_skip_existing'), self.config, 'te_skip_existing')
                self._set_control("text_encoder_dtype", ui.select(
                    ['', 'bfloat16', 'float16', 'float32'],
                    label=t('text_encoder_dtype', 'Text Encoder Dtype'),
                    value=self.config.get('text_encoder_dtype', ''),
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"'))

    def _render_arch_specific(self):
        """渲染架构专属参数"""
        self._model_specific_container = ui.column().classes('w-full gap-3')
        with self._model_specific_container:
            self._render_dynamic_arch_specific("FLUX.2")

    def _render_dynamic_arch_specific(self, arch_name: str):
        """根据架构渲染专属参数"""
        if arch_name == "FLUX.2":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='FLUX.2')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4'):
                    self.config.setdefault('fp8_text_encoder', False)
                    toggle_switch('FP8 Text Encoder', self.config, 'fp8_text_encoder')
            self._render_dopsd_teacher_cache_card(arch_name)

        elif arch_name == "HunyuanVideo":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='HunyuanVideo')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4'):
                    self.config.setdefault('vae_chunk_size', 32)
                    editable_slider('VAE Chunk Size', self.config, 'vae_chunk_size', min_val=1, max_val=256, step=1, decimals=0)
                    self.config.setdefault('vae_spatial_tile_sample_min_size', 256)
                    editable_slider('VAE Spatial Tile Min Size', self.config, 'vae_spatial_tile_sample_min_size', min_val=64, max_val=512, step=64, decimals=0)
                with ui.row().classes('w-full gap-4 q-mt-md'):
                    self.config.setdefault('vae_tiling', True)
                    toggle_switch(t('vae_tiling'), self.config, 'vae_tiling')
                    self.config.setdefault('fp8_llm', False)
                    toggle_switch('FP8 LLM (Text Encoder 1)', self.config, 'fp8_llm')

        elif arch_name == "Wan2.1":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='Wan2.1')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                    self.config.setdefault('fp8_t5', False)
                    toggle_switch('FP8 T5', self.config, 'fp8_t5')
                    self.config.setdefault('vae_cache_cpu', True)
                    toggle_switch(t('vae_cache_cpu'), self.config, 'vae_cache_cpu')
                    self.config.setdefault('i2v', False)
                    toggle_switch('I2V Cache', self.config, 'i2v')
                    self.config.setdefault('one_frame', False)
                    toggle_switch(t('one_frame_mode'), self.config, 'one_frame')
                    self.config.setdefault('one_frame_no_2x', False)
                    toggle_switch(t('one_frame_no_2x'), self.config, 'one_frame_no_2x')
                    self.config.setdefault('one_frame_no_4x', False)
                    toggle_switch(t('one_frame_no_4x'), self.config, 'one_frame_no_4x')

        elif arch_name == "FramePack":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='FramePack')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4'):
                    self.config.setdefault('vae_chunk_size', 32)
                    editable_slider('VAE Chunk Size', self.config, 'vae_chunk_size', min_val=1, max_val=256, step=1, decimals=0)
                    self.config.setdefault('vae_spatial_tile_sample_min_size', 256)
                    editable_slider('VAE Spatial Tile Min Size', self.config, 'vae_spatial_tile_sample_min_size', min_val=64, max_val=512, step=64, decimals=0)
                with ui.row().classes('w-full gap-4 q-mt-md'):
                    self.config.setdefault('fp8_llm', False)
                    toggle_switch('FP8 LLM', self.config, 'fp8_llm')
                    self.config.setdefault('f1_mode', False)
                    toggle_switch(t('f1_mode'), self.config, 'f1_mode')
                    self.config.setdefault('one_frame', False)
                    toggle_switch(t('one_frame_mode'), self.config, 'one_frame')
                    self.config.setdefault('one_frame_no_2x', False)
                    toggle_switch(t('one_frame_no_2x'), self.config, 'one_frame_no_2x')
                    self.config.setdefault('one_frame_no_4x', False)
                    toggle_switch(t('one_frame_no_4x'), self.config, 'one_frame_no_4x')

        elif arch_name in ("Qwen Image", "Long-CAT"):
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(f'{arch_name} 专属参数').classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4'):
                    self.config.setdefault('fp8_vl', False)
                    toggle_switch(t('fp8_vl'), self.config, 'fp8_vl')
                    self.config.setdefault('text_encoder_cpu', False)
                    toggle_switch(t('text_encoder_cpu'), self.config, 'text_encoder_cpu')
                if arch_name == "Qwen Image":
                    with ui.row().classes('w-full gap-4 q-mt-md'):
                        self._set_control("edit_version", ui.select(
                            ['', 'original', '2509', '2511'],
                            label='Edit Version', value=''
                        ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"'), scope="arch_specific")
                        self.config.setdefault('edit_mode', False)
                        toggle_switch(t('edit_mode'), self.config, 'edit_mode')
                        self.config.setdefault('edit_plus', False)
                        toggle_switch(t('edit_plus_mode'), self.config, 'edit_plus')

        elif arch_name == "Z-Image":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='Z-Image')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4'):
                    self.config.setdefault('fp8_llm', False)
                    toggle_switch('FP8 Qwen3', self.config, 'fp8_llm')
                    self.config.setdefault('i2v', False)
                    toggle_switch('SOAR / I2V Cache', self.config, 'i2v')
                    self.config.setdefault('text_encoder_cpu', False)
                    toggle_switch(t('text_encoder_cpu'), self.config, 'text_encoder_cpu')
            self._render_dopsd_teacher_cache_card(arch_name)

        elif arch_name == "HiDream O1":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='HiDream O1')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4'):
                    self.config.setdefault('fp8_te', False)
                    toggle_switch(t('fp8_te'), self.config, 'fp8_te')

        elif arch_name == "HV 1.5":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='HunyuanVideo 1.5')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4'):
                    self.config.setdefault('vae_sample_size', 128)
                    editable_slider('VAE Sample Size', self.config, 'vae_sample_size', min_val=64, max_val=512, step=64, decimals=0)
                with ui.row().classes('w-full gap-4 q-mt-md'):
                    self.config.setdefault('text_encoder_cpu', False)
                    toggle_switch(t('text_encoder_cpu'), self.config, 'text_encoder_cpu')
                    self.config.setdefault('vae_enable_patch_conv', False)
                    toggle_switch(t('vae_enable_patch_conv'), self.config, 'vae_enable_patch_conv')

        elif arch_name == "FLUX Kontext":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='FLUX Kontext')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4'):
                    self.config.setdefault('fp8_t5', False)
                    toggle_switch('FP8 T5', self.config, 'fp8_t5')

        else:
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('select_arch_first')).classes('text-body1').style('color: var(--color-text-muted);')

    def _render_dopsd_teacher_cache_card(self, arch_name: str) -> None:
        version = self._current_model_version(arch_name)
        if not model_catalog.supports_dopsd_cache(arch_name, version=version):
            self.config['dopsd_cache_teacher_outputs'] = False
            return

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('dopsd_teacher_cache')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4 q-mt-md items-center flex-wrap'):
                self.config.setdefault('dopsd_cache_teacher_outputs', False)
                self._set_control("dopsd_cache_teacher_outputs", toggle_switch(
                    'dopsd_cache_teacher_outputs',
                    self.config,
                    'dopsd_cache_teacher_outputs',
                    label_default='Cache D-OPSD Teacher Outputs',
                ), scope="arch_specific")
                if arch_name == "Z-Image":
                    self.config.setdefault('dopsd_teacher_already_reweighted', False)
                    self._set_control("dopsd_teacher_already_reweighted", toggle_switch(
                        'dopsd_teacher_already_reweighted',
                        self.config,
                        'dopsd_teacher_already_reweighted',
                        label_default='Teacher Already Reweighted',
                    ), scope="arch_specific")
                    self.config.setdefault('dopsd_teacher_allow_raw_vlm', False)
                    self._set_control("dopsd_teacher_allow_raw_vlm", toggle_switch(
                        'dopsd_teacher_allow_raw_vlm',
                        self.config,
                        'dopsd_teacher_allow_raw_vlm',
                        label_default='Allow Raw VLM Teacher',
                    ), scope="arch_specific")
                    self.config.setdefault('dopsd_teacher_dtype', 'bfloat16')
                    self._set_control("dopsd_teacher_dtype", ui.select(
                        ['bfloat16', 'float16', 'float32'],
                        label=t('dopsd_teacher_dtype'),
                        value=self.config.get('dopsd_teacher_dtype', 'bfloat16'),
                    ).classes('modern-select force-light-bg dopsd-dtype-select').style(
                        'min-width: 220px; max-width: 300px; flex: 0 1 240px;'
                    ).props('dense stack-label use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"'), scope="arch_specific")

            if arch_name == "Z-Image":
                with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                    self._set_control("dopsd_teacher_text_encoder_path", create_path_selector(
                        label=t('dopsd_teacher_text_encoder'),
                        selection_type='file',
                        file_filter='*.safetensors *.pt *.pth *.bin',
                        placeholder='Qwen3-VL weights file'
                    ), scope="arch_specific")

    def _render_debug_settings(self):
        """渲染调试设置"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('debug_mode')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.debug_mode = ui.select(['', 'image', 'console'], label='调试模式', value='').classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
            self.debug_mode.tooltip(t('debug_mode_tooltip', 'image: save debug images, console: show in terminal'))

            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.config.setdefault('console_width', 80)
                editable_slider(t('console_width'), self.config, 'console_width', min_val=1, max_val=200, step=1, decimals=0)
                self.console_back = ui.select(
                    ['black', 'white'],
                    label='终端背景色',
                    value='black'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.config.setdefault('console_num_images', 16)
                editable_slider(t('console_num_images'), self.config, 'console_num_images', min_val=1, max_val=100, step=1, decimals=0)

    def _on_arch_change(self, arch_name: str, arch_info: dict):
        """架构改变时的处理"""
        version = self._current_model_version(arch_name)
        if arch_name == self._selected_arch and version == self._selected_version:
            return

        self.arch_info = arch_info
        self._selected_arch = arch_name
        self._selected_version = version
        self._clear_control_scope("model_paths")
        self._clear_control_scope("arch_specific")
        self._sync_vae_model_card(arch_name)

        # 重新渲染动态模型路径区域
        if self._model_path_container:
            self._model_path_container.clear()
            with self._model_path_container:
                self._render_dynamic_model_paths(arch_name)

        # 重新渲染架构专属参数区域
        if self._model_specific_container:
            self._model_specific_container.clear()
            with self._model_specific_container:
                self._render_dynamic_arch_specific(arch_name)

        self._apply_model_path_defaults(arch_name, version)

    def _sync_vae_model_card(self, arch_name: str) -> None:
        if self._vae_model_card is None:
            return
        visible = arch_name != "HiDream O1"
        self._vae_model_card.visible = visible
        if not visible and hasattr(self, "vae_path"):
            self._write_control_value(self.vae_path, "")

    def _current_model_version(self, arch_name: str) -> str | None:
        if self.model_selector is not None:
            return self.model_selector.version
        return model_catalog.get_default_version(arch_name, "cache")

    def _apply_model_path_defaults(self, arch_name: str, version: str | None = None) -> None:
        for key, value in model_catalog.get_path_defaults(arch_name, "cache", version).items():
            control = getattr(self, key, None)
            if control is not None:
                self._write_control_value(control, value)

    def _get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self._collect_form_state()

    def _apply_config(self, config: Dict[str, Any]):
        """应用配置"""
        self._apply_form_state(config)

    async def _start_cache(self):
        """开始缓存"""
        try:
            project_config = config_manager.load_project_config(self.project_dir)
            jobs = build_cache_jobs(self._get_config(), self.project_dir, project_config)
        except CommandBuildError as exc:
            ui.notify(str(exc), type='negative')
            return

        for job in jobs:
            result = await self.exec_panel.run_job(
                script_key=job.script_key,
                args=job.args,
                name=job.name,
                runner_kwargs=job.runner_kwargs,
            )
            if result.status != ProcessStatus.SUCCESS:
                return


def render_cache_step():
    """渲染缓存步骤"""
    step = CacheStep()
    step.render()

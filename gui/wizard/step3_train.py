"""步骤 3: 训练 / 微调 - 完整参数"""
from nicegui import ui
from pathlib import Path
from typing import Dict, Any
from theme import get_classes, COLORS
from components.path_selector import create_path_selector
from components.model_selector import create_model_selector, get_arch_info
from components.preset_manager import create_preset_manager
from components.advanced_inputs import editable_slider, toggle_switch, searchable_select
from components.execution_panel import ExecutionPanel
from utils.config_manager import config_manager
from utils.command_builder import CommandBuildError, SCRIPT_DEFAULT_OUTPUT_DIR, build_train_job, get_train_optimizer_template_args
from utils.dataset_config import summarize_dataset_state
from utils.form_state import FormStateMixin
from utils.i18n import t
from utils import model_catalog


# 优化器类型列表
OPTIMIZER_TYPES = [
    'AdamW', 'AdamW8bit', 'PagedAdamW8bit',
    'AdamW_adv', 'Prodigy_adv', 'Adopt_adv', 'Lion_adv', 'Lion_Prodigy_adv',
    'Simplified_AdEMAMix',
    'Prodigy', 'Lion', 'Lion8bit', 'PagedLion8bit',
    'adafactor', 'Sophia', 'Ranger', 'Adan', 'StableAdamW', 'Tiger',
    'AdEMAMix8bit', 'PagedAdEMAMix8bit', 'ademamix',
    'SOAP', 'sgdsai', 'adopt', 'Fira', 'came',
    'LoRARite', 'FlashAdamW', 'DualAdam', 'ROSE',
    'adammini', 'adamg', 'AdaMuon', 'BCOS', 'Ano',
    'EmoNavi', 'EmoFact', 'EmoLynx', 'EmoNeco', 'EmoZeal',
    'DAdaptAdam', 'DAdaptLion', 'DAdaptAdan', 'DAdaptSGD',
    'AdamWScheduleFree', 'SGDScheduleFree',
]

LR_SCHEDULERS = [
    'cosine_with_min_lr', 'cosine', 'cosine_with_restarts',
    'constant', 'constant_with_warmup',
    'linear', 'polynomial',
    'warmup_stable_decay', 'inverse_sqrt',
]

TIMESTEP_SAMPLING_METHODS = [
    'sigma', 'uniform', 'sigmoid', 'shift', 'flux_shift',
    'flux2_shift', 'qwen_shift', 'logsnr', 'qinglong_flux', 'qinglong_qwen',
]

WEIGHTING_SCHEMES = [
    'none', 'sigma_sqrt', 'logit_normal', 'mode', 'cosmap',
]

LYCORIS_ALGOS = [
    'lokr', 'lora', 'locon', 'loha', 'ia3', 'dylora', 'full', 'diag-oft',
]

IDEOGRAM4_SAMPLER_PRESETS = ['V4_QUALITY_48', 'V4_DEFAULT_20', 'V4_TURBO_12']

HIDREAM_TRAIN_VERSION_DEFAULTS = {
    'full': {
        'guidance_scale': 5.0,
        'discrete_flow_shift': 3.0,
        'noise_scale_start': 8.0,
        'noise_scale_end': 8.0,
        'noise_clip_std': 0.0,
    },
    'dev': {
        'guidance_scale': 0.0,
        'discrete_flow_shift': 1.0,
        'noise_scale_start': 7.5,
        'noise_scale_end': 7.5,
        'noise_clip_std': 2.5,
    },
}


class TrainStep(FormStateMixin):
    """训练页面 - 完整参数"""

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.project_dir = Path(__file__).resolve().parents[2]
        self.model_selector = None
        self.train_mode = None
        self.exec_panel = None
        self.arch_info = None
        self._selected_arch = None
        self._selected_version = None
        self._model_path_container = None
        self._vae_path_container = None
        self._tabs = None
        self._tab_basic = None
        self._tab_network = None
        self._tab_lycoris = None
        self._finetune_options_card = None
        self._finetune_disabled_fp8_controls = []
        self._soar_options_card = None
        self._dopsd_options_card = None
        self._dopsd_full_ema_device_container = None
        self._hidream_train_options_card = None
        self._ideogram4_train_options_card = None
        self._init_form_state()

    def render(self):
        """渲染页面"""
        with ui.column().classes(get_classes('page_container') + ' gap-4'):
            # 页面标题
            with ui.row().classes('w-full items-center gap-3 q-mb-sm'):
                ui.icon('model_training', size='32px')
                with ui.column().classes('gap-0'):
                    ui.label(t('train_model', '训练 / 微调')).classes('text-h4 text-weight-bold').style('color: var(--color-text);')
                    ui.label(t('feature_list')['multi_arch']).classes('text-body2').style('color: var(--color-text-secondary);')

            # 预设管理
            create_preset_manager(
                get_config=self._get_config,
                apply_config=self._apply_config,
                scope="train",
            )

            with ui.tabs().classes('w-full') as tabs:
                self._tabs = tabs
                tab_basic = ui.tab(t('basic_settings'), icon='settings')
                tab_model = ui.tab(t('model_settings'), icon='folder')
                tab_training = ui.tab(t('basic_train_params'), icon='trending_up')
                tab_lr = ui.tab(t('lr_settings'), icon='show_chart')
                tab_timestep = ui.tab(t('timestep_sampling'), icon='timeline')
                tab_network = ui.tab(t('network_settings'), icon='hub')
                tab_optimizer = ui.tab(t('optimizer_settings'), icon='speed')
                tab_memory = ui.tab(t('memory_optimization'), icon='memory')
                tab_lycoris = ui.tab(t('lycoris_settings'), icon='extension')
                tab_save = ui.tab(t('save_precision'), icon='save')
                tab_sample = ui.tab(t('sampling_settings'), icon='photo_camera')
                tab_advanced = ui.tab(t('advanced_settings', '高级'), icon='settings_suggest')
                self._tab_basic = tab_basic
                self._tab_network = tab_network
                self._tab_lycoris = tab_lycoris

            with ui.tab_panels(tabs, value=tab_basic).classes('w-full'):
                with ui.tab_panel(tab_basic):
                    self._render_basic_tab()
                with ui.tab_panel(tab_model):
                    self._render_model_tab()
                with ui.tab_panel(tab_training):
                    self._render_training_tab()
                with ui.tab_panel(tab_lr):
                    self._render_lr_tab()
                with ui.tab_panel(tab_timestep):
                    self._render_timestep_tab()
                with ui.tab_panel(tab_network):
                    self._render_network_tab()
                with ui.tab_panel(tab_optimizer):
                    self._render_optimizer_tab()
                with ui.tab_panel(tab_memory):
                    self._render_memory_tab()
                with ui.tab_panel(tab_lycoris):
                    self._render_lycoris_tab()
                with ui.tab_panel(tab_save):
                    self._render_save_tab()
                with ui.tab_panel(tab_sample):
                    self._render_sample_tab()
                with ui.tab_panel(tab_advanced):
                    self._render_advanced_tab()

            # 执行面板 (含 Start/Stop 按钮 + progress bar + 日志)
            self.exec_panel = ExecutionPanel(
                start_label=t('start_train'),
                height='400px',
                on_start=self._start_train,
            )

        self._on_arch_change("FLUX.2", get_arch_info("FLUX.2"))

    def _render_basic_tab(self):
        """基础设置标签"""
        with ui.row().classes('w-full gap-4'):
            self.model_selector = create_model_selector(
                on_change=self._on_arch_change,
                default_arch="FLUX.2",
                page_key="train",
            )
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                with ui.row().classes('w-full items-center gap-2 q-mb-md'):
                    ui.icon('tune', size='24px')
                    ui.label(t('train_mode', '训练模式')).classes('text-h6 text-weight-bold').style('color: var(--color-text);')
                self.train_mode = ui.select(
                    self._train_mode_options("FLUX.2"),
                    label='',
                    value=model_catalog.get_default_train_mode("FLUX.2"),
                    on_change=lambda e: self._on_train_mode_change(e.value),
                ).classes('w-full modern-select force-light-bg')
                self.train_mode.props('dense stack-label dropdown-icon="arrow_drop_down"')

    def _render_model_tab(self):
        """模型路径标签"""
        dataset_summary = summarize_dataset_state(
            config_manager.load_project_config(self.project_dir),
            self.project_dir,
        )

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('dataset_and_model', 'Dataset & Base Model')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            ui.label(
                t('train_dataset_reference_desc', 'Dataset state is owned by Step 1. Training only reads the saved project dataset.')
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
            self.dit_path = create_path_selector(
                label=t('dit_path', 'DiT Model Path'),
                selection_type='file_or_dir',
                placeholder=t('select_dit', 'Select DiT checkpoint')
            )
            with ui.column().classes('w-full') as self._vae_path_container:
                self.vae_path = create_path_selector(
                    label=t('vae_path'),
                    selection_type='file',
                    placeholder=t('select_vae')
                )

        # 动态文本编码器路径
        self._model_path_container = ui.column().classes('w-full gap-3 q-mt-md')
        with self._model_path_container:
            self._render_dynamic_te_paths("FLUX.2")

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('model_output')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            # Initialize config values
            self.config.setdefault('seed', 1026)
            
            with ui.row().classes('w-full gap-4'):
                self.output_name = ui.input(t('output_model_name'), value='flux2_lora').classes('flex-1')
                editable_slider(t('seed'), self.config, 'seed', min_val=0, max_val=9999999999, step=1, decimals=0)
            self.output_dir = create_path_selector(
                label=t('output_dir'),
                default_path=SCRIPT_DEFAULT_OUTPUT_DIR,
                selection_type='dir',
                placeholder=t('output_dir'),
            )

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('resume_training')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.resume_path = create_path_selector(
                label=t('resume_path'),
                selection_type='dir',
                placeholder=t('resume_path_placeholder')
            )
            self.network_weights = create_path_selector(
                label=t('network_weights'),
                selection_type='file',
                placeholder=t('network_weights_placeholder')
            )

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('copy_machine')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.base_weights = ui.input(t('base_weights'), placeholder=t('base_weights_placeholder')).classes('w-full')
            self.base_weights_multiplier = ui.input(t('base_weights_multiplier'), value='1.0', placeholder=t('base_weights_multiplier_placeholder')).classes('w-full')

    def _render_dynamic_te_paths(self, arch_name: str):
        """根据架构渲染文本编码器路径"""
        if arch_name == "HiDream O1":
            return

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('text_encoder_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')

            if arch_name == "FLUX.2":
                self._set_control("text_encoder_path", create_path_selector(
                    label=t('te_mistral_qwen'),
                    selection_type='file',
                    placeholder=t('select_te')
                ), scope="model_paths")
                self.config.setdefault('fp8_text_encoder', False)
                toggle_switch(t('fp8_text_encoder'), self.config, 'fp8_text_encoder')
            elif arch_name == "FLUX Kontext":
                self._set_control("te1_path", create_path_selector(
                    label='Text Encoder 1 (T5-XXL)',
                    selection_type='file', placeholder='t5xxl_fp16.safetensors'
                ), scope="model_paths")
                self._set_control("te2_path", create_path_selector(
                    label=t('te2_clip'),
                    selection_type='file', placeholder='clip_l.safetensors'
                ), scope="model_paths")
                self._set_control("image_encoder_path", create_path_selector(
                    label=t('image_encoder'),
                    selection_type='file', placeholder='sigclip_vision_patch14_384.safetensors'
                ), scope="model_paths")
                self.config.setdefault('fp8_t5', False)
                toggle_switch(t('fp8_t5'), self.config, 'fp8_t5')
            elif arch_name == "HunyuanVideo":
                self._set_control("te1_path", create_path_selector(
                    label=t('te1_llava'),
                    selection_type='file', placeholder='llava_llama3_fp16.safetensors'
                ), scope="model_paths")
                self._set_control("te2_path", create_path_selector(
                    label=t('te2_clip'),
                    selection_type='file', placeholder='clip_l.safetensors'
                ), scope="model_paths")
                self._set_control("t5_path", create_path_selector(
                    label=t('t5_path'),
                    selection_type='file', placeholder='models_t5_umt5-xxl-enc-bf16.pth'
                ), scope="model_paths")
                self._set_control("clip_path", create_path_selector(
                    label=t('clip_path'),
                    selection_type='file', placeholder='models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
                ), scope="model_paths")
                self.config.setdefault('fp8_llm', False)
                toggle_switch(t('fp8_llm'), self.config, 'fp8_llm')
                with ui.row().classes('w-full gap-4 q-mt-md'):
                    self._set_control("dit_dtype", ui.select(
                        ['', 'float16', 'bfloat16', 'float32'],
                        label='DiT Dtype',
                        value=self.config.get('dit_dtype', ''),
                    ).classes('flex-1').props(
                        'use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"'
                    ), scope="model_paths")
                    self._set_control("dit_in_channels", ui.input(
                        'DiT In Channels',
                        value=self.config.get('dit_in_channels', ''),
                        placeholder='16 / 32',
                    ).classes('flex-1'), scope="model_paths")
            elif arch_name == "Wan2.1":
                self._set_control("te1_path", create_path_selector(
                    label=t('te1_llava'),
                    selection_type='file', placeholder='llava_llama3_fp16.safetensors'
                ), scope="model_paths")
                self._set_control("te2_path", create_path_selector(
                    label=t('te2_clip'),
                    selection_type='file', placeholder='clip_l.safetensors'
                ), scope="model_paths")
                self._set_control("t5_path", create_path_selector(
                    label=t('t5_path'),
                    selection_type='file', placeholder='models_t5_umt5-xxl-enc-bf16.pth'
                ), scope="model_paths")
                self._set_control("clip_path", create_path_selector(
                    label=t('clip_path'),
                    selection_type='file', placeholder='open-clip-xlm-roberta-large-vit-huge-14.pth'
                ), scope="model_paths")
                self._set_control("image_encoder_path", create_path_selector(
                    label=t('image_encoder'),
                    selection_type='file', placeholder='sigclip_vision_patch14_384.safetensors'
                ), scope="model_paths")
                self.config.setdefault('fp8_t5', False)
                toggle_switch(t('fp8_t5'), self.config, 'fp8_t5')
            elif arch_name in ("Qwen Image", "Long-CAT"):
                self._set_control("te1_path", create_path_selector(
                    label='Text Encoder 1 (T5-XXL)',
                    selection_type='file', placeholder='t5xxl_fp16.safetensors'
                ), scope="model_paths")
                self._set_control("te2_path", create_path_selector(
                    label=t('te2_clip'),
                    selection_type='file', placeholder='clip_l.safetensors'
                ), scope="model_paths")
                self._set_control("t5_path", create_path_selector(
                    label=t('t5_path'),
                    selection_type='file', placeholder='models_t5_umt5-xxl-enc-bf16.pth'
                ), scope="model_paths")
                self._set_control("clip_path", create_path_selector(
                    label=t('clip_path'),
                    selection_type='file', placeholder='models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'
                ), scope="model_paths")
                self._set_control("image_encoder_path", create_path_selector(
                    label=t('image_encoder'),
                    selection_type='file', placeholder='sigclip_vision_patch14_384.safetensors'
                ), scope="model_paths")
                self._set_control("text_encoder_path", create_path_selector(
                    label=t('te_qwen25'),
                    selection_type='file', placeholder='qwen_2.5_vl_7b.safetensors'
                ), scope="model_paths")
            elif arch_name == "Z-Image":
                self._set_control("text_encoder_path", create_path_selector(
                    label=t('te_qwen3'),
                    selection_type='file', placeholder='qwen_3_4b.safetensors'
                ), scope="model_paths")
            elif arch_name == "Lens":
                self._set_control("text_encoder_path", create_path_selector(
                    label=t('lens_text_encoder', 'Lens Text Encoder'),
                    selection_type='file',
                    file_filter='*.safetensors *.pt *.pth',
                    placeholder='./ckpts/text_encoder/gpt_oss_20b_nvfp4.safetensors'
                ), scope="model_paths")
                with ui.row().classes('w-full gap-4 q-mt-md'):
                    self._set_control("text_encoder_dtype", ui.select(
                        ['', 'bfloat16', 'float16', 'float32'],
                        label=t('text_encoder_dtype', 'Text Encoder Dtype'),
                        value=self.config.get('text_encoder_dtype', 'bfloat16'),
                    ).classes('flex-1').props(
                        'use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"'
                    ), scope="model_paths")
            elif arch_name == "Ideogram-4":
                self._set_control("unconditional_dit_path", create_path_selector(
                    label='Unconditional DiT',
                    selection_type='file',
                    file_filter='*.safetensors *.pt *.pth',
                    placeholder='./ckpts/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors'
                ), scope="model_paths")
                self._set_control("text_encoder_path", create_path_selector(
                    label='Qwen3-VL 8B BF16 Text Encoder',
                    selection_type='file',
                    file_filter='*.safetensors *.pt *.pth',
                    placeholder='./ckpts/text_encoder/qwen3vl_8b_bf16.safetensors'
                ), scope="model_paths")
            elif arch_name == "HV 1.5":
                self._set_control("text_encoder_path", create_path_selector(
                    label=t('te_qwen25'),
                    selection_type='file', placeholder='qwen_2.5_vl_7b.safetensors'
                ), scope="model_paths")
                self._set_control("byt5_path", create_path_selector(
                    label=t('byt5_path'),
                    selection_type='file', placeholder='byt5_model.safetensors'
                ), scope="model_paths")
                self._set_control("image_encoder_path", create_path_selector(
                    label=t('image_encoder'),
                    selection_type='file', placeholder='sigclip_vision_patch14_384.safetensors'
                ), scope="model_paths")
                with ui.row().classes('w-full gap-4 q-mt-md'):
                    self._set_control("dit_dtype", ui.select(
                        ['', 'float16', 'bfloat16', 'float32'],
                        label='DiT Dtype',
                        value=self.config.get('dit_dtype', ''),
                    ).classes('flex-1').props(
                        'use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"'
                    ), scope="model_paths")
            elif arch_name == "FramePack":
                self._set_control("te1_path", create_path_selector(
                    label=t('te1_llava'),
                    selection_type='file', placeholder='llava_llama3_fp16.safetensors'
                ), scope="model_paths")
                self._set_control("te2_path", create_path_selector(
                    label=t('te2_clip'),
                    selection_type='file', placeholder='clip_l.safetensors'
                ), scope="model_paths")
                self._set_control("image_encoder_path", create_path_selector(
                    label=t('image_encoder'),
                    selection_type='file', placeholder='sigclip_vision_patch14_384.safetensors'
                ), scope="model_paths")

    def _render_training_tab(self):
        """训练参数标签"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('training_params')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            # Initialize config values
            self.config.setdefault('max_train_epochs', 20)
            self.config.setdefault('gradient_accumulation_steps', 1)
            
            with ui.row().classes('w-full gap-4'):
                editable_slider(t('max_train_epochs'), self.config, 'max_train_epochs', min_val=1, max_val=1000, step=1)
                self.max_train_steps = ui.input(t('max_train_steps'), placeholder=t('max_train_steps_placeholder')).classes('flex-1')
                editable_slider(t('gradient_accumulation_steps'), self.config, 'gradient_accumulation_steps', min_val=1, max_val=128, step=1)
            # Initialize config value
            self.config.setdefault('guidance_scale', 1.0)
            
            with ui.row().classes('w-full gap-4 q-mt-md'):
                self._set_control(
                    "guidance_scale",
                    editable_slider('Guidance Scale', self.config, 'guidance_scale', min_val=0, max_val=20, step=0.1, decimals=1, label_default='Guidance Scale'),
                )

    def _render_lr_tab(self):
        """学习率标签"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('lr_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                self.learning_rate = ui.input(t('learning_rate'), value='1e-4').classes('flex-1')
                self.lr_scheduler = ui.select(
                    LR_SCHEDULERS, label=t('lr_scheduler'), value='cosine_with_min_lr'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
            # Initialize config values
            self.config.setdefault('lr_warmup_steps', 0)
            self.config.setdefault('lr_scheduler_num_cycles', 1)
            self.config.setdefault('lr_scheduler_power', 1)
            self.config.setdefault('lr_scheduler_timescale', 0)
            self.config.setdefault('lr_scheduler_min_lr_ratio', 0.1)
            
            with ui.row().classes('w-full gap-4 q-mt-md'):
                editable_slider(t('lr_warmup_steps'), self.config, 'lr_warmup_steps', min_val=0, max_val=10000, step=100, decimals=0)
                self.lr_decay_steps = ui.input(t('lr_decay_steps'), value='0.2').classes('flex-1')
                self.lr_decay_steps.tooltip(t('lr_decay_steps_tooltip'))
            with ui.row().classes('w-full gap-4 q-mt-md'):
                editable_slider(t('lr_num_cycles'), self.config, 'lr_scheduler_num_cycles', min_val=1, max_val=10, step=1)
                editable_slider(t('lr_power'), self.config, 'lr_scheduler_power', min_val=1, max_val=10, step=1)
                editable_slider(t('lr_timescale'), self.config, 'lr_scheduler_timescale', min_val=0, max_val=10000, step=100)
                editable_slider(t('lr_min_ratio'), self.config, 'lr_scheduler_min_lr_ratio', min_val=0, max_val=1, step=0.01, decimals=2)

    def _render_timestep_tab(self):
        """时间步采样标签"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('timestep_sampling')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            # Initialize config values
            self.config.setdefault('discrete_flow_shift', 7.0)
            self.config.setdefault('sigmoid_scale', 1.0)
            
            with ui.row().classes('w-full gap-4'):
                self.timestep_sampling = ui.select(
                    TIMESTEP_SAMPLING_METHODS, label=t('sampling_method'), value='shift'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                editable_slider('Discrete Flow Shift', self.config, 'discrete_flow_shift', min_val=0, max_val=20, step=0.1, decimals=1, label_default='Discrete Flow Shift')
                editable_slider('Sigmoid Scale', self.config, 'sigmoid_scale', min_val=0.1, max_val=10, step=0.1, decimals=1, label_default='Sigmoid Scale')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('weighting_scheme')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            # Initialize config values
            self.config.setdefault('logit_mean', 0.0)
            self.config.setdefault('logit_std', 1.0)
            self.config.setdefault('mode_scale', 1.29)
            self.config.setdefault('min_timestep', 0)
            self.config.setdefault('max_timestep', 1000)
            
            with ui.row().classes('w-full gap-4'):
                self.weighting_scheme = ui.select(
                    [''] + WEIGHTING_SCHEMES, label=t('weighting_scheme'), value=''
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                editable_slider('Logit Mean', self.config, 'logit_mean', min_val=-10, max_val=10, step=0.1, decimals=1, label_default='Logit Mean')
                editable_slider('Logit Std', self.config, 'logit_std', min_val=0.1, max_val=5, step=0.1, decimals=1, label_default='Logit Std')
                editable_slider('Mode Scale', self.config, 'mode_scale', min_val=0.5, max_val=3, step=0.01, decimals=2, label_default='Mode Scale')
            with ui.row().classes('w-full gap-4 q-mt-md'):
                editable_slider(t('min_timestep'), self.config, 'min_timestep', min_val=0, max_val=1000, step=1)
                editable_slider(t('max_timestep'), self.config, 'max_timestep', min_val=0, max_val=1000, step=1)
                self.show_timesteps = ui.select(['', 'console', 'image'], label=t('show_timesteps'), value='').classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')

        self.config.setdefault('noise_scale_start', 8.0)
        self.config.setdefault('noise_scale_end', 8.0)
        self.config.setdefault('noise_clip_std', 0.0)
        self.config.setdefault('dino_loss_weight', 0.0)
        self.config.setdefault('dino_loss_backend', 'vit')
        self.config.setdefault('dino_loss_model_type', '')
        self.config.setdefault('dino_loss_layer', '')
        self.config.setdefault('dino_loss_feature_mode', 'patch')
        self.config.setdefault('dino_loss_resize', 224)
        self.config.setdefault('dino_loss_every_n_steps', 1)
        self.config.setdefault('dino_loss_use_gram', False)
        self.config.setdefault('dino_loss_no_norm', False)
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md') as self._hidream_train_options_card:
            ui.label(t('arch_specific_params').format(arch='HiDream O1')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                editable_slider(
                    'Noise Scale Start',
                    self.config,
                    'noise_scale_start',
                    min_val=0,
                    max_val=20,
                    step=0.1,
                    decimals=1,
                    label_default='Noise Scale Start',
                )
                editable_slider(
                    'Noise Scale End',
                    self.config,
                    'noise_scale_end',
                    min_val=0,
                    max_val=20,
                    step=0.1,
                    decimals=1,
                    label_default='Noise Scale End',
                )
                editable_slider(
                    'Noise Clip Std',
                    self.config,
                    'noise_clip_std',
                    min_val=0,
                    max_val=10,
                    step=0.1,
                    decimals=1,
                    label_default='Noise Clip Std',
                )
            ui.label('Full defaults: start/end 8.0 and clip 0.0. Dev defaults: start/end 7.5 and clip 2.5.').classes('text-caption q-mt-sm').style('color: var(--color-text-secondary);')
            ui.separator().classes('q-my-md')
            ui.label('DINOv3 Auxiliary Loss').classes('text-subtitle1 text-weight-medium').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4 q-mt-sm'):
                editable_slider(
                    'DINO Loss Weight',
                    self.config,
                    'dino_loss_weight',
                    min_val=0,
                    max_val=1,
                    step=0.001,
                    decimals=3,
                    label_default='DINO Loss Weight',
                )
                self.dino_loss_backend = ui.select(
                    ['vit', 'convnext'],
                    label='DINO Backend',
                    value=self.config.get('dino_loss_backend', 'vit'),
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.dino_loss_feature_mode = ui.select(
                    ['all', 'patch', 'cls', 'both'],
                    label='DINO Feature Mode',
                    value=self.config.get('dino_loss_feature_mode', 'patch'),
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.dino_loss_model_type = ui.input(
                    'DINO Model Type',
                    value=self.config.get('dino_loss_model_type', ''),
                    placeholder='small / tiny / ...',
                ).classes('flex-1')
                self.dino_loss_layer = ui.input(
                    'DINO Layer',
                    value=self.config.get('dino_loss_layer', ''),
                    placeholder='default: -4 for vit, -1 for convnext',
                ).classes('flex-1')
                editable_slider(
                    'DINO Resize',
                    self.config,
                    'dino_loss_resize',
                    min_val=0,
                    max_val=1024,
                    step=16,
                    decimals=0,
                    label_default='DINO Resize',
                )
                editable_slider(
                    'DINO Every N Steps',
                    self.config,
                    'dino_loss_every_n_steps',
                    min_val=1,
                    max_val=100,
                    step=1,
                    decimals=0,
                    label_default='DINO Every N Steps',
                )
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                toggle_switch('DINO Gram Loss', self.config, 'dino_loss_use_gram', label_default='DINO Gram Loss')
                toggle_switch('DINO No Norm', self.config, 'dino_loss_no_norm', label_default='DINO No Norm')
            ui.label('DINO is only added to the launch command when weight is greater than 0.').classes('text-caption q-mt-sm').style('color: var(--color-text-secondary);')
            self._hidream_train_options_card.visible = False

        self.config.setdefault('sampler_preset', 'V4_DEFAULT_20')
        self.config.setdefault('initial_sigma', 1.004)
        self.config.setdefault('ideogram4_timestep_mu', 0.0)
        self.config.setdefault('ideogram4_timestep_std', 1.0)
        self.config.setdefault('warn_on_caption_issues', False)
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md') as self._ideogram4_train_options_card:
            ui.label(t('arch_specific_params').format(arch='Ideogram-4')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                self._set_control("sampler_preset", ui.select(
                    IDEOGRAM4_SAMPLER_PRESETS,
                    label='Sampler Preset',
                    value=self.config.get('sampler_preset', 'V4_DEFAULT_20'),
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"'))
                editable_slider('Initial Sigma', self.config, 'initial_sigma', min_val=0.9, max_val=1.1, step=0.0005, decimals=4, label_default='Initial Sigma')
                editable_slider('Ideogram Timestep Mu', self.config, 'ideogram4_timestep_mu', min_val=-5, max_val=5, step=0.1, decimals=1, label_default='Ideogram Timestep Mu')
                editable_slider('Ideogram Timestep Std', self.config, 'ideogram4_timestep_std', min_val=0.1, max_val=5, step=0.1, decimals=1, label_default='Ideogram Timestep Std')
                toggle_switch('Warn On Caption Issues', self.config, 'warn_on_caption_issues', label_default='Warn On Caption Issues')
            self._ideogram4_train_options_card.visible = False

        self.config.setdefault('soar', False)
        self.config.setdefault('soar_lambda_aux', 1.0)
        self.config.setdefault('soar_trajectory_length', 6)
        self.config.setdefault('soar_num_sampling_steps', 40)
        self.config.setdefault('soar_sigma_upper_ratio', 1.5)
        self.config.setdefault('soar_cfg_scale_sampling', 4.5)
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md') as self._soar_options_card:
            ui.label('SOAR').classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                toggle_switch('enable_soar', self.config, 'soar', label_default='Enable SOAR')
                editable_slider('soar_lambda_aux', self.config, 'soar_lambda_aux', min_val=0, max_val=10, step=0.1, decimals=2, label_default='SOAR Lambda Aux')
                editable_slider('soar_trajectory_length', self.config, 'soar_trajectory_length', min_val=1, max_val=32, step=1, label_default='SOAR Trajectory Length')
                editable_slider('soar_num_sampling_steps', self.config, 'soar_num_sampling_steps', min_val=2, max_val=200, step=1, label_default='SOAR Sampling Steps')
                editable_slider('soar_sigma_upper_ratio', self.config, 'soar_sigma_upper_ratio', min_val=1, max_val=4, step=0.1, decimals=2, label_default='SOAR Sigma Upper Ratio')
                editable_slider('soar_cfg_scale_sampling', self.config, 'soar_cfg_scale_sampling', min_val=0.1, max_val=8, step=0.1, decimals=2, label_default='SOAR CFG Scale Sampling')
        self._sync_soar_options_ui()

        self.config.setdefault('dopsd', False)
        self.config.setdefault('dopsd_loss_weight', 1.0)
        self.config.setdefault('dopsd_num_sampling_steps', 8)
        self.config.setdefault('dopsd_ema_decay', 0.9999)
        self.config.setdefault('dopsd_full_ema_device', 'auto')
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md') as self._dopsd_options_card:
            ui.label('D-OPSD').classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                toggle_switch('enable_dopsd', self.config, 'dopsd', label_default='Enable D-OPSD')
                editable_slider('dopsd_loss_weight', self.config, 'dopsd_loss_weight', min_val=0.01, max_val=10, step=0.01, decimals=2, label_default='D-OPSD Loss Weight')
                editable_slider('dopsd_num_sampling_steps', self.config, 'dopsd_num_sampling_steps', min_val=1, max_val=64, step=1, label_default='D-OPSD Sampling Steps')
                editable_slider('dopsd_ema_decay', self.config, 'dopsd_ema_decay', min_val=0, max_val=1, step=0.0001, decimals=4, label_default='D-OPSD EMA Decay')
                with ui.element('div').classes('dopsd-full-ema-device').style('min-width: 180px; flex: 0 1 220px;') as self._dopsd_full_ema_device_container:
                    self._set_control("dopsd_full_ema_device", ui.select(
                        {'auto': 'auto', 'cpu': 'cpu', 'gpu': 'gpu'},
                        label=t('dopsd_full_ema_device', 'D-OPSD Full EMA Device'),
                        value=self.config.get('dopsd_full_ema_device', 'auto'),
                    ).classes('w-full modern-select force-light-bg').props(
                        'dense stack-label use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"'
                    ))
        self._sync_dopsd_options_ui()

    def _render_network_tab(self):
        """网络结构标签"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('lora_network')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            # Initialize config values
            self.config.setdefault('network_dim', 32)
            self.config.setdefault('network_alpha', 16)
            self.config.setdefault('network_dropout', 0)
            self.config.setdefault('scale_weight_norms', 0)
            
            with ui.row().classes('w-full gap-4'):
                editable_slider(t('network_dim'), self.config, 'network_dim', min_val=1, max_val=512, step=1)
                editable_slider(t('network_alpha'), self.config, 'network_alpha', min_val=0, max_val=512, step=1)
                editable_slider('Dropout', self.config, 'network_dropout', min_val=0, max_val=1, step=0.01, decimals=2, label_default='Dropout')
            with ui.row().classes('w-full gap-4 q-mt-md'):
                editable_slider(t('scale_weight_norms'), self.config, 'scale_weight_norms', min_val=0, max_val=10, step=0.1, decimals=1)
                self.scale_weight_norms_tooltip = ui.icon('help_outline', size='20px').style('cursor: help;')
                self.scale_weight_norms_tooltip.tooltip(t('scale_weight_norms_tooltip'))
                self.config.setdefault('dim_from_weights', False)
                toggle_switch(t('dim_from_weights'), self.config, 'dim_from_weights')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('lora_plus')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            # Initialize config value
            self.config.setdefault('loraplus_lr_ratio', 4)
            
            with ui.row().classes('w-full gap-4'):
                self.config.setdefault('enable_lora_plus', False)
                toggle_switch(t('enable_lora_plus'), self.config, 'enable_lora_plus')
                editable_slider('LoRA+ LR Ratio', self.config, 'loraplus_lr_ratio', min_val=1, max_val=64, step=1, label_default='LoRA+ LR Ratio')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('target_blocks')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                self.config.setdefault('enable_blocks', False)
                toggle_switch(t('enable_blocks'), self.config, 'enable_blocks')
            self.exclude_patterns = ui.input(
                'Exclude Patterns',
                placeholder="如: exclude_patterns=[r'.*single_blocks.*']"
            ).classes('w-full q-mt-sm')
            self.include_patterns = ui.input(
                'Include Patterns',
                placeholder="如: include_patterns=[r'.*single_blocks\\.\\d{2}\\.linear.*']"
            ).classes('w-full q-mt-sm')

    def _render_optimizer_tab(self):
        """优化器标签"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('optimizer_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            # Initialize config value
            self.config.setdefault('max_grad_norm', 1.0)
            
            with ui.row().classes('w-full gap-4'):
                self.optimizer_type = ui.select(
                    OPTIMIZER_TYPES, label=t('optimizer_type'), value='AdamW_adv',
                    on_change=lambda e: self._set_optimizer_args_template(force=True),
                ).classes('flex-1')
                self.optimizer_type.props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                editable_slider(t('max_grad_norm'), self.config, 'max_grad_norm', min_val=0, max_val=10, step=0.1, decimals=1)
            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.d_coef = ui.input(t('d_coef'), value='0.5').classes('flex-1')
                self.d_coef.tooltip(t('d_coef_tooltip'))
                self.d0 = ui.input(t('d0'), value='1e-3').classes('flex-1')
                self.d0.tooltip(t('d0_tooltip'))
            with ui.row().classes('w-full items-start gap-2 q-mt-md'):
                self.optimizer_extra_args = ui.textarea(
                    t('optimizer_extra_args', 'Optimizer Args'),
                    value='',
                    placeholder='key=value',
                ).classes('flex-1').props('autogrow outlined')
                ui.button(
                    icon='restart_alt',
                    on_click=lambda: self._set_optimizer_args_template(force=True),
                ).classes('modern-btn-ghost').props('dense').tooltip(t('reset_optimizer_template', 'Reset template'))
            self._set_optimizer_args_template(force=True)

    def _render_memory_tab(self):
        """内存优化标签"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('precision_memory')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                self.mixed_precision = ui.select(
                    ['bf16', 'fp16'], label=t('mixed_precision'), value='bf16'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.attn_mode = ui.select(
                    ['flash', 'xformers', 'sdpa', 'sageattn', 'flash3'],
                    label=t('attn_mode'), value='flash'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.vae_dtype = ui.select(
                    ['', 'float32', 'float16', 'bfloat16'],
                    label=t('vae_dtype'), value=''
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
            # Initialize config defaults
            self.config.setdefault('gradient_checkpointing', True)
            self.config.setdefault('gradient_checkpointing_cpu_offload', False)
            self.config.setdefault('fp8_base', False)
            self.config.setdefault('fp8_scaled', False)
            self.config.setdefault('split_attn', True)
            
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                toggle_switch(t('gradient_checkpointing'), self.config, 'gradient_checkpointing', 
                            label_default='Gradient Checkpointing')
                toggle_switch(t('gradient_checkpointing_cpu'), self.config, 'gradient_checkpointing_cpu_offload',
                            label_default='CPU Offload')
                self._finetune_disabled_fp8_controls = [
                    toggle_switch(t('fp8_base'), self.config, 'fp8_base', label_default='FP8 Base'),
                    toggle_switch(t('fp8_scaled'), self.config, 'fp8_scaled', label_default='FP8 Scaled'),
                ]
                toggle_switch('Split Attention', self.config, 'split_attn', label_default='Split Attention')

        self.config.setdefault('full_bf16', False)
        self.config.setdefault('fused_backward_pass', False)
        self.config.setdefault('mem_eff_save', False)
        self.config.setdefault('block_swap_optimizer_patch_params', False)
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md') as self._finetune_options_card:
            ui.label(t('finetune_settings', '微调专用')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                toggle_switch(t('full_bf16'), self.config, 'full_bf16', label_default='Full BF16')
                toggle_switch(t('fused_backward_pass', 'Fused Backward'), self.config, 'fused_backward_pass', label_default='Fused Backward')
                toggle_switch(t('mem_eff_save', '内存优化保存'), self.config, 'mem_eff_save', label_default='Memory Efficient Save')
                toggle_switch(
                    t('block_swap_optimizer_patch_params', 'Block Swap 优化器补丁'),
                    self.config,
                    'block_swap_optimizer_patch_params',
                    label_default='Block Swap Optimizer Patch',
                )

        # Initialize config value
        self.config.setdefault('blocks_to_swap', 0)
        
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('model_offload')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                editable_slider(t('blocks_to_swap'), self.config, 'blocks_to_swap', min_val=0, max_val=40, step=1)
                
                # Initialize defaults
                self.config.setdefault('use_pinned_memory', True)
                self.config.setdefault('img_in_txt_in_offloading', True)
                
                toggle_switch(t('use_pinned_memory'), self.config, 'use_pinned_memory', 
                            label_default='Pinned Memory')
                toggle_switch(t('img_in_txt_in_offloading'), self.config, 'img_in_txt_in_offloading',
                            label_default='img_in/txt_in Offloading')

        # Initialize config value
        self.config.setdefault('max_data_loader_n_workers', 8)
        
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('data_loading')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                editable_slider(t('max_data_loader_workers'), self.config, 'max_data_loader_n_workers', min_val=0, max_val=16, step=1)
                
                # Initialize default
                self.config.setdefault('persistent_workers', True)
                toggle_switch(t('persistent_workers'), self.config, 'persistent_workers',
                            label_default='Persistent Workers')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('compile_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            
            # Initialize defaults
            self.config.setdefault('compile', False)
            self.config.setdefault('compile_fullgraph', False)
            self.config.setdefault('cuda_allow_tf32', True)
            self.config.setdefault('cuda_cudnn_benchmark', True)
            
            with ui.row().classes('w-full gap-4'):
                toggle_switch(t('compile'), self.config, 'compile', label_default='Enable Compile')
                self.compile_backend = ui.select(
                    ['inductor', 'aot_eager', 'cudagraphs'],
                    label=t('compile_backend'), value='inductor'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.compile_mode = ui.select(
                    ['default', 'reduce-overhead', 'max-autotune-no-cudagraphs'],
                    label=t('compile_mode'), value='default'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
            # Initialize config value
            self.config.setdefault('compile_cache_size_limit', 32)
            
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                toggle_switch('Fullgraph', self.config, 'compile_fullgraph', label_default='Fullgraph')
                self.compile_dynamic = ui.select(
                    ['auto', 'true', 'false'],
                    label='Dynamic', value='auto'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                editable_slider(t('compile_cache_size'), self.config, 'compile_cache_size_limit', min_val=1, max_val=128, step=1)
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                toggle_switch(t('cuda_allow_tf32'), self.config, 'cuda_allow_tf32', label_default='Allow TF32')
                toggle_switch(t('cuda_cudnn_benchmark'), self.config, 'cuda_cudnn_benchmark', label_default='cuDNN Benchmark')

    def _render_lycoris_tab(self):
        """LyCORIS 标签"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('lycoris_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            ui.label(t('lycoris_desc')).classes('text-caption q-mb-md').style('color: var(--color-text-muted);')
            
            # Initialize default
            self.config.setdefault('enable_lycoris', False)
            toggle_switch(t('enable_lycoris'), self.config, 'enable_lycoris', label_default='Enable LyCORIS')

            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.lycoris_algo = ui.select(
                    LYCORIS_ALGOS, label=t('lycoris_algo'), value='lokr'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.lycoris_preset = ui.select(
                    ['attn-mlp', 'full', 'full-lin', 'attn-only',
                     'unet-transformer-only', 'unet-convblock-only'],
                    label=t('lycoris_preset'), value='attn-mlp'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')

            # Initialize config values
            self.config.setdefault('lycoris_conv_dim', 0)
            self.config.setdefault('lycoris_conv_alpha', 0)
            self.config.setdefault('lycoris_factor', 8)
            self.config.setdefault('lycoris_dropout', 0)
            self.config.setdefault('lycoris_block_size', 4)
            self.config.setdefault('lycoris_rescaled', 1)
            
            with ui.row().classes('w-full gap-4 q-mt-md'):
                editable_slider(t('lycoris_conv_dim'), self.config, 'lycoris_conv_dim', min_val=0, max_val=256, step=1)
                editable_slider(t('lycoris_conv_alpha'), self.config, 'lycoris_conv_alpha', min_val=0, max_val=256, step=0.1, decimals=1)
                editable_slider(t('lycoris_factor'), self.config, 'lycoris_factor', min_val=-1, max_val=8, step=1)
                editable_slider('Dropout', self.config, 'lycoris_dropout', min_val=0, max_val=1, step=0.01, decimals=2, label_default='Dropout')

            with ui.row().classes('w-full gap-4 q-mt-md'):
                editable_slider(t('lycoris_block_size'), self.config, 'lycoris_block_size', min_val=1, max_val=16, step=1)
                editable_slider(t('lycoris_rescaled'), self.config, 'lycoris_rescaled', min_val=0, max_val=10, step=1)

            # Initialize LyCORIS checkbox defaults
            self.config.setdefault('lycoris_use_tucker', False)
            self.config.setdefault('lycoris_use_scalar', False)
            self.config.setdefault('lycoris_train_norm', False)
            self.config.setdefault('lycoris_dora_wd', True)
            self.config.setdefault('lycoris_full_matrix', False)
            self.config.setdefault('lycoris_bypass_mode', False)
            self.config.setdefault('lycoris_decompose_both', False)
            self.config.setdefault('lycoris_constrain', False)
            
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                toggle_switch('Use Tucker', self.config, 'lycoris_use_tucker', label_default='Use Tucker')
                toggle_switch('Use Scalar', self.config, 'lycoris_use_scalar', label_default='Use Scalar')
                toggle_switch('Train Norm', self.config, 'lycoris_train_norm', label_default='Train Norm')
                toggle_switch('DoRA WD', self.config, 'lycoris_dora_wd', label_default='DoRA WD')
                toggle_switch('Full Matrix', self.config, 'lycoris_full_matrix', label_default='Full Matrix')
                toggle_switch('Bypass Mode', self.config, 'lycoris_bypass_mode', label_default='Bypass Mode')
                toggle_switch('Decompose Both (LoKr)', self.config, 'lycoris_decompose_both', label_default='Decompose Both')
                toggle_switch('Constrain (COFT)', self.config, 'lycoris_constrain', label_default='Constrain')

    def _render_save_tab(self):
        """保存设置标签"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('save_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                self.save_every_n_epochs = ui.input(t('save_every_n_epochs'), value='2').classes('flex-1')
                self.save_every_n_steps = ui.input(t('save_every_n_steps'), placeholder=t('save_every_n_steps_placeholder')).classes('flex-1')
            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.save_last_n_epochs = ui.input(t('save_last_n_epochs'), placeholder=t('save_last_n_placeholder')).classes('flex-1')
                self.save_last_n_steps = ui.input(t('save_last_n_steps'), placeholder=t('save_last_n_placeholder')).classes('flex-1')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('state_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            
            # Initialize defaults
            self.config.setdefault('save_state', False)
            self.config.setdefault('save_state_on_train_end', False)
            
            with ui.row().classes('w-full gap-4 flex-wrap'):
                toggle_switch(t('save_state'), self.config, 'save_state', label_default='Save State')
                toggle_switch(t('save_state_on_end'), self.config, 'save_state_on_train_end', label_default='Save on End')
            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.save_last_n_epochs_state = ui.input(t('save_last_n_epochs_state'), placeholder=t('optional')).classes('flex-1')
                self.save_last_n_steps_state = ui.input(t('save_last_n_steps_state'), placeholder=t('optional')).classes('flex-1')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('metadata')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.training_comment = ui.input(
                t('training_comment'), value="this LoRA model created by bdsqlsz's script"
            ).classes('w-full')
            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.metadata_title = ui.input(t('metadata_title'), placeholder=t('optional')).classes('flex-1')
                self.metadata_author = ui.input(t('metadata_author'), placeholder=t('optional')).classes('flex-1')
            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.metadata_description = ui.input(t('metadata_description'), placeholder=t('optional')).classes('flex-1')
                self.metadata_license = ui.input(t('metadata_license'), placeholder=t('optional')).classes('flex-1')
                self.metadata_tags = ui.input(t('metadata_tags'), placeholder=t('optional')).classes('flex-1')

    def _render_sample_tab(self):
        """采样输出标签"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('sample_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            ui.label(t('sample_desc')).classes('text-caption q-mb-md').style('color: var(--color-text-muted);')
            self.config.setdefault('enable_sample', True)
            self.config.setdefault('sample_at_first', True)
            with ui.row().classes('w-full gap-4 flex-wrap'):
                toggle_switch(t('enable_sample'), self.config, 'enable_sample')
                toggle_switch(t('sample_at_first'), self.config, 'sample_at_first')
            # Initialize config values
            self.config.setdefault('sample_every_n_epochs', 1)
            self.config.setdefault('sample_every_n_steps', 0)
            
            with ui.row().classes('w-full gap-4 q-mt-md'):
                editable_slider(t('sample_every_n_epochs'), self.config, 'sample_every_n_epochs', min_val=1, max_val=100, step=1)
                editable_slider(t('sample_every_n_steps'), self.config, 'sample_every_n_steps', min_val=0, max_val=1000, step=10)
            self.sample_prompts = create_path_selector(
                label=t('sample_prompts'),
                selection_type='file',
                file_filter='*.txt',
                placeholder=t('sample_prompts_placeholder')
            )

    def _render_advanced_tab(self):
        """高级设置标签"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('wandb_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.wandb_api_key = ui.input(t('wandb_api_key'), placeholder=t('optional'), password=True).classes('w-full')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('huggingface_upload')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            
            # Initialize default
            self.config.setdefault('async_upload', False)
            toggle_switch(t('async_upload'), self.config, 'async_upload', label_default='Async Upload')
            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.huggingface_repo_id = ui.input(t('huggingface_repo_id'), placeholder='user/repo').classes('flex-1')
                self.huggingface_token = ui.input(t('huggingface_token'), placeholder=t('optional'), password=True).classes('flex-1')
            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.huggingface_repo_type = ui.select(
                    ['', 'model', 'dataset'], label=t('huggingface_repo_type'), value=''
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.huggingface_path_in_repo = ui.input(t('huggingface_path_in_repo'), placeholder=t('optional')).classes('flex-1')
                self.huggingface_repo_visibility = ui.select(
                    ['', 'public', 'private'], label=t('huggingface_visibility'), value=''
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
            # Initialize defaults
            self.config.setdefault('save_state_to_huggingface', False)
            self.config.setdefault('resume_from_huggingface', False)
            
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                toggle_switch(t('save_state_to_huggingface'), self.config, 'save_state_to_huggingface', label_default='Upload State')
                toggle_switch(t('resume_from_huggingface'), self.config, 'resume_from_huggingface', label_default='Resume from HF')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('multi_gpu')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            
            # Initialize defaults
            self.config.setdefault('multi_gpu', False)
            self.config.setdefault('ddp_gradient_as_bucket_view', True)
            self.config.setdefault('ddp_static_graph', True)
            
            # Initialize config value
            self.config.setdefault('ddp_timeout', 120)
            
            toggle_switch(t('enable_multi_gpu'), self.config, 'multi_gpu', label_default='Enable Multi GPU')
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                editable_slider(t('ddp_timeout'), self.config, 'ddp_timeout', min_val=30, max_val=600, step=10)
                toggle_switch(t('ddp_gradient_bucket'), self.config, 'ddp_gradient_as_bucket_view', label_default='Gradient Bucket')
                toggle_switch(t('ddp_static_graph'), self.config, 'ddp_static_graph', label_default='Static Graph')

    def _on_arch_change(self, arch_name: str, arch_info: dict):
        """架构改变时的处理"""
        version = self._current_model_version(arch_name)
        if arch_name == self._selected_arch and version == self._selected_version:
            self._refresh_train_mode_options(arch_name)
            return

        self.arch_info = arch_info
        self._selected_arch = arch_name
        self._selected_version = version
        self._refresh_train_mode_options(arch_name)
        self._clear_control_scope("model_paths")
        self._sync_vae_path_ui(arch_name)
        if self._model_path_container:
            self._model_path_container.clear()
            with self._model_path_container:
                self._render_dynamic_te_paths(arch_name)

        self._apply_model_path_defaults(arch_name, version)
        self._apply_lens_train_defaults(arch_name)
        self._apply_hidream_train_version_defaults(arch_name, version)
        self._sync_hidream_train_options_ui()
        self._sync_ideogram4_train_options_ui()

    def _sync_vae_path_ui(self, arch_name: str) -> None:
        if self._vae_path_container is None:
            return
        visible = arch_name != "HiDream O1"
        self._vae_path_container.visible = visible
        if not visible and hasattr(self, "vae_path"):
            self._write_control_value(self.vae_path, "")

    def _current_model_version(self, arch_name: str) -> str | None:
        if self.model_selector is not None:
            return self.model_selector.version
        return model_catalog.get_default_version(arch_name, "train")

    def _apply_model_path_defaults(self, arch_name: str, version: str | None = None) -> None:
        for key, value in model_catalog.get_path_defaults(arch_name, "train", version).items():
            control = getattr(self, key, None)
            if control is not None:
                self._write_control_value(control, value)

    def _apply_lens_train_defaults(self, arch_name: str) -> None:
        if arch_name != "Lens":
            return

        defaults = {
            'split_attn': False,
            'blocks_to_swap': 0,
        }
        self.config.update(defaults)
        self._write_bound_control_values(defaults)
        if hasattr(self, 'attn_mode'):
            self._write_control_value(self.attn_mode, 'sdpa')
        if hasattr(self, 'vae_dtype'):
            self._write_control_value(self.vae_dtype, 'float32')

    def _apply_hidream_train_version_defaults(self, arch_name: str, version: str | None = None) -> None:
        if arch_name != "HiDream O1":
            return
        version_key = str(version or "full").strip().lower()
        defaults = HIDREAM_TRAIN_VERSION_DEFAULTS.get(version_key)
        if defaults is None:
            return

        for key, value in defaults.items():
            self.config[key] = value
        self._write_bound_control_values(defaults)
        for key, value in defaults.items():
            control = getattr(self, key, None)
            if control is not None:
                self._write_control_value(control, value)

    def _train_mode_options(self, arch_name: str) -> Dict[str, str]:
        label_by_mode = {
            'lora': t('train_mode_lora', 'LoRA'),
            'finetune': t('train_mode_finetune', 'Fine-tune'),
        }
        return {
            mode: label_by_mode.get(mode, label)
            for mode, label in model_catalog.get_train_modes(arch_name).items()
        }

    def _refresh_train_mode_options(self, arch_name: str) -> None:
        if self.train_mode is None:
            return
        options = self._train_mode_options(arch_name)
        current_value = self.train_mode.value
        self.train_mode.options = options
        if current_value not in options:
            self.train_mode.value = model_catalog.get_default_train_mode(arch_name)
        self.train_mode.update()
        self._sync_train_mode_ui()

    def _on_train_mode_change(self, _value: str) -> None:
        self._sync_train_mode_ui()

    def _sync_train_mode_ui(self) -> None:
        mode = self.train_mode.value if self.train_mode is not None else 'lora'
        is_lora = mode == 'lora'
        if self._tab_network is not None:
            self._tab_network.visible = is_lora
        if self._tab_lycoris is not None:
            self._tab_lycoris.visible = is_lora
        if self._finetune_options_card is not None:
            self._finetune_options_card.visible = not is_lora
        for control in self._finetune_disabled_fp8_controls:
            control.visible = is_lora
            if not is_lora and hasattr(control, 'set_toggle_value'):
                control.set_toggle_value(False)
        self._sync_soar_options_ui()
        self._sync_dopsd_options_ui()
        self._sync_hidream_train_options_ui()
        self._sync_ideogram4_train_options_ui()

    def _sync_hidream_train_options_ui(self) -> None:
        if self._hidream_train_options_card is None:
            return
        arch_name = self._selected_arch or 'FLUX.2'
        self._hidream_train_options_card.visible = arch_name == "HiDream O1"

    def _sync_ideogram4_train_options_ui(self) -> None:
        if self._ideogram4_train_options_card is None:
            return
        arch_name = self._selected_arch or 'FLUX.2'
        self._ideogram4_train_options_card.visible = arch_name == "Ideogram-4"

    def _sync_soar_options_ui(self) -> None:
        if self._soar_options_card is None:
            return
        arch_name = self._selected_arch or 'FLUX.2'
        mode = self.train_mode.value if self.train_mode is not None else model_catalog.get_default_train_mode(arch_name)
        version = self.model_selector.version if self.model_selector is not None else None
        visible = model_catalog.supports_soar_training(arch_name, mode, version=version)
        self._soar_options_card.visible = visible
        if not visible:
            self.config['soar'] = False

    def _sync_dopsd_options_ui(self) -> None:
        if self._dopsd_options_card is None:
            return
        arch_name = self._selected_arch or 'FLUX.2'
        mode = self.train_mode.value if self.train_mode is not None else model_catalog.get_default_train_mode(arch_name)
        version = self.model_selector.version if self.model_selector is not None else None
        visible = model_catalog.supports_dopsd_training(arch_name, mode, version=version)
        self._dopsd_options_card.visible = visible
        if self._dopsd_full_ema_device_container is not None:
            self._dopsd_full_ema_device_container.visible = visible and arch_name == "Z-Image" and mode == "finetune"
        if not visible:
            self.config['dopsd'] = False

    def _get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self._collect_form_state()

    def _apply_config(self, config: Dict[str, Any]):
        """应用配置"""
        self._apply_form_state(config)
        arch_name = self._selected_arch or config.get('arch') or 'FLUX.2'
        self._refresh_train_mode_options(arch_name)
        if 'train_mode' not in config and self.train_mode is not None:
            self.train_mode.set_value(model_catalog.get_default_train_mode(arch_name))
        if 'optimizer_extra_args' not in config:
            self._set_optimizer_args_template(force=True)

    def _optimizer_template_state(self) -> Dict[str, Any]:
        state = self._collect_form_state()
        state.setdefault('network_dim', self.config.get('network_dim', 32))
        state.setdefault('compile', self.config.get('compile', False))
        return state

    def _set_optimizer_args_template(self, force: bool = False):
        if not hasattr(self, 'optimizer_extra_args') or not hasattr(self, 'optimizer_type'):
            return
        if not force and self.optimizer_extra_args.value:
            return
        template_args = get_train_optimizer_template_args(
            self.optimizer_type.value,
            self._optimizer_template_state(),
        )
        self.optimizer_extra_args.set_value('\n'.join(template_args))

    async def _start_train(self):
        """开始训练（使用 accelerate launch）"""
        try:
            project_config = config_manager.load_project_config(self.project_dir)
            job = build_train_job(self._get_config(), self.project_dir, project_config)
        except CommandBuildError as exc:
            ui.notify(str(exc), type='negative')
            return

        await self.exec_panel.run_job(
            script_key=job.script_key,
            args=job.args,
            name=job.name,
            runner_kwargs=job.runner_kwargs,
        )


def render_train_step():
    """渲染训练步骤"""
    step = TrainStep()
    step.render()

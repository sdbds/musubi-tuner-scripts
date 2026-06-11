"""步骤 4: 推理生成 - 完整参数"""
from nicegui import ui
from pathlib import Path
from typing import Optional, Dict, Any
from theme import get_classes, COLORS
from components.path_selector import create_path_selector
from components.model_selector import create_model_selector, get_arch_info
from components.preset_manager import create_preset_manager
from components.advanced_inputs import toggle_switch, editable_slider
from components.execution_panel import ExecutionPanel
from utils.command_builder import CommandBuildError, SCRIPT_DEFAULT_OUTPUT_DIR, build_generate_job
from utils.form_state import FormStateMixin
from utils.i18n import t
from utils import model_catalog


SAMPLE_SOLVERS = ['vanilla', 'unipc', 'dpm++']

ATTN_MODES = ['sageattn', 'torch', 'sdpa', 'xformers', 'flash', 'flash2', 'flash3']

OUTPUT_TYPES = ['video', 'images', 'both', 'latent', 'latent_images']


class GenerateStep(FormStateMixin):
    """推理生成页面 - 完整参数"""

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.project_dir = Path(__file__).resolve().parents[2]
        self.model_selector = None
        self.exec_panel = None
        self.arch_info = None
        self._selected_arch = None
        self._selected_version = None
        self._model_path_container = None
        self._arch_specific_container = None
        self._vae_path_container = None
        self._init_form_state()
        self._dynamic_field_names = {
            'text_encoder_path', 'te1_path', 'te2_path', 't5_path', 'image_encoder_path',
            'text_encoder_vl_path', 'byt5_path', 'unconditional_dit_path',
            'dit_high_noise', 'timestep_boundary',
            'magcache_mag_ratios', 'image_path', 'end_image_path', 'control_image_path',
            'control_image_mask_path', 'control_path', 'image_mask_path', 'end_image_mask_path',
            'ckpt_dir', 'video_path', 'video_sections', 'one_frame_inference', 'dit_in_channels',
            'strength', 'custom_system_prompt', 'latent_paddings', 'rope_scaling_timestep_threshold',
            'automatic_prompt_lang_for_layered', 'num_layers', 'rcm_threshold', 'mask_path',
            'longcat_flow_target', 'ref_images', 'dtype', 'dit_dtype', 'text_encoder_dtype',
            'base_resolution', 'aspect_ratio', 'sampler_preset',
        }

    def render(self):
        """渲染页面"""
        with ui.column().classes(get_classes('page_container') + ' gap-4'):
            # 页面标题
            with ui.row().classes('w-full items-center gap-3 q-mb-sm'):
                ui.icon('image', size='32px')
                with ui.column().classes('gap-0'):
                    ui.label(t('inference_generation')).classes('text-h4 text-weight-bold').style('color: var(--color-text);')
                    ui.label(t('inference_desc')).classes('text-body2').style('color: var(--color-text-secondary);')

            # 预设管理
            create_preset_manager(
                get_config=self._get_config,
                apply_config=self._apply_config,
                scope="generate",
            )

            with ui.tabs().classes('w-full') as tabs:
                tab_basic = ui.tab(t('basic_settings'), icon='settings')
                tab_model = ui.tab(t('model_paths'), icon='folder')
                tab_lora = ui.tab(t('lora'), icon='tune')
                tab_prompt = ui.tab(t('prompts'), icon='edit')
                tab_generation = ui.tab(t('generation_params'), icon='photo_camera')
                tab_inference = ui.tab(t('inference_settings'), icon='speed')
                tab_arch = ui.tab(t('arch_specific'), icon='extension')
                tab_compile = ui.tab(t('compile_perf'), icon='bolt')

            with ui.tab_panels(tabs, value=tab_basic).classes('w-full'):
                with ui.tab_panel(tab_basic):
                    self._render_basic_tab()
                with ui.tab_panel(tab_model):
                    self._render_model_tab()
                with ui.tab_panel(tab_lora):
                    self._render_lora_tab()
                with ui.tab_panel(tab_prompt):
                    self._render_prompt_tab()
                with ui.tab_panel(tab_generation):
                    self._render_generation_tab()
                with ui.tab_panel(tab_inference):
                    self._render_inference_tab()
                with ui.tab_panel(tab_arch):
                    self._render_arch_specific_tab()
                with ui.tab_panel(tab_compile):
                    self._render_compile_tab()

            # 执行面板 (含 Start/Stop 按钮 + 日志)
            self.exec_panel = ExecutionPanel(
                start_label=t('start_generate'),
                height='400px',
                on_start=self._start_generate,
            )

        self._on_arch_change("FLUX.2", get_arch_info("FLUX.2"))

    def _render_basic_tab(self):
        """基础设置"""
        self.model_selector = create_model_selector(
            on_change=self._on_arch_change,
            default_arch="FLUX.2",
            page_key="generate",
        )

    def _render_model_tab(self):
        """模型路径"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('base_model')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.dit_path = create_path_selector(
                label=t('dit_path'),
                selection_type='file_or_dir',
                placeholder=t('select_dit')
            )
            with ui.column().classes('w-full') as self._vae_path_container:
                self.vae_path = create_path_selector(
                    label=t('vae_path'),
                    selection_type='file',
                    placeholder=t('select_vae')
                )
                self.vae_dtype = ui.select(
                    ['', 'float32', 'float16', 'bfloat16'],
                    label=t('vae_dtype'), value=''
                ).classes('w-64').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')

        # 动态文本编码器路径
        self._model_path_container = ui.column().classes('w-full gap-3 q-mt-md')
        with self._model_path_container:
            self._render_dynamic_te_paths("FLUX.2")

    def _render_dynamic_te_paths(self, arch_name: str):
        """根据架构渲染文本编码器路径"""
        if arch_name == "HiDream O1":
            return

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('text_encoder')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')

            if arch_name in ("FLUX.2",):
                self.text_encoder_path = create_path_selector(
                    label=t('text_encoder_mistral'),
                    selection_type='file', placeholder=t('select_te')
                )
                with ui.row().classes('w-full gap-4 q-mt-sm'):
                    self.config.setdefault('fp8_text_encoder', False)
                    toggle_switch(t('fp8_te'), self.config, 'fp8_text_encoder')

            elif arch_name == "HunyuanVideo":
                self.te1_path = create_path_selector(
                    label=t('te1_llava'),
                    selection_type='file', placeholder='llava_llama3_fp16.safetensors'
                )
                self.te2_path = create_path_selector(
                    label=t('te2_clip'),
                    selection_type='file', placeholder='clip_l.safetensors'
                )
                self.config.setdefault('fp8_llm', False)
                toggle_switch(t('fp8_llm'), self.config, 'fp8_llm')

            elif arch_name == "Wan2.1":
                self.te1_path = create_path_selector(
                    label='Text Encoder 1 (LLaVA LLaMA3)',
                    selection_type='file', placeholder='llava_llama3_fp16.safetensors'
                )
                self.te2_path = create_path_selector(
                    label='Text Encoder 2 (CLIP-L)',
                    selection_type='file', placeholder='clip_l.safetensors'
                )
                self.t5_path = create_path_selector(
                    label=t('t5_path'),
                    selection_type='file', placeholder='models_t5_umt5-xxl-enc-bf16.pth'
                )
                self.config.setdefault('fp8_t5', False)
                toggle_switch(t('fp8_t5'), self.config, 'fp8_t5')

            elif arch_name == "FramePack":
                self.te1_path = create_path_selector(
                    label='Text Encoder 1 (LLaVA LLaMA3)',
                    selection_type='file', placeholder='llava_llama3_fp16.safetensors'
                )
                self.te2_path = create_path_selector(
                    label='Text Encoder 2 (CLIP-L)',
                    selection_type='file', placeholder='clip_l.safetensors'
                )
                self.image_encoder_path = create_path_selector(
                    label=t('image_encoder'),
                    selection_type='file', placeholder='sigclip_vision_patch14_384.safetensors'
                )
                self.config.setdefault('fp8_llm', False)
                toggle_switch(t('fp8_llm_short'), self.config, 'fp8_llm')

            elif arch_name == "FLUX Kontext":
                self.te1_path = create_path_selector(
                    label=t('te1_t5'),
                    selection_type='file', placeholder='t5xxl_fp16.safetensors'
                )
                self.te2_path = create_path_selector(
                    label=t('te2_clip'),
                    selection_type='file', placeholder='clip_l.safetensors'
                )
                self.image_encoder_path = create_path_selector(
                    label=t('image_encoder'),
                    selection_type='file', placeholder='sigclip_vision_patch14_384.safetensors'
                )

            elif arch_name in ("Qwen Image",):
                self.te1_path = create_path_selector(
                    label='Text Encoder 1 (T5-XXL)',
                    selection_type='file', placeholder='t5xxl_fp16.safetensors'
                )
                self.te2_path = create_path_selector(
                    label='Text Encoder 2 (CLIP-L)',
                    selection_type='file', placeholder='clip_l.safetensors'
                )
                self.text_encoder_vl_path = create_path_selector(
                    label=t('te_vl_qwen'),
                    selection_type='file', placeholder='qwen_2.5_vl_7b.safetensors'
                )
                self.image_encoder_path = create_path_selector(
                    label=t('image_encoder'),
                    selection_type='file', placeholder='sigclip_vision_patch14_384.safetensors'
                )

            elif arch_name == "Long-CAT":
                self.te1_path = create_path_selector(
                    label='Text Encoder 1 (T5-XXL)',
                    selection_type='file', placeholder='t5xxl_fp16.safetensors'
                )
                self.te2_path = create_path_selector(
                    label='Text Encoder 2 (CLIP-L)',
                    selection_type='file', placeholder='clip_l.safetensors'
                )
                self.text_encoder_vl_path = create_path_selector(
                    label=t('te_vl_qwen'),
                    selection_type='file', placeholder='qwen_2.5_vl_7b.safetensors'
                )
                self.image_encoder_path = create_path_selector(
                    label=t('image_encoder'),
                    selection_type='file', placeholder='sigclip_vision_patch14_384.safetensors'
                )

            elif arch_name == "Z-Image":
                self.text_encoder_path = create_path_selector(
                    label=t('text_encoder_qwen3'),
                    selection_type='file', placeholder='qwen_3_4b.safetensors'
                )
                self.config.setdefault('fp8_llm', False)
                toggle_switch(t('fp8_qwen3'), self.config, 'fp8_llm')

            elif arch_name == "Lens":
                self.text_encoder_path = create_path_selector(
                    label=t('lens_text_encoder', 'Lens Text Encoder'),
                    selection_type='file',
                    file_filter='*.safetensors *.pt *.pth',
                    placeholder='./ckpts/text_encoder/gpt_oss_20b_nvfp4.safetensors'
                )

            elif arch_name == "Ideogram-4":
                self._set_control("unconditional_dit_path", create_path_selector(
                    label='Unconditional DiT',
                    selection_type='file',
                    file_filter='*.safetensors *.pt *.pth',
                    placeholder='./ckpts/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors'
                ))
                self._set_control("text_encoder_path", create_path_selector(
                    label='Qwen3-VL 8B BF16 Text Encoder',
                    selection_type='file',
                    file_filter='*.safetensors *.pt *.pth',
                    placeholder='./ckpts/text_encoder/qwen3vl_8b_bf16.safetensors'
                ))

            elif arch_name == "HV 1.5":
                self.text_encoder_path = create_path_selector(
                    label=t('text_encoder_hv15'),
                    selection_type='file', placeholder='qwen_2.5_vl_7b.safetensors'
                )
                self.byt5_path = create_path_selector(
                    label='BYT5',
                    selection_type='file', placeholder='byt5_model.safetensors'
                )
                self.image_encoder_path = create_path_selector(
                    label=t('image_encoder_i2v'),
                    selection_type='file', placeholder='sigclip_vision_patch14_384.safetensors'
                )

    def _render_lora_tab(self):
        """LoRA 设置"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('lora_weights')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.lora_weight = create_path_selector(
                label=t('lora_weight_file'),
                selection_type='file',
                file_filter='*.safetensors',
                placeholder=t('select_lora')
            )
            self.config.setdefault('lora_multiplier', 1.0)
            with ui.row().classes('w-full gap-4 q-mt-md'):
                editable_slider(t('lora_multiplier'), self.config, 'lora_multiplier', min_val=0, max_val=2, step=0.05, decimals=2)
                self.config.setdefault('lycoris', False)
                toggle_switch(t('use_lycoris'), self.config, 'lycoris')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('lora_filter')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.include_patterns = ui.input(
                'Include Patterns', placeholder=t('apply_all_layers')
            ).classes('w-full')
            self.exclude_patterns = ui.input(
                'Exclude Patterns', placeholder=t('exclude_none')
            ).classes('w-full q-mt-sm')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('high_noise_lora')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.lora_weight_high_noise = create_path_selector(
                label=t('high_noise_lora_weight'),
                selection_type='file',
                file_filter='*.safetensors',
                placeholder=t('high_noise_lora_placeholder')
            )
            self.config.setdefault('lora_multiplier_high_noise', 1.0)
            editable_slider(t('lora_multiplier_high_noise'), self.config, 'lora_multiplier_high_noise', min_val=0, max_val=2, step=0.05, decimals=2)

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('merge_model')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.config.setdefault('save_merged_model', False)
            toggle_switch(t('save_merged_model'), self.config, 'save_merged_model')

    def _render_prompt_tab(self):
        """提示词"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('prompts')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.prompt = ui.textarea(
                t('positive_prompt'),
                placeholder=t('positive_placeholder'),
                value=''
            ).classes('w-full').props('rows=6')
            self.negative_prompt = ui.textarea(
                t('negative_prompt'),
                placeholder=t('negative_placeholder'),
                value=''
            ).classes('w-full q-mt-md').props('rows=3')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('batch_generate')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.from_file = create_path_selector(
                label=t('prompt_file'),
                selection_type='file',
                file_filter='*.txt',
                placeholder=t('batch_placeholder')
            )

    def _render_generation_tab(self):
        """生成参数"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('output_size_frames')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                self.video_size = ui.input(t('video_size'), value='1024 1024', placeholder=t('width_height')).classes('flex-1')
                self.video_size.tooltip(t('video_size_tooltip'))
                self.config.setdefault('video_length', 1)
                editable_slider(t('video_length'), self.config, 'video_length', min_val=1, max_val=257, step=1, decimals=0).tooltip(t('video_length_tooltip'))
                self.config.setdefault('fps', 24)
                editable_slider('FPS', self.config, 'fps', min_val=1, max_val=60, step=1, decimals=0, label_default='FPS')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('inference_params')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                self.config.setdefault('infer_steps', 25)
                editable_slider(t('infer_steps'), self.config, 'infer_steps', min_val=1, max_val=100, step=1, decimals=0)
                self.config.setdefault('guidance_scale', 5.0)
                editable_slider('Guidance Scale', self.config, 'guidance_scale', min_val=0, max_val=20, step=0.5, decimals=1, label_default='Guidance Scale')
                self.config.setdefault('embedded_cfg_scale', 7.0)
                editable_slider('Embedded CFG Scale', self.config, 'embedded_cfg_scale', min_val=0, max_val=20, step=0.5, decimals=1, label_default='Embedded CFG Scale')
            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.flow_shift = ui.input('Flow Shift', value='3.0', placeholder=t('flow_shift_placeholder')).classes('flex-1')
                self.config.setdefault('seed', 1026)
                editable_slider(t('seed'), self.config, 'seed', min_val=0, max_val=9999999999, step=1, decimals=0)
                self.sample_solver = ui.select(
                    SAMPLE_SOLVERS, label=t('sample_solver'), value='vanilla'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('output_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.save_path = create_path_selector(
                label=t('output_dir'),
                default_path=SCRIPT_DEFAULT_OUTPUT_DIR,
                selection_type='dir',
                placeholder=t('save_path_placeholder')
            )
            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.output_type = ui.select(
                    OUTPUT_TYPES, label=t('output_format'), value='images'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.config.setdefault('no_metadata', False)
                toggle_switch(t('no_metadata'), self.config, 'no_metadata')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('latent_decode')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.latent_path = create_path_selector(
                label=t('latent_decode_skip'),
                selection_type='file',
                placeholder=t('latent_placeholder')
            )

    def _render_inference_tab(self):
        """推理设置"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('precision_attention')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                self.attn_mode = ui.select(
                    ATTN_MODES, label=t('attn_mode'), value='sageattn'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.device = ui.select(
                    ['', 'cuda', 'cpu'], label=t('device'), value=''
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                self.config.setdefault('fp8', True)
                toggle_switch(t('fp8_dit'), self.config, 'fp8')
                self.config.setdefault('fp8_scaled', True)
                toggle_switch('FP8 Scaled', self.config, 'fp8_scaled')
                self.config.setdefault('fp8_fast', True)
                toggle_switch('FP8 Fast (RTX 4XXX+)', self.config, 'fp8_fast')
                self.config.setdefault('split_attn', True)
                toggle_switch('Split Attention', self.config, 'split_attn')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('memory_optimize')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                self.config.setdefault('blocks_to_swap', 8)
                editable_slider(t('blocks_to_swap'), self.config, 'blocks_to_swap', min_val=0, max_val=40, step=1, decimals=0).tooltip(t('blocks_to_swap_tooltip'))
                self.config.setdefault('use_pinned_memory', True)
                toggle_switch(t('pinned_memory'), self.config, 'use_pinned_memory')
                self.config.setdefault('img_in_txt_in_offloading', True)
                toggle_switch('img_in/txt_in Offloading', self.config, 'img_in_txt_in_offloading')
                self.config.setdefault('disable_numpy_memmap', False)
                toggle_switch(t('disable_numpy_memmap'), self.config, 'disable_numpy_memmap')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('cfg_optimize')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                self.cfg_skip_mode = ui.select(
                    ['', 'late', 'early'], label='CFG Skip Mode', value=''
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.config.setdefault('cfg_apply_ratio', 0.3)
                editable_slider('CFG Apply Ratio', self.config, 'cfg_apply_ratio', min_val=0, max_val=1, step=0.05, decimals=2, label_default='CFG Apply Ratio')
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                self.config.setdefault('cpu_noise', False)
                toggle_switch(t('cpu_noise'), self.config, 'cpu_noise')
                self.config.setdefault('vae_cache_cpu', False)
                toggle_switch('VAE Cache CPU', self.config, 'vae_cache_cpu')

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('slg_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4'):
                self.slg_layers = ui.input('SLG Layers', placeholder=t('slg_layers_placeholder')).classes('flex-1')
                self.config.setdefault('slg_scale', 3.0)
                editable_slider('SLG Scale', self.config, 'slg_scale', min_val=0, max_val=10, step=0.1, decimals=1, label_default='SLG Scale')
            with ui.row().classes('w-full gap-4 q-mt-md'):
                self.config.setdefault('slg_start', 0.0)
                editable_slider('SLG Start', self.config, 'slg_start', min_val=0, max_val=1, step=0.05, decimals=2, label_default='SLG Start')
                self.config.setdefault('slg_end', 0.3)
                editable_slider('SLG End', self.config, 'slg_end', min_val=0, max_val=1, step=0.05, decimals=2, label_default='SLG End')
                self.slg_mode = ui.select(
                    ['uncond', 'original'], label='SLG Mode', value='uncond'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')

    def _render_arch_specific_tab(self):
        """架构专属参数"""
        self._arch_specific_container = ui.column().classes('w-full gap-3')
        with self._arch_specific_container:
            self._render_dynamic_arch_specific("FLUX.2")

    def _render_dynamic_arch_specific(self, arch_name: str):
        """根据架构渲染专属参数"""
        if arch_name == "HiDream O1":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='HiDream O1')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4'):
                    self.dtype = ui.select(
                        ['bfloat16', 'float16', 'float32'],
                        label='Model Dtype',
                        value='bfloat16',
                    ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                    self.config.setdefault('noise_scale_start', 8.0)
                    editable_slider('Noise Scale Start', self.config, 'noise_scale_start', min_val=0, max_val=20, step=0.1, decimals=1, label_default='Noise Scale Start')
                    self.config.setdefault('noise_scale_end', 8.0)
                    editable_slider('Noise Scale End', self.config, 'noise_scale_end', min_val=0, max_val=20, step=0.1, decimals=1, label_default='Noise Scale End')
                    self.config.setdefault('noise_clip_std', 0.0)
                    editable_slider('Noise Clip Std', self.config, 'noise_clip_std', min_val=0, max_val=10, step=0.1, decimals=1, label_default='Noise Clip Std')
                with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                    self.config.setdefault('keep_original_aspect', False)
                    toggle_switch('Keep Original Aspect', self.config, 'keep_original_aspect')
                with ui.row().classes('w-full gap-4 q-mt-md'):
                    self.editing_scheduler = ui.select(
                        ['flow_match', 'flash'],
                        label='Editing Scheduler',
                        value=self.config.get('editing_scheduler', 'flow_match'),
                    ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.ref_images = ui.textarea(
                    'Reference Images',
                    placeholder='One image path per line',
                ).classes('w-full q-mt-md').props('autogrow outlined')
                self.layout_bboxes = ui.textarea(
                    'Layout BBoxes',
                    placeholder='JSON string or JSON file path',
                ).classes('w-full q-mt-md').props('autogrow outlined')

        elif arch_name == "HunyuanVideo":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='HunyuanVideo')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4'):
                    self.config.setdefault('vae_chunk_size', 32)
                    editable_slider('VAE Chunk Size', self.config, 'vae_chunk_size', min_val=1, max_val=256, step=1, decimals=0, label_default='VAE Chunk Size')
                    self.config.setdefault('vae_spatial_tile_sample_min_size', 128)
                    editable_slider('VAE Spatial Tile Min', self.config, 'vae_spatial_tile_sample_min_size', min_val=64, max_val=512, step=64, decimals=0, label_default='VAE Spatial Tile Min')
                    self.dit_in_channels = ui.input(
                        'DiT In Channels',
                        placeholder='auto / 16 / 32',
                    ).classes('flex-1')
                with ui.row().classes('w-full gap-4 q-mt-md'):
                    self.config.setdefault('strength', 0.8)
                    editable_slider('Strength', self.config, 'strength', min_val=0, max_val=1, step=0.05, decimals=2, label_default='Strength')
                with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                    self.config.setdefault('split_uncond', False)
                    toggle_switch('Split Uncond', self.config, 'split_uncond')
                    self.config.setdefault('exclude_single_blocks', False)
                    toggle_switch('Exclude Single Blocks', self.config, 'exclude_single_blocks')

            with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
                ui.label(t('image_input_i2v')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                self.video_path = create_path_selector(
                    label='Video Path',
                    selection_type='file',
                    placeholder=t('optional'),
                )
                self.image_path = create_path_selector(
                    label=t('image_prompt_file'),
                    selection_type='file',
                    placeholder=t('optional'),
                )

        elif arch_name == "Wan2.1":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='Wan2.1')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                self.ckpt_dir = create_path_selector(
                    label='Official Ckpt Dir',
                    selection_type='dir',
                    placeholder=t('optional'),
                )
                self.dit_high_noise = create_path_selector(
                    label=t('dit_high_noise'),
                    selection_type='file', placeholder=t('optional')
                )
                with ui.row().classes('w-full gap-4 q-mt-md'):
                    self.config.setdefault('trim_tail_frames', 0)
                    editable_slider(t('trim_tail_frames'), self.config, 'trim_tail_frames', min_val=0, max_val=10, step=1, decimals=0)
                    self.config.setdefault('guidance_scale_high_noise', 5.0)
                    editable_slider('High Noise Guidance', self.config, 'guidance_scale_high_noise', min_val=0, max_val=20, step=0.5, decimals=1, label_default='High Noise Guidance')
                    self.one_frame_inference = ui.input(
                        t('one_frame_inference'),
                        placeholder='target_index=9,control_indices=0',
                    ).classes('flex-1')
                with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                    self.config.setdefault('offload_inactive_dit', False)
                    toggle_switch(t('offload_inactive_dit'), self.config, 'offload_inactive_dit')
                    self.config.setdefault('lazy_loading', False)
                    toggle_switch('Lazy Loading', self.config, 'lazy_loading')
                    self.config.setdefault('force_v2_1_time_embedding', False)
                    toggle_switch('Force v2.1 Time Embedding', self.config, 'force_v2_1_time_embedding')
                    self.timestep_boundary = ui.input('Timestep Boundary', placeholder=t('timestep_boundary_placeholder')).classes('flex-1')

            with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
                ui.label(t('megacache_wan')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    self.config.setdefault('enable_megacache', False)
                    toggle_switch(t('enable_megacache'), self.config, 'enable_megacache')
                    self.config.setdefault('magcache_calibration', False)
                    toggle_switch(t('megacache_calibration'), self.config, 'magcache_calibration')
                with ui.row().classes('w-full gap-4 q-mt-md'):
                    self.magcache_mag_ratios = ui.input('Mag Ratios', placeholder=t('mag_ratios_placeholder')).classes('flex-1')
                    self.config.setdefault('magcache_retention_ratio', 0.2)
                    editable_slider('Retention Ratio', self.config, 'magcache_retention_ratio', min_val=0, max_val=1, step=0.05, decimals=2, label_default='Retention Ratio')
                    self.config.setdefault('magcache_threshold', 0.24)
                    editable_slider('Threshold', self.config, 'magcache_threshold', min_val=0, max_val=1, step=0.01, decimals=2, label_default='Threshold')
                    self.config.setdefault('magcache_k', 6)
                    editable_slider('K', self.config, 'magcache_k', min_val=1, max_val=20, step=1, decimals=0, label_default='K')

            with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
                ui.label(t('image_input_i2v')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                self.video_path = create_path_selector(
                    label='Video Path',
                    selection_type='file',
                    placeholder=t('optional'),
                )
                self.image_path = create_path_selector(
                    label=t('image_prompt_file'),
                    selection_type='file', placeholder=t('i2v_required')
                )
                self.end_image_path = create_path_selector(
                    label=t('end_image_path'),
                    selection_type='file', placeholder=t('optional')
                )
                self.control_image_path = create_path_selector(
                    label=t('control_image_path'),
                    selection_type='file', placeholder=t('optional')
                )
                self.control_image_mask_path = create_path_selector(
                    label='Control Image Mask',
                    selection_type='file',
                    placeholder=t('optional'),
                )
                self.control_path = create_path_selector(
                    label=t('control_video_path'),
                    selection_type='file', placeholder=t('control_video_placeholder')
                )

        elif arch_name == "FramePack":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='FramePack')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4'):
                    self.config.setdefault('vae_chunk_size', 32)
                    editable_slider('VAE Chunk Size', self.config, 'vae_chunk_size', min_val=1, max_val=256, step=1, decimals=0, label_default='VAE Chunk Size')
                    self.config.setdefault('vae_spatial_tile_sample_min_size', 128)
                    editable_slider('Spatial Tile Min', self.config, 'vae_spatial_tile_sample_min_size', min_val=64, max_val=512, step=64, decimals=0, label_default='Spatial Tile Min')
                    self.config.setdefault('latent_window_size', 9)
                    editable_slider('Latent Window Size', self.config, 'latent_window_size', min_val=1, max_val=32, step=1, decimals=0, label_default='Latent Window Size')
                with ui.row().classes('w-full gap-4 q-mt-md'):
                    self.config.setdefault('video_seconds', 5)
                    editable_slider(t('video_seconds'), self.config, 'video_seconds', min_val=1, max_val=60, step=1, decimals=0)
                    self.video_sections = ui.input(t('video_sections'), value='1').classes('flex-1')
                    self.config.setdefault('guidance_rescale', 0.0)
                    editable_slider('Guidance Rescale', self.config, 'guidance_rescale', min_val=0, max_val=1, step=0.05, decimals=2, label_default='Guidance Rescale')
                with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                    self.config.setdefault('f1_mode', False)
                    toggle_switch(t('f1_mode'), self.config, 'f1_mode')
                    self.config.setdefault('bulk_decode', False)
                    toggle_switch(t('bulk_decode'), self.config, 'bulk_decode')
                    self.config.setdefault('one_frame_auto_resize', False)
                    toggle_switch('One Frame Auto Resize', self.config, 'one_frame_auto_resize')
                    self.config.setdefault('vae_tiling', False)
                    toggle_switch('VAE Tiling', self.config, 'vae_tiling')
                self.one_frame_inference = ui.input(
                    t('one_frame_inference'), value='target_index=9,control_indices=0'
                ).classes('w-full q-mt-md')
                self.custom_system_prompt = ui.textarea(
                    'Custom System Prompt',
                    placeholder=t('optional'),
                ).classes('w-full q-mt-md').props('autogrow outlined')
                with ui.row().classes('w-full gap-4 q-mt-md'):
                    self.latent_paddings = ui.input(
                        'Latent Paddings',
                        placeholder='comma separated values',
                    ).classes('flex-1')
                    self.config.setdefault('rope_scaling_factor', 0.5)
                    editable_slider('RoPE Scaling Factor', self.config, 'rope_scaling_factor', min_val=0, max_val=2, step=0.05, decimals=2, label_default='RoPE Scaling Factor')
                    self.rope_scaling_timestep_threshold = ui.input(
                        'RoPE Timestep Threshold',
                        placeholder='e.g. 800',
                    ).classes('flex-1')

            with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
                ui.label(t('image_input')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                self.image_path = create_path_selector(
                    label=t('image_prompt_file'), selection_type='file', placeholder=t('i2v_input_image')
                )
                self.end_image_path = create_path_selector(
                    label=t('end_image_path'), selection_type='file', placeholder=t('optional')
                )
                self.image_mask_path = create_path_selector(
                    label=t('image_mask_path'), selection_type='file', placeholder=t('optional')
                )
                self.end_image_mask_path = create_path_selector(
                    label=t('end_image_mask_path'), selection_type='file', placeholder=t('optional')
                )
                self.control_image_path = create_path_selector(
                    label=t('control_image'), selection_type='file', placeholder=t('optional')
                )
                self.control_image_mask_path = create_path_selector(
                    label='Control Image Mask',
                    selection_type='file',
                    placeholder=t('optional'),
                )

        elif arch_name == "Qwen Image":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='Qwen Image')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    self.config.setdefault('fp8_vl', False)
                    toggle_switch('FP8 VL', self.config, 'fp8_vl')
                    self.config.setdefault('text_encoder_cpu', False)
                    toggle_switch(t('text_encoder_cpu'), self.config, 'text_encoder_cpu')
                    self.config.setdefault('edit_mode', False)
                    toggle_switch(t('edit_mode'), self.config, 'edit_mode')
                    self.config.setdefault('edit_plus', False)
                    toggle_switch(t('edit_plus_mode'), self.config, 'edit_plus')
                    self.config.setdefault('vae_enable_tiling', False)
                    toggle_switch('VAE Enable Tiling', self.config, 'vae_enable_tiling')
                    self.config.setdefault('append_original_name', False)
                    toggle_switch('Append Original Name', self.config, 'append_original_name')
                with ui.row().classes('w-full gap-4 q-mt-md'):
                    self.num_layers = ui.input(
                        'DiT Num Layers',
                        placeholder='default: 60',
                    ).classes('flex-1')
                    self.config.setdefault('output_layers', 4)
                    editable_slider('Output Layers', self.config, 'output_layers', min_val=1, max_val=16, step=1, decimals=0, label_default='Output Layers')
                    self.automatic_prompt_lang_for_layered = ui.select(
                        {'': t('optional'), 'en': 'English', 'cn': '中文'},
                        label='Layered Auto Prompt Lang',
                        value='',
                    ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                    self.config.setdefault('resize_control_to_image_size', False)
                    toggle_switch('Resize Control to Image Size', self.config, 'resize_control_to_image_size')
                    self.config.setdefault('resize_control_to_official_size', False)
                    toggle_switch('Resize Control to Official Size', self.config, 'resize_control_to_official_size')
                    self.config.setdefault('bell', False)
                    toggle_switch('Bell', self.config, 'bell')

            with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
                ui.label(t('rcm_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4'):
                    self.rcm_threshold = ui.input('RCM Threshold', value='0.2').classes('flex-1')
                    self.config.setdefault('rcm_kernel_size', 3)
                    editable_slider('Kernel Size', self.config, 'rcm_kernel_size', min_val=1, max_val=10, step=1, decimals=0, label_default='Kernel Size')
                    self.config.setdefault('rcm_dilate_size', 1)
                    editable_slider('Dilate Size', self.config, 'rcm_dilate_size', min_val=0, max_val=10, step=1, decimals=0, label_default='Dilate Size')
                with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                    self.config.setdefault('rcm_relative_threshold', True)
                    toggle_switch('Relative Threshold', self.config, 'rcm_relative_threshold')
                    self.config.setdefault('rcm_debug_save', True)
                    toggle_switch('Debug Save', self.config, 'rcm_debug_save')

            with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
                ui.label(t('image_input')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                self.control_image_path = create_path_selector(
                    label=t('control_image_path'),
                    selection_type='file',
                    placeholder=t('edit_i2v_input'),
                )
                self.image_path = create_path_selector(
                    label=t('image_prompt_file'), selection_type='file', placeholder=t('edit_i2v_input')
                )
                self.mask_path = create_path_selector(
                    label=t('mask_path'), selection_type='file', placeholder=t('optional')
                )

        elif arch_name == "Long-CAT":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='Long-CAT')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    self.config.setdefault('fp8_vl', False)
                    toggle_switch('FP8 VL', self.config, 'fp8_vl')
                    self.config.setdefault('text_encoder_cpu', False)
                    toggle_switch(t('text_encoder_cpu'), self.config, 'text_encoder_cpu')
                    self.config.setdefault('longcat_i2v', False)
                    toggle_switch('LongCat I2V', self.config, 'longcat_i2v')
                    self.config.setdefault('disable_numpy_memmap', False)
                    toggle_switch(t('disable_numpy_memmap'), self.config, 'disable_numpy_memmap')
                with ui.row().classes('w-full gap-4 q-mt-md'):
                    self.longcat_flow_target = ui.select(
                        ['x1_minus_x0', 'velocity'],
                        label='Flow Target', value='x1_minus_x0'
                    ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                    self.config.setdefault('latent_window_size', 9)
                    editable_slider('Latent Window Size', self.config, 'latent_window_size', min_val=1, max_val=32, step=1, decimals=0, label_default='Latent Window Size')

            with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
                ui.label(t('image_input')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                self.image_path = create_path_selector(
                    label=t('image_prompt_file'), selection_type='file', placeholder=t('optional')
                )
                self.end_image_path = create_path_selector(
                    label=t('end_image_path'), selection_type='file', placeholder=t('optional')
                )
                self.control_image_path = create_path_selector(
                    label=t('control_image'), selection_type='file', placeholder=t('optional')
                )
                self.control_path = create_path_selector(
                    label=t('control_video_path'), selection_type='file', placeholder=t('control_video_placeholder')
                )

        elif arch_name == "Z-Image":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='Z-Image')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    self.config.setdefault('text_encoder_cpu', False)
                    toggle_switch(t('text_encoder_cpu'), self.config, 'text_encoder_cpu')
                    self.config.setdefault('use_32bit_attention', False)
                    toggle_switch('32-bit Attention', self.config, 'use_32bit_attention')
                    self.config.setdefault('bell', False)
                    toggle_switch('Bell', self.config, 'bell')

        elif arch_name == "Lens":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='Lens')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    self.dit_dtype = ui.select(
                        ['bfloat16', 'float16', 'float32'],
                        label='DiT Dtype',
                        value='bfloat16',
                    ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                    self.text_encoder_dtype = ui.select(
                        ['bfloat16', 'float16', 'float32'],
                        label=t('text_encoder_dtype', 'Text Encoder Dtype'),
                        value='bfloat16',
                    ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                    self.config.setdefault('disable_numpy_memmap', False)
                    toggle_switch(t('disable_numpy_memmap', 'Disable Numpy Memmap'), self.config, 'disable_numpy_memmap')

        elif arch_name == "Ideogram-4":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='Ideogram-4')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    self._set_control("sampler_preset", ui.select(
                        ['V4_DEFAULT_20', 'V4_DEFAULT_16', 'V4_DEFAULT_12'],
                        label='Sampler Preset',
                        value=self.config.get('sampler_preset', 'V4_DEFAULT_20'),
                    ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"'))
                    self._set_control("dtype", ui.select(
                        ['bfloat16', 'float16', 'float32'],
                        label='Dtype',
                        value=self.config.get('dtype', 'bfloat16'),
                    ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"'))
                    self.config.setdefault('disable_numpy_memmap', False)
                    toggle_switch(t('disable_numpy_memmap', 'Disable Numpy Memmap'), self.config, 'disable_numpy_memmap')
                    self.config.setdefault('warn_on_caption_issues', False)
                    toggle_switch('Warn On Caption Issues', self.config, 'warn_on_caption_issues')

        elif arch_name == "HV 1.5":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='HunyuanVideo 1.5')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                with ui.row().classes('w-full gap-4 flex-wrap'):
                    self.config.setdefault('vae_sample_size', 128)
                    editable_slider('VAE Sample Size', self.config, 'vae_sample_size', min_val=64, max_val=512, step=64, decimals=0, label_default='VAE Sample Size')
                    self.config.setdefault('text_encoder_cpu', False)
                    toggle_switch(t('text_encoder_cpu'), self.config, 'text_encoder_cpu')
                    self.config.setdefault('vae_enable_patch_conv', False)
                    toggle_switch('Patch-based Conv', self.config, 'vae_enable_patch_conv')
                self.image_path = create_path_selector(
                    label=t('input_image_i2v'), selection_type='file', placeholder=t('input_image_i2v_placeholder')
                )

        elif arch_name == "FLUX.2":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='FLUX.2')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                self.control_image_path = create_path_selector(
                    label=t('control_image_edit'), selection_type='file', placeholder=t('optional')
                )
                self.config.setdefault('no_resize_control', False)
                toggle_switch(t('no_resize_control'), self.config, 'no_resize_control')

        elif arch_name == "FLUX Kontext":
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('arch_specific_params').format(arch='FLUX Kontext')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
                self.image_path = create_path_selector(
                    label=t('input_image'), selection_type='file', placeholder=t('optional')
                )
                self.config.setdefault('no_resize_control', False)
                toggle_switch(t('no_resize_control'), self.config, 'no_resize_control')

        else:
            with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
                ui.label(t('select_arch_first')).classes('text-body1').style('color: var(--color-text-muted);')

    def _render_compile_tab(self):
        """编译与性能"""
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label(t('compile_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            with ui.row().classes('w-full gap-4 flex-wrap'):
                self.config.setdefault('compile', False)
                toggle_switch(t('compile'), self.config, 'compile')
                self.compile_backend = ui.select(
                    ['inductor', 'aot_eager', 'cudagraphs'],
                    label=t('compile_backend'), value='inductor'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.compile_mode = ui.select(
                    ['default', 'reduce-overhead', 'max-autotune-no-cudagraphs'],
                    label=t('compile_mode'), value='default'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
            with ui.row().classes('w-full gap-4 q-mt-md flex-wrap'):
                self.config.setdefault('compile_fullgraph', False)
                toggle_switch('Fullgraph', self.config, 'compile_fullgraph')
                self.compile_dynamic = ui.select(
                    ['auto', 'true', 'false'],
                    label='Dynamic', value='auto'
                ).classes('flex-1').props('use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
                self.config.setdefault('compile_cache_size_limit', 32)
                editable_slider('Cache Size Limit', self.config, 'compile_cache_size_limit', min_val=1, max_val=128, step=1, decimals=0, label_default='Cache Size Limit')
            self.compile_args = ui.input(
                'Legacy Compile Args',
                placeholder='BACKEND MODE DYNAMIC FULLGRAPH, e.g. inductor default auto false',
            ).classes('w-full q-mt-md')
            self.compile_args.tooltip(t(
                'legacy_compile_args_tooltip',
                'Wan2.1/HunyuanVideo legacy --compile_args; leave empty when using the individual compile controls.',
            ))

        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label(t('tf32_settings')).classes('text-h6 text-weight-bold q-mb-md').style('color: var(--color-text);')
            self.config.setdefault('enable_tf32', True)
            toggle_switch(t('enable_tf32'), self.config, 'enable_tf32')

    def _on_arch_change(self, arch_name: str, arch_info: dict):
        """架构改变时的处理"""
        version = self._current_model_version(arch_name)
        if arch_name == self._selected_arch and version == self._selected_version:
            return

        self.arch_info = arch_info
        self._selected_arch = arch_name
        self._selected_version = version
        for name in self._dynamic_field_names:
            if hasattr(self, name):
                delattr(self, name)
        self._sync_vae_path_ui(arch_name)

        if self._model_path_container:
            self._model_path_container.clear()
            with self._model_path_container:
                self._render_dynamic_te_paths(arch_name)

        if self._arch_specific_container:
            self._arch_specific_container.clear()
            with self._arch_specific_container:
                self._render_dynamic_arch_specific(arch_name)

        self._apply_model_path_defaults(arch_name, version)
        self._apply_lens_generate_defaults(arch_name)

    def _sync_vae_path_ui(self, arch_name: str) -> None:
        if self._vae_path_container is None:
            return
        visible = arch_name != "HiDream O1"
        self._vae_path_container.visible = visible
        if not visible:
            if hasattr(self, "vae_path"):
                self._write_control_value(self.vae_path, "")
            if hasattr(self, "vae_dtype"):
                self._write_control_value(self.vae_dtype, "")

    def _current_model_version(self, arch_name: str) -> str | None:
        if self.model_selector is not None:
            return self.model_selector.version
        return model_catalog.get_default_version(arch_name, "generate")

    def _apply_model_path_defaults(self, arch_name: str, version: str | None = None) -> None:
        for key, value in model_catalog.get_path_defaults(arch_name, "generate", version).items():
            control = getattr(self, key, None)
            if control is not None:
                self._write_control_value(control, value)

    def _apply_lens_generate_defaults(self, arch_name: str) -> None:
        if arch_name != "Lens":
            return
        if hasattr(self, "vae_dtype"):
            self._write_control_value(self.vae_dtype, "float32")

    def _get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self._collect_form_state()

    def _apply_config(self, config: Dict[str, Any]):
        """应用配置"""
        self._apply_form_state(config)

    async def _start_generate(self):
        """开始生成"""
        try:
            job = build_generate_job(self._get_config(), self.project_dir)
        except CommandBuildError as exc:
            ui.notify(str(exc), type='negative')
            return

        await self.exec_panel.run_job(
            script_key=job.script_key,
            args=job.args,
            name=job.name,
            runner_kwargs=job.runner_kwargs,
        )


def render_generate_step():
    """渲染生成步骤"""
    step = GenerateStep()
    step.render()

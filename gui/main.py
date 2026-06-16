"""
Musubi Tuner GUI
基于 NiceGUI 的现代化图形化界面

使用方法:
    cd gui
    python main.py

或者:
    python gui/main.py
"""

import asyncio
import json
import sys
from importlib import import_module
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nicegui import app, ui

from components.side_tools import create_side_tools
from theme import COLORS, apply_theme, get_classes
from utils import model_catalog
from utils.i18n import get_i18n, get_translation_pairs, set_language, t
from utils.port_utils import resolve_gui_host, resolve_gui_native, resolve_gui_port, resolve_gui_show


APP_TITLE = "Musubi Tuner Scripts"
APP_VERSION = "1.0.0"

THEME_SCRIPT = """
<script>
(function() {
    var normalizeButtonsQueued = false;

    function setThemeClasses(isDark) {
        document.documentElement.classList.toggle('dark-mode', isDark);
        document.documentElement.classList.toggle('light-mode', !isDark);
        if (document.body) {
            document.body.classList.toggle('dark-mode', isDark);
            document.body.classList.toggle('light-mode', !isDark);
        }
    }

    function isDarkMode() {
        return document.documentElement.classList.contains('dark-mode')
            || (document.body && document.body.classList.contains('dark-mode'));
    }

    function resolveThemePrimary() {
        var styles = getComputedStyle(document.body);
        var primary = styles.getPropertyValue('--ql-accent').trim();
        return primary || '#80618a';
    }

    function normalizeCustomButtons() {
        var selectors = [
            '.q-btn.ql-btn-primary',
            '.q-btn.modern-btn-primary',
            '.q-btn.modern-btn-success',
            '.q-btn.gold-btn',
            '.q-btn.green-btn',
            '.q-btn.ql-btn-ghost',
            '.q-btn.modern-btn-ghost',
            '.q-btn.ql-btn-secondary',
            '.q-btn.modern-btn-secondary',
            '.q-btn.ql-btn-danger',
            '.q-btn.modern-btn-danger',
            '.q-btn.red-btn'
        ].join(',');
        document.querySelectorAll(selectors).forEach(function(btn) {
            btn.classList.remove(
                'bg-primary', 'bg-secondary', 'bg-positive', 'bg-negative',
                'bg-info', 'bg-warning', 'text-white'
            );
        });
    }

    function queueNormalizeCustomButtons() {
        if (normalizeButtonsQueued) return;
        normalizeButtonsQueued = true;
        requestAnimationFrame(function() {
            normalizeButtonsQueued = false;
            normalizeCustomButtons();
        });
    }

    function syncThemeToggleIcons() {
        var isDark = isDarkMode();
        document.querySelectorAll('.theme-toggle-btn .q-icon').forEach(function(icon) {
            icon.textContent = isDark ? 'dark_mode' : 'light_mode';
        });
    }

    function syncQuasarDark(isDark) {
        if (window.Quasar && Quasar.Dark) Quasar.Dark.set(isDark);
        var accent = resolveThemePrimary();
        document.documentElement.style.setProperty('--q-primary', accent);
        document.documentElement.style.setProperty('--q-color-primary', accent);
    }

    window.syncThemeToggleIcons = syncThemeToggleIcons;
    window.normalizeCustomButtons = normalizeCustomButtons;
    window.queueNormalizeCustomButtons = queueNormalizeCustomButtons;

    window.toggleDarkMode = function() {
        var isDark = !isDarkMode();
        setThemeClasses(isDark);
        localStorage.setItem('dark_mode', isDark);
        syncQuasarDark(isDark);
        syncThemeToggleIcons();
        queueNormalizeCustomButtons();
        return isDark;
    };

    var saved = localStorage.getItem('dark_mode');
    var isDark = saved === null ? true : saved === 'true';
    setThemeClasses(isDark);
    syncQuasarDark(isDark);
    syncThemeToggleIcons();
    queueNormalizeCustomButtons();

    requestAnimationFrame(function() {
        syncQuasarDark(isDark);
        queueNormalizeCustomButtons();
    });
    requestAnimationFrame(syncThemeToggleIcons);
    setTimeout(function() {
        syncQuasarDark(isDark);
        syncThemeToggleIcons();
        queueNormalizeCustomButtons();
    }, 300);

    new MutationObserver(function() {
        queueNormalizeCustomButtons();
    }).observe(document.body, { childList: true, subtree: true });
})();
</script>
"""


def _load_wizard_attr(module_name: str, attr_name: str):
    """按需加载页面模块，避免首页启动时引入全部训练页面。"""
    module = import_module(module_name)
    return getattr(module, attr_name)


def _sync_visible_language_text(old_lang: str, new_lang: str) -> None:
    """Update visible static text after a language change without rebuilding the page."""
    pairs = get_translation_pairs(old_lang, new_lang)
    if not pairs:
        return

    pairs_json = json.dumps(pairs, ensure_ascii=False)
    ui.run_javascript(f"""
        (function(pairs) {{
            var skipTags = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEXTAREA']);

            function translated(value) {{
                if (!value) return value;
                if (Object.prototype.hasOwnProperty.call(pairs, value)) return pairs[value];
                for (var oldText in pairs) {{
                    if (!Object.prototype.hasOwnProperty.call(pairs, oldText)) continue;
                    var suffix = value.slice(oldText.length);
                    if (value.startsWith(oldText) && /^\\s+\\d+$/.test(suffix)) {{
                        return pairs[oldText] + suffix;
                    }}
                }}
                return value;
            }}

            function replaceTextNode(node) {{
                var parent = node.parentElement;
                if (!parent || skipTags.has(parent.tagName)) return;
                var raw = node.nodeValue || '';
                var trimmed = raw.trim();
                if (!trimmed) return;
                var next = translated(trimmed);
                if (next !== trimmed) node.nodeValue = raw.replace(trimmed, next);
            }}

            var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            var nodes = [];
            var node;
            while ((node = walker.nextNode())) nodes.push(node);
            nodes.forEach(replaceTextNode);

            document.querySelectorAll('[placeholder], [title], [aria-label]').forEach(function(el) {{
                ['placeholder', 'title', 'aria-label'].forEach(function(attr) {{
                    var value = el.getAttribute(attr);
                    var next = translated(value);
                    if (next !== value) el.setAttribute(attr, next);
                }});
            }});

            if (window.queueNormalizeCustomButtons) window.queueNormalizeCustomButtons();
        }})({pairs_json});
    """)


def create_header() -> None:
    """创建对齐参考项目的头部导航。"""
    with ui.header().classes(get_classes("header")):
        with ui.row().classes("w-full items-center justify-between").style("padding: 4px 0;"):
            with ui.row().classes("items-center gap-3"):
                ui.label("🐉").style("font-size: 28px; line-height: 1;")
                with ui.column().classes("gap-0"):
                    ui.label(t("app_title", APP_TITLE)).classes("text-subtitle1 header-title")
                    ui.label(f"v{APP_VERSION}").classes("header-version")

            with ui.row().classes("gap-1"):
                nav_items = [
                    (t("nav_home"), "/", "home"),
                    (t("nav_dataset", "Dataset"), "/tagging", "label"),
                    (t("nav_cache"), "/cache", "save"),
                    (t("nav_train"), "/train", "model_training"),
                    (t("nav_generate"), "/generate", "image"),
                ]

                for label_text, path, icon in nav_items:
                    btn = ui.button(label_text, on_click=lambda p=path: ui.navigate.to(p), icon=icon, color=None)
                    btn.props("flat no-caps")
                    btn.classes(get_classes("nav_btn"))

            with ui.row().classes("items-center gap-2"):
                settings_dlg = _load_wizard_attr("wizard.step7_settings", "create_settings_dialog")()
                settings_btn = ui.button(icon="settings", on_click=settings_dlg.open).props("flat round dense")
                settings_btn.style("color: var(--ql-text-secondary);")
                settings_btn.tooltip(t("nav_settings", "Settings"))

                theme_btn = ui.button(icon="light_mode").props("flat round dense").classes("theme-toggle-btn")
                theme_btn.on_click(lambda: ui.run_javascript("window.toggleDarkMode()"))

                with ui.row().classes("items-center gap-1"):
                    ui.icon("language", size="18px")
                    lang_select = (
                        ui.select(
                            {
                                "zh": "🇨🇳 中文",
                                "en": "🇺🇸 English",
                                "ja": "🇯🇵 日本語",
                                "ko": "🇰🇷 한국어",
                            },
                            label="",
                            value=get_i18n().lang,
                        )
                        .props('dense outlined use-input fill-input hide-selected input-debounce="0" dropdown-icon="expand_more"')
                        .classes("lang-selector")
                    )

                    def on_lang_change(e) -> None:
                        lang = e.value
                        if lang and lang in ["zh", "en", "ja", "ko"]:
                            if lang == get_i18n().lang:
                                return
                            old_lang = get_i18n().lang
                            set_language(lang)
                            _sync_visible_language_text(old_lang, lang)
                            ui.notify(t("language_changed"), type="positive")

                    lang_select.on_value_change(on_lang_change)


def page_base(content_func) -> None:
    """页面基础包装器。"""
    apply_theme()
    ui.add_body_html(THEME_SCRIPT)
    create_header()
    create_side_tools()
    content_func()
    ui.run_javascript("""
        requestAnimationFrame(function() {
            if (window.syncThemeToggleIcons) window.syncThemeToggleIcons();
            if (window.queueNormalizeCustomButtons) window.queueNormalizeCustomButtons();
        });
    """)


def home_page() -> None:
    """首页/欢迎页面。"""

    def content() -> None:
        with ui.column().classes(get_classes("page_container") + " gap-6"):
            with ui.element("div").classes("w-full text-center").style("padding: 48px 0 32px;"):
                ui.label(t("app_title", APP_TITLE)).classes("text-h3").style(
                    "font-weight: 600; "
                    "background: linear-gradient(135deg, var(--ql-accent), var(--ql-secondary)); "
                    "-webkit-background-clip: text; -webkit-text-fill-color: transparent; "
                    "background-clip: text;"
                )
                ui.label(t("app_description")).classes("text-body1 app-desc").style("margin-top: 8px;")

            with ui.card().classes(get_classes("card") + " w-full q-pa-lg"):
                with ui.row().classes("w-full items-center justify-between q-mb-md"):
                    ui.label(t("quick_start")).classes("text-h6 section-title").style("font-weight: 600;")
                    with ui.row().classes("gap-2 items-center"):
                        ui.label(t("support")).classes("text-caption").style("color: var(--ql-text-muted);")
                        ui.label(str(len(model_catalog.get_architecture_names()))).classes(get_classes("badge") + " ql-badge--primary")
                        ui.label(t("model_architectures")).classes("text-caption").style("color: var(--ql-text-muted);")

                with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 w-full"):
                    steps = [
                        ("label", t("step") + " 1", t("nav_dataset", "Dataset"), t("feature_list")["workflow"], "/tagging"),
                        ("save", t("step") + " 2", t("cache_processing"), t("feature_list")["efficient"], "/cache"),
                        ("model_training", t("step") + " 3", t("train_lora"), t("feature_list")["multi_arch"], "/train"),
                        ("image", t("step") + " 4", t("inference"), t("feature_list")["modern_ui"], "/generate"),
                    ]

                    for icon, step_num, title, desc, path in steps:
                        with ui.card().classes("step-card flex-1").on("click", lambda p=path: ui.navigate.to(p)):
                            ui.icon(icon, size="36px").style("margin-bottom: 10px;")
                            ui.label(step_num).classes("text-caption text-uppercase step-label")
                            ui.label(title).classes("text-subtitle1 step-title").style("font-weight: 600;")
                            ui.label(desc).classes("text-caption step-desc")

            with ui.card().classes(get_classes("card") + " w-full q-pa-lg"):
                with ui.row().classes("w-full items-center gap-3 q-mb-md"):
                    ui.icon("view_module", size="24px")
                    ui.label(t("supported_models")).classes("text-h6 section-title").style("font-weight: 600;")

                model_list = t("model_architecture_list")
                if not isinstance(model_list, dict):
                    model_list = {}

                with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3 w-full"):
                    for name, arch_info in model_catalog.get_all_architectures().items():
                        arch_id = arch_info.get("id", "")
                        desc = model_list.get(arch_id, name)
                        icon_text = str(arch_info.get("icon") or "✓")
                        color = str(arch_info.get("color") or "#7ee2a5")
                        with ui.row().classes("model-item items-center gap-3"):
                            with ui.element("span").classes("model-icon").style(
                                f"background: {color}1f; border-color: {color}55; color: {color};"
                            ):
                                ui.label(icon_text)
                            with ui.column().classes("gap-0"):
                                ui.label(name).classes("text-body2 model-name").style("font-weight: 500;")
                                ui.label(desc).classes("text-caption model-desc")

            with ui.card().classes(get_classes("card") + " w-full q-pa-lg"):
                with ui.row().classes("w-full items-center gap-3 q-mb-md"):
                    ui.icon("auto_awesome", size="24px")
                    ui.label(t("features")).classes("text-h6 section-title").style("font-weight: 600;")

                feature_list = t("feature_list")
                features = [
                    ("⚡", feature_list["efficient"]),
                    ("🎯", feature_list["multi_arch"]),
                    ("🔧", feature_list["workflow"]),
                    ("💾", feature_list["preset"]),
                    ("📊", feature_list["monitor"]),
                    ("🎨", feature_list["modern_ui"]),
                ]

                with ui.element("div").classes("grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 w-full"):
                    for icon, desc in features:
                        with ui.row().classes("model-item items-start gap-3"):
                            ui.label(icon).classes("text-xl")
                            with ui.column().classes("gap-0"):
                                ui.label(desc).classes("text-body2 feature-desc")

    page_base(content)


def tagging_page() -> None:
    page_base(_load_wizard_attr("wizard.step1_tagging", "render_tagging_step"))


def cache_page() -> None:
    page_base(_load_wizard_attr("wizard.step2_cache", "render_cache_step"))


def train_page() -> None:
    page_base(_load_wizard_attr("wizard.step3_train", "render_train_step"))


def generate_page() -> None:
    page_base(_load_wizard_attr("wizard.step4_generate", "render_generate_step"))


def setup_page() -> None:
    page_base(_load_wizard_attr("wizard.step0_setup", "render_setup_step"))


def console_page() -> None:
    apply_theme()
    ui.add_body_html(THEME_SCRIPT)
    create_side_tools()
    _load_wizard_attr("wizard.console_page", "render_console_page")()


def _install_exception_filter() -> None:
    """在 NiceGUI handler 之前拦截 parent slot 噪音异常。"""
    original_handle = app.handle_exception

    def filtered_handle(exception: Exception) -> None:
        if isinstance(exception, RuntimeError) and "parent slot" in str(exception):
            return
        original_handle(exception)

    app.handle_exception = filtered_handle

    def on_startup() -> None:
        loop = asyncio.get_running_loop()
        default_handler = loop.get_exception_handler()

        def loop_exception_handler(loop, ctx) -> None:
            exception = ctx.get("exception")
            if isinstance(exception, RuntimeError) and "parent slot" in str(exception):
                return
            if default_handler:
                default_handler(loop, ctx)
            else:
                loop.default_exception_handler(ctx)

        loop.set_exception_handler(loop_exception_handler)

    app.on_startup(on_startup)


def main() -> None:
    _install_exception_filter()

    ui.page("/")(home_page)
    ui.page("/tagging")(tagging_page)
    ui.page("/cache")(cache_page)
    ui.page("/train")(train_page)
    ui.page("/generate")(generate_page)
    ui.page("/setup")(setup_page)
    ui.page("/console")(console_page)

    startup_title = t("app_title", APP_TITLE)
    app.config.title = startup_title
    host = resolve_gui_host()
    preferred_port, port = resolve_gui_port()
    native = resolve_gui_native()
    show = resolve_gui_show()

    if port != preferred_port:
        print(f"Preferred port {preferred_port} is busy, falling back to {port}.")

    run_kwargs = {
        "title": startup_title,
        "favicon": "🐉",
        "dark": False,
        "reload": False,
        "host": host,
        "port": port,
        "show": show,
        "native": native,
        "reconnect_timeout": 30.0,
    }
    if native:
        run_kwargs["window_size"] = (1400, 900)

    ui.run(**run_kwargs)


if __name__ in {"__main__", "__mp_main__"}:
    main()

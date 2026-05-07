"""
示例：如何复用 sd-scripts/gui 的样式
演示 Modern Theme 和 Green Gold Theme 的使用
"""

from nicegui import ui
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from theme import (
    apply_theme, 
    get_classes, 
    COLORS, 
    apply_green_gold_styles,
    apply_card,
    apply_button,
    MODERN_CLASSES
)
from utils.i18n import t, set_language, tl


def show_modern_theme():
    """展示 Modern Theme 样式"""
    ui.label('Modern Theme (默认主题)').classes('text-h4 q-mb-lg')
    
    # 颜色展示
    with ui.card().classes(get_classes('card') + ' q-pa-md q-mb-lg'):
        ui.label('主题颜色').classes('text-h6 q-mb-md')
        with ui.row():
            for name in ['primary', 'secondary', 'accent', 'success', 'warning']:
                with ui.element('div').style(f'''
                    width: 80px;
                    height: 40px;
                    background: {COLORS[name]};
                    border-radius: 8px;
                    margin: 4px;
                '''):
                    ui.label(name).classes('text-caption text-center text-white')
    
    # 按钮样式
    with ui.card().classes(get_classes('card') + ' q-pa-md q-mb-lg'):
        ui.label('按钮样式').classes('text-h6 q-mb-md')
        with ui.row():
            btn1 = ui.button('主按钮')
            apply_button(btn1, 'primary')
            
            btn2 = ui.button('次按钮')
            apply_button(btn2, 'secondary')
            
            btn3 = ui.button('危险按钮')
            apply_button(btn3, 'danger')
            
            btn4 = ui.button('幽灵按钮')
            apply_button(btn4, 'ghost')
    
    # 卡片样式
    with ui.row().classes('q-mb-lg'):
        card1 = ui.card()
        apply_card(card1)
        with card1:
            ui.label('标准卡片').classes('text-h6')
            ui.label('这是一个使用 modern-card 样式的卡片')
        
        card2 = ui.card()
        apply_card(card2, hover=True)
        with card2:
            ui.label('悬停效果卡片').classes('text-h6')
            ui.label('鼠标悬停时会提升')
    
    # 步骤卡片 (homepage 样式)
    with ui.card().classes(get_classes('card') + ' q-pa-md q-mb-lg'):
        ui.label('步骤卡片 (Step Cards)').classes('text-h6 q-mb-md')
        with ui.row():
            for i, (icon, title) in enumerate([
                ('label', '步骤 1'),
                ('save', '步骤 2'),
                ('model_training', '步骤 3'),
            ]):
                with ui.card().classes('step-card').style('width: 150px;'):
                    ui.icon(icon, size='32px').classes('q-mb-sm')
                    ui.label(title).classes('text-body2 text-weight-bold')


def show_green_gold_theme():
    """展示 Green Gold Theme 样式 (来自 sd-scripts)"""
    ui.label('Green Gold Theme (sd-scripts 主题)').classes('text-h4 q-mb-lg')
    
    # section-card 样式
    with ui.card().classes('section-card q-pa-md q-mb-lg'):
        ui.label('分区卡片 (section-card)').classes('section-title')
        ui.label('这是来自 sd-scripts 的 section-card 样式，带有浅色背景和边框')
    
    # 按钮样式
    with ui.card().classes('section-card q-pa-md q-mb-lg'):
        ui.label('sd-scripts 按钮样式').classes('section-title')
        with ui.row():
            ui.button('Gold 按钮').classes('gold-btn')
            ui.button('Green 按钮').classes('green-btn')
            ui.button('Red 按钮').classes('red-btn')
    
    # 切换开关
    with ui.card().classes('section-card q-pa-md q-mb-lg'):
        ui.label('切换开关 (Toggle)').classes('section-title')
        with ui.row():
            # 关闭状态
            with ui.button().classes('toggle-container'):
                with ui.element('div').classes('toggle-switch'):
                    ui.element('div').classes('toggle-knob')
                ui.label('选项 1').classes('toggle-label')
                ui.label('OFF').classes('toggle-status')
            
            # 开启状态
            with ui.button().classes('toggle-container active'):
                with ui.element('div').classes('toggle-switch'):
                    ui.element('div').classes('toggle-knob')
                ui.label('选项 2').classes('toggle-label')
                ui.label('ON').classes('toggle-status')
    
    # 滑块样式
    with ui.card().classes('section-card q-pa-md q-mb-lg'):
        ui.label('可编辑滑块 (Editable Slider)').classes('section-title')
        
        # 模拟滑块
        with ui.element('div').classes('editable-slider').style('width: 300px;'):
            with ui.row().classes('slider-label-row'):
                ui.label('学习率').classes('slider-label')
                ui.button('0.0001').classes('slider-value')
            with ui.element('div').classes('slider-container'):
                ui.element('div').classes('slider-track')
                ui.element('div').classes('slider-fill').style('width: 50%;')
                ui.element('div').classes('slider-thumb').style('left: 50%;')


def show_i18n():
    """展示国际化功能"""
    ui.label('国际化 (i18n)').classes('text-h4 q-mb-lg')
    
    with ui.card().classes(get_classes('card') + ' q-pa-md q-mb-lg'):
        ui.label('语言切换').classes('text-h6 q-mb-md')
        
        # 显示当前语言
        lang_label = ui.label(f'当前语言: {t("nav_home")}')
        
        # 语言切换按钮
        with ui.row():
            for lang_code, lang_name in [('zh', '中文'), ('en', 'English'), ('ja', '日本語'), ('ko', '한국어')]:
                ui.button(lang_name, on_click=lambda l=lang_code, lbl=lang_label: (
                    set_language(l),
                    lbl.set_text(f'当前语言: {t("nav_home")}')
                ))
    
    # 翻译键示例
    with ui.card().classes(get_classes('card') + ' q-pa-md'):
        ui.label('常用翻译键').classes('text-h6 q-mb-md')
        keys = ['nav_home', 'nav_train', 'start_train', 'save_preset', 'train_log']
        for key in keys:
            with ui.row().classes('items-center q-mb-sm'):
                ui.label(f'{key}:').classes('text-weight-bold').style('width: 150px;')
                ui.label(t(key))


@ui.page('/')
def main():
    """主页面"""
    # 应用 Modern 主题 (默认)
    apply_theme()
    
    # 或使用 Green Gold 主题
    # apply_green_gold_styles()
    
    with ui.column().classes('q-pa-lg').style('max-width: 1200px; margin: 0 auto;'):
        ui.label('样式复用示例').classes('text-h3 q-mb-xl')
        
        # 创建标签页
        with ui.tabs().classes('w-full') as tabs:
            ui.tab('modern', label='Modern Theme')
            ui.tab('green_gold', label='Green Gold Theme')
            ui.tab('i18n', label='国际化')
        
        with ui.tab_panels(tabs, value='modern').classes('w-full'):
            with ui.tab_panel('modern'):
                show_modern_theme()
            
            with ui.tab_panel('green_gold'):
                show_green_gold_theme()
            
            with ui.tab_panel('i18n'):
                show_i18n()


if __name__ in {'__main__', '__mp_main__'}:
    ui.run(
        title='样式复用示例',
        port=8081,
        reload=False,
        show=True,
    )

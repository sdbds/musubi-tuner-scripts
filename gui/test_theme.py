"""测试主题是否正确应用"""
from nicegui import ui
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from theme import apply_theme, get_classes, COLORS

# 应用主题
apply_theme()

@ui.page('/')
def test_page():
    with ui.column().classes('q-pa-lg'):
        ui.label('主题测试页面').classes('text-h3 q-mb-lg')
        
        # 测试颜色
        ui.label('颜色测试:').classes('text-h6')
        with ui.row():
            for name in ['primary', 'secondary', 'accent', 'success', 'error']:
                with ui.element('div').style(f'''
                    width: 100px;
                    height: 50px;
                    background: {COLORS[name]};
                    border-radius: 8px;
                    margin: 4px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                '''):
                    ui.label(name).classes('text-caption text-white')
        
        # 测试卡片样式
        ui.label('卡片样式测试:').classes('text-h6 q-mt-lg')
        
        # Modern 卡片
        with ui.card().classes(get_classes('card') + ' q-pa-md').style('width: 300px;'):
            ui.label('Modern Card').classes('text-h6')
            ui.label('这个卡片应该使用 modern-card 样式')
        
        # Section 卡片 (sd-scripts 样式)
        with ui.card().classes('section-card q-pa-md q-mt-md').style('width: 300px;'):
            ui.label('Section Card (sd-scripts)').classes('section-title')
            ui.label('这个卡片应该使用 section-card 样式')
        
        # 测试按钮样式
        ui.label('按钮样式测试:').classes('text-h6 q-mt-lg')
        with ui.row():
            ui.button('Gold Button').classes('gold-btn')
            ui.button('Green Button').classes('green-btn')
            ui.button('Red Button').classes('red-btn')
        
        # 显示当前 CSS 类
        ui.label('CSS 类名:').classes('text-h6 q-mt-lg')
        ui.label(f'card: {get_classes("card")}')
        ui.label(f'btn_primary: {get_classes("btn_primary")}')
        ui.label(f'gold_btn: {get_classes("gold_btn")}')
        ui.label(f'green_btn: {get_classes("green_btn")}')

if __name__ in {'__main__', '__mp_main__'}:
    ui.run(
        title='主题测试',
        port=8082,
        reload=False,
        show=True,
    )

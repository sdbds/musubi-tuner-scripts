"""主题调试脚本 - 检查 CSS 是否正确注入"""
from nicegui import ui
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from theme import apply_theme, get_classes, COLORS

print("=" * 50)
print("主题系统诊断")
print("=" * 50)

# 1. 检查导入
print("\n1. 检查导入:")
print(f"   apply_theme: {apply_theme}")
print(f"   get_classes: {get_classes}")
print(f"   COLORS: {type(COLORS)}")

# 2. 检查颜色
print("\n2. 检查颜色配置:")
for key in ['primary', 'secondary', 'accent', 'text', 'background']:
    print(f"   {key}: {COLORS.get(key, 'MISSING')}")

# 3. 检查 CSS 类
print("\n3. 检查 CSS 类映射:")
classes_to_check = ['card', 'btn_primary', 'nav_btn', 'gold_btn', 'green_btn', 'section_card']
for cls in classes_to_check:
    result = get_classes(cls)
    print(f"   {cls} -> '{result}'")

# 4. 应用主题并启动测试页面
print("\n4. 启动测试页面...")
print("   请在浏览器中打开 http://localhost:8083")
print("   按 F12 检查元素样式")

@ui.page('/')
def debug_page():
    # 在页面函数内部应用主题
    apply_theme()
    # 添加明显的调试标记
    ui.add_head_html('''
        <style>
            .debug-marker {
                border: 3px solid red !important;
            }
        </style>
    ''')
    
    with ui.column().classes('q-pa-lg'):
        ui.label('🔧 主题调试页面').classes('text-h3 q-mb-lg')
        
        ui.label('如果你看到这个红色边框，说明自定义 CSS 可以生效：')
        with ui.card().classes('debug-marker q-pa-md q-mb-lg'):
            ui.label('这个卡片应该有红色边框')
        
        ui.label('Modern Card 样式测试:')
        with ui.card().classes(get_classes('card') + ' q-pa-md').style('width: 300px;'):
            ui.label('Modern Card').classes('text-h6')
            ui.label(f'使用的类名: {get_classes("card")}')
            ui.label(f'背景色应该为: {COLORS["surface"]}')
        
        ui.label('Section Card 样式测试 (sd-scripts):')
        with ui.card().classes('section-card q-pa-md q-mt-md').style('width: 300px;'):
            ui.label('Section Card').classes('section-title')
            ui.label('应该有浅色背景和左边框')
        
        ui.label('按钮样式测试:')
        with ui.row():
            ui.button('Gold Button').classes('gold-btn')
            ui.button('Green Button').classes('green-btn')
            ui.button('Red Button').classes('red-btn')
        
        # 显示原始 CSS 信息
        ui.label('调试信息:').classes('text-h6 q-mt-lg')
        ui.label(f'Primary Color: {COLORS["primary"]}')
        ui.label(f'Secondary Color: {COLORS["secondary"]}')
        ui.label(f'Card Class: {get_classes("card")}')
        
        # 检查 body 样式
        ui.label('Body 背景应该是深绿色渐变').classes('q-mt-md')

if __name__ in {'__main__', '__mp_main__'}:
    ui.run(
        title='主题调试',
        port=8083,
        reload=False,
        show=True,
    )

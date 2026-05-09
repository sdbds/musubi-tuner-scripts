"""预设管理组件 - 保存/加载/应用配置预设 - 现代化样式"""
from nicegui import ui
from typing import Dict, Any, Callable
from components.advanced_inputs import styled_select
from utils.config_manager import config_manager
from theme import get_classes, COLORS
from utils.i18n import t


class PresetManager:
    """预设管理器，支持保存和加载配置 - 现代化样式"""
    
    def __init__(self, 
                 get_current_config: Callable[[], Dict[str, Any]],
                 apply_config: Callable[[Dict[str, Any]], None],
                 scope: str,
                 default_name: str = ""):
        self.get_current_config = get_current_config
        self.apply_config = apply_config
        self.scope = scope
        
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            with ui.row().classes('w-full items-center gap-2'):
                # 图标和标题
                ui.icon('bookmarks', size='22px')
                ui.label(t('config_preset')).classes('text-subtitle1 text-weight-bold').style('color: var(--color-text);')
            
            with ui.row().classes('w-full items-end gap-3 q-mt-sm'):
                # 预设选择下拉框
                with ui.column().classes('flex-grow'):
                    self.preset_select = styled_select(
                        self._get_preset_options(),
                        value=default_name if default_name in self._get_preset_list() else None,
                        label=t('config_preset'),
                        icon='bookmarks',
                        placeholder=t('config_preset'),
                        on_change=self._on_preset_change,
                    )
                
                # 操作按钮组
                with ui.row().classes('items-center gap-2'):
                    refresh_btn = ui.button(icon='refresh', on_click=self._refresh_presets)
                    refresh_btn.classes('modern-btn-ghost')
                    refresh_btn.props('dense').tooltip(t('refresh'))
                    
                    save_btn = ui.button(icon='save', on_click=self._show_save_dialog)
                    save_btn.classes('modern-btn-secondary')
                    save_btn.props('dense').tooltip(t('save_preset'))
                    
                    delete_btn = ui.button(icon='delete', on_click=self._delete_preset)
                    delete_btn.classes('modern-btn-ghost')
                    delete_btn.props('dense').tooltip(t('delete_preset'))
    
    def _get_preset_list(self) -> list:
        """获取预设列表"""
        return config_manager.list_configs(self.scope)

    def _get_preset_options(self) -> Dict[str, str]:
        options: Dict[str, str] = {}
        for entry in config_manager.list_config_entries(self.scope):
            suffix = "（内置）" if entry["source"] == "builtin" else "（用户）"
            label = entry.get("label", entry["name"])
            options[entry["name"]] = f'{label} {suffix}'
        return options
    
    def _refresh_presets(self):
        """刷新预设列表"""
        options = self._get_preset_options()
        self.preset_select.options = options
        self.preset_select.update()
        ui.notify('✅ ' + t('refresh') + ' - ' + t('success'), type='positive')
    
    def _on_preset_change(self, value):
        """预设选择变化"""
        if value:
            self._apply_selected()
    
    def _apply_selected(self):
        """应用选中的预设"""
        preset_name = self.preset_select.value
        if not preset_name:
            ui.notify('⚠️ ' + t('config_preset') + ' - ' + t('error'), type='warning')
            return
        
        config = config_manager.load_config(self.scope, preset_name)
        if config:
            self.apply_config(config)
            ui.notify(f'✅ {t("apply_preset")}: {preset_name}', type='positive')
        else:
            ui.notify('❌ ' + t('load') + ' - ' + t('failed'), type='negative')
    
    def _show_save_dialog(self):
        """显示保存预设对话框"""
        with ui.dialog() as dialog:
            with ui.card().classes(get_classes('card') + ' q-pa-lg').style('min-width: 400px;'):
                # 标题
                with ui.row().classes('w-full items-center gap-2 q-mb-md'):
                    ui.icon('save', size='24px')
                    ui.label(t('save_preset')).classes('text-h6 text-weight-bold').style('color: var(--color-text);')
                
                # 输入框
                name_input = ui.input(t('preset_name'), placeholder='输入预设名称')
                name_input.value = self.preset_select.value or ""
                name_input.classes('w-full modern-input q-mb-lg')
                name_input.props('outlined')
                
                # 按钮
                with ui.row().classes('w-full justify-end gap-2'):
                    cancel_btn = ui.button(t('cancel'), on_click=dialog.close)
                    cancel_btn.classes('modern-btn-ghost')
                    
                    save_btn = ui.button(t('save'), on_click=lambda: self._save_preset(name_input.value, dialog))
                    save_btn.classes('modern-btn-primary')
        
        dialog.open()
    
    def _save_preset(self, name: str, dialog):
        """保存当前配置为预设"""
        if not name:
            ui.notify('⚠️ 请输入预设名称', type='warning')
            return
        
        config = self.get_current_config()
        if config_manager.save_config(self.scope, name, config):
            self._refresh_presets()
            self.preset_select.value = name
            self.preset_select.update()
            dialog.close()
            ui.notify(f'✅ 预设已保存: {name}', type='positive')
        else:
            ui.notify('❌ 保存失败', type='negative')
    
    def _delete_preset(self):
        """删除选中的预设"""
        preset_name = self.preset_select.value
        if not preset_name:
            ui.notify('⚠️ 请先选择要删除的预设', type='warning')
            return
        
        # 确认对话框
        with ui.dialog() as dialog:
            with ui.card().classes(get_classes('card') + ' q-pa-lg').style('min-width: 350px;'):
                # 警告图标和标题
                with ui.row().classes('w-full items-center justify-center gap-2 q-mb-md'):
                    ui.icon('warning', size='32px')
                
                ui.label(f'确定要删除预设 "{preset_name}" 吗？').classes('text-body1 text-center q-mb-md').style('color: var(--color-text);')
                ui.label(t('irreversible_action')).classes('text-caption text-center q-mb-lg').style('color: var(--color-text-secondary);')
                
                with ui.row().classes('w-full justify-center gap-2'):
                    cancel_btn = ui.button(t('cancel'), on_click=dialog.close)
                    cancel_btn.classes('modern-btn-ghost')
                    
                    delete_btn = ui.button(t('delete'), on_click=lambda: self._confirm_delete(preset_name, dialog))
                    delete_btn.classes('modern-btn-danger')
        
        dialog.open()
    
    def _confirm_delete(self, name: str, dialog):
        """确认删除"""
        if config_manager.delete_config(self.scope, name):
            self._refresh_presets()
            self.preset_select.value = None
            self.preset_select.update()
            ui.notify(f'🗑️ 已删除预设: {name}', type='positive')
            dialog.close()
            return

        source = config_manager.get_config_source(self.scope, name)
        if source == "builtin":
            ui.notify('⚠️ 内置预设为只读，不能删除', type='warning')
        else:
            ui.notify('❌ 删除失败', type='negative')


def create_preset_manager(get_config: Callable, apply_config: Callable, **kwargs) -> PresetManager:
    """创建预设管理器的便捷函数"""
    return PresetManager(get_config, apply_config, **kwargs)

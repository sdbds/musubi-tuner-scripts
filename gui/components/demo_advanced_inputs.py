"""
Demo page for advanced input components
"""
from nicegui import ui
from components.advanced_inputs import editable_slider, toggle_switch, searchable_select, model_selector
from theme import get_classes, COLORS
from utils.i18n import t


def demo_page():
    """Demo page showcasing advanced input components"""
    
    # Demo config
    demo_config = {
        'learning_rate': 0.0001,
        'batch_size': 4,
        'enable_gradient_checkpointing': True,
        'network_dim': 32,
        'network_alpha': 16,
        'dropout': 0.0,
        'selected_model': '',
        'optimizer': 'AdamW',
    }
    
    with ui.column().classes(get_classes('page_container') + ' gap-4'):
        # Header
        with ui.row().classes('w-full items-center gap-3 q-mb-md'):
            ui.icon('tune', size='32px').style(f'color: {COLORS["primary"]};')
            ui.label('Advanced Components Demo').classes('text-h4 text-weight-bold')
        
        # Editable Sliders Section
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md'):
            ui.label('Editable Sliders').classes('text-h6 text-weight-bold q-mb-md')
            
            with ui.row().classes('w-full gap-4'):
                # Learning rate slider
                editable_slider(
                    label_key='learning_rate',
                    value_ref=demo_config,
                    value_key='learning_rate',
                    min_val=0.00001,
                    max_val=0.01,
                    step=0.00001,
                    decimals=5,
                    label_default='Learning Rate',
                    on_change=lambda v: ui.notify(f'Learning rate: {v}')
                )
                
                # Batch size slider
                editable_slider(
                    label_key='batch_size',
                    value_ref=demo_config,
                    value_key='batch_size',
                    min_val=1,
                    max_val=16,
                    step=1,
                    decimals=0,
                    label_default='Batch Size',
                    on_change=lambda v: ui.notify(f'Batch size: {v}')
                )
            
            with ui.row().classes('w-full gap-4 q-mt-md'):
                # Network dim slider
                editable_slider(
                    label_key='network_dim',
                    value_ref=demo_config,
                    value_key='network_dim',
                    min_val=1,
                    max_val=128,
                    step=1,
                    decimals=0,
                    label_default='Network Dim',
                    on_change=lambda v: ui.notify(f'Network dim: {v}')
                )
                
                # Dropout slider
                editable_slider(
                    label_key='dropout',
                    value_ref=demo_config,
                    value_key='dropout',
                    min_val=0,
                    max_val=1,
                    step=0.01,
                    decimals=2,
                    label_default='Dropout',
                    on_change=lambda v: ui.notify(f'Dropout: {v}')
                )
        
        # Toggle Switches Section
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label('Toggle Switches').classes('text-h6 text-weight-bold q-mb-md')
            
            with ui.row().classes('w-full gap-4 flex-wrap'):
                toggle_switch(
                    label_key='gradient_checkpointing',
                    value_ref=demo_config,
                    value_key='enable_gradient_checkpointing',
                    label_default='Gradient Checkpointing',
                    on_change=lambda v: ui.notify(f'Gradient checkpointing: {v}')
                )
                
                toggle_switch(
                    label_key='fp8_base',
                    value_ref=demo_config,
                    value_key='fp8_base',
                    label_default='FP8 Base Model',
                    on_change=lambda v: ui.notify(f'FP8 base: {v}')
                )
                
                toggle_switch(
                    label_key='full_bf16',
                    value_ref=demo_config,
                    value_key='full_bf16',
                    label_default='Full BF16',
                    on_change=lambda v: ui.notify(f'Full BF16: {v}')
                )
        
        # Searchable Dropdowns Section
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label('Searchable Dropdowns').classes('text-h6 text-weight-bold q-mb-md')
            
            # Model selector
            ui.label('Model Selector:').classes('text-subtitle2 q-mb-sm')
            model_selector(
                value_ref=demo_config,
                value_key='selected_model',
                on_change=lambda v: ui.notify(f'Selected model: {v}')
            )
            
            # Generic searchable select
            ui.label('Optimizer:').classes('text-subtitle2 q-mb-sm q-mt-md')
            searchable_select(
                options={
                    'AdamW': 'AdamW',
                    'AdamW8bit': 'AdamW 8-bit',
                    'Lion': 'Lion',
                    'Lion8bit': 'Lion 8-bit',
                    'SGD': 'SGD',
                    'Adafactor': 'Adafactor',
                    'Prodigy': 'Prodigy',
                },
                value_ref=demo_config,
                value_key='optimizer',
                label_key='optimizer',
                label_default='Optimizer',
                placeholder_key='search_or_select',
                placeholder_default='Search or select optimizer...',
                on_change=lambda v: ui.notify(f'Optimizer: {v}')
            )
        
        # Current Values Display
        with ui.card().classes(get_classes('card') + ' w-full q-pa-md q-mt-md'):
            ui.label('Current Configuration').classes('text-h6 text-weight-bold q-mb-md')
            
            values_display = ui.json_editor({'content': demo_config}).classes('w-full')
            values_display.props('readonly mode="text"')
            
            def refresh_values():
                values_display.update_value({'content': demo_config})
            
            ui.button('Refresh Values', on_click=refresh_values).classes('q-mt-md')


if __name__ == '__main__':
    from theme import apply_theme
    apply_theme()
    demo_page()
    ui.run(title='Advanced Components Demo')

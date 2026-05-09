"""
Advanced Input Components
Includes: Editable Slider, Toggle Switch, Searchable Dropdown
From sd-scripts/gui with enhancements
"""
from nicegui import ui
from typing import Optional, Callable, Dict, Any, List
from utils.i18n import t, get_i18n
from theme import COLORS
import uuid


def editable_slider(
    label_key: str,
    value_ref: Dict[str, Any],
    value_key: str,
    min_val: float,
    max_val: float,
    step: float = 1,
    decimals: int = 0,
    label_default: str = None,
    flex: int = 1,
    on_change: Callable = None
):
    """
    Create an editable slider component with two-way binding
    
    Args:
        label_key: Translation key for the label
        value_ref: Dictionary containing the value (e.g., self.config)
        value_key: Key in the dictionary for this value
        min_val: Minimum value
        max_val: Maximum value
        step: Step size
        decimals: Number of decimal places to display
        label_default: Default label text if translation not found
        flex: Flex grow value for layout
        on_change: Callback when value changes
    """
    with ui.element('div').classes('editable-slider').style(
        f'flex: {flex}; margin: 0; padding: 0; min-width: 140px; min-height: 56px;'
    ):
        # Label row with value display
        with ui.row().classes('w-full items-center justify-between no-wrap').style('margin: 0; padding: 0; min-height: 20px;'):
            label_el = ui.label(t(label_key, label_default or label_key)).classes('slider-label').style(
                'min-width: 60px; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin: 0; padding: 0;'
            )
            
            # Editable value display
            current_val = value_ref.get(value_key, min_val)
            value_btn = ui.button(f'{current_val:.{decimals}f}').props('flat dense type="button"').classes('slider-value')
            value_btn.style('padding: 0 4px; min-height: 18px; height: 18px; font-size: 11px; margin: 0;')
        
        # Register for translation updates
        def update_label():
            try:
                label_el.set_text(t(label_key, label_default or label_key))
            except Exception:
                pass
        get_i18n().bind(update_label)
        
        # NiceGUI native slider
        slider = ui.slider(min=min_val, max=max_val, step=step, value=current_val).classes('w-full').style(
            'margin: 0; padding: 0; min-height: 16px; height: 16px;'
        )
        slider.props('dense')
        
        # Sync value display when slider changes
        def sync_display():
            val = slider.value
            value_ref[value_key] = val
            value_btn.set_text(f'{val:.{decimals}f}')
            if on_change:
                on_change(val)
        
        slider.on_value_change(sync_display)
        
        # Click on value to edit
        def start_edit():
            current_val = value_ref.get(value_key, min_val)
            value_btn.visible = False
            
            input_id = f'slider-edit-{uuid.uuid4().hex[:8]}'
            
            edit_container = ui.element('span')
            with edit_container:
                edit_input = ui.input(value=f'{current_val:.{decimals}f}')\
                    .classes('slider-edit-input')\
                    .style('width: 60px;')\
                    .props(f'id="{input_id}"')
            
            finished = [False]
            
            def finish_edit():
                if finished[0]:
                    return
                finished[0] = True
                
                try:
                    new_val = float(edit_input.value)
                    new_val = max(min_val, min(max_val, new_val))
                    if decimals == 0:
                        new_val = int(new_val)
                    else:
                        new_val = round(new_val, decimals)
                    
                    value_ref[value_key] = new_val
                    slider.set_value(new_val)
                    value_btn.set_text(f'{new_val:.{decimals}f}')
                    
                    if on_change:
                        on_change(new_val)
                except ValueError:
                    pass
                finally:
                    edit_container.delete()
                    value_btn.visible = True
            
            edit_input.on('blur', finish_edit)
            edit_input.on('keyup.enter', finish_edit)
            
            ui.run_javascript(f'''
                setTimeout(() => {{
                    const input = document.getElementById('{input_id}');
                    if (input) {{
                        input.focus();
                        input.select();
                    }}
                }}, 10);
            ''')
        
        value_btn.on_click(start_edit)

        def set_bound_value(new_val: Any):
            try:
                numeric_val = float(new_val)
                if decimals == 0:
                    numeric_val = int(numeric_val)
                else:
                    numeric_val = round(numeric_val, decimals)
                value_ref[value_key] = numeric_val
                slider.set_value(numeric_val)
                value_btn.set_text(f'{numeric_val:.{decimals}f}')
            except (TypeError, ValueError):
                return

        slider.set_bound_value = set_bound_value
    
    return slider


def toggle_switch(
    label_key: str,
    value_ref: Dict[str, Any],
    value_key: str,
    label_default: str = None,
    on_change: Callable = None
):
    """
    Create a toggle switch button (turn on/off style)
    
    Args:
        label_key: Translation key for the label
        value_ref: Dictionary containing the value
        value_key: Key in the dictionary for this value
        label_default: Default label text if translation not found
        on_change: Callback when value changes
    """
    value = value_ref.get(value_key, False)
    
    btn = ui.button().props('flat unelevated').classes(f'toggle-container {"active" if value else ""}')
    btn.value = bool(value)
    
    with btn:
        with ui.element('div').classes('toggle-switch'):
            ui.element('div').classes('toggle-knob')
        
        label_el = ui.label(t(label_key, label_default or label_key)).classes('toggle-label')
        
        status_text = t('status_on') if value else t('status_off')
        status_label = ui.label(status_text).classes('toggle-status')
    
    # Register for translation updates
    def update_toggle_text():
        try:
            label_el.set_text(t(label_key, label_default or label_key))
            current_val = value_ref.get(value_key, False)
            status_label.set_text(t('status_on') if current_val else t('status_off'))
        except Exception:
            pass
    get_i18n().bind(update_toggle_text)
    
    def apply_value(new_value: bool):
        new_value = bool(new_value)
        value_ref[value_key] = new_value
        btn.value = new_value
        
        if new_value:
            btn.classes('active')
            status_label.set_text(t('status_on'))
        else:
            btn.classes(remove='active')
            status_label.set_text(t('status_off'))
        
        if on_change:
            on_change(new_value)

    def toggle():
        apply_value(not value_ref.get(value_key, False))
    
    btn.on_click(toggle)
    btn.set_toggle_value = apply_value
    return btn


def searchable_select(
    options: Dict[str, str],
    value_ref: Dict[str, Any],
    value_key: str,
    label_key: str = None,
    label_default: str = None,
    placeholder_key: str = None,
    placeholder_default: str = 'Search or select...',
    on_change: Callable = None,
    classes: str = '',
    style: str = ''
):
    """
    Create a searchable dropdown select with input filtering
    
    Args:
        options: Dictionary of {value: label} pairs
        value_ref: Dictionary containing the value
        value_key: Key in the dictionary for this value
        label_key: Translation key for the label
        label_default: Default label text
        placeholder_key: Translation key for placeholder
        placeholder_default: Default placeholder text
        on_change: Callback when value changes
        classes: Additional CSS classes
        style: Additional inline styles
    """
    current_value = value_ref.get(value_key, list(options.keys())[0] if options else None)
    
    with ui.column().classes(f'w-full {classes}').style(style):
        if label_key:
            label_el = ui.label(t(label_key, label_default or label_key)).classes('text-sm font-medium q-mb-xs')
            
            def update_label():
                try:
                    label_el.set_text(t(label_key, label_default or label_key))
                except Exception:
                    pass
            get_i18n().bind(update_label)
        
        select = ui.select(
            options,
            value=current_value,
            label=''
        ).classes('w-full modern-select force-light-bg')
        
        # Enable search/filter functionality
        select.props('dense stack-label use-input fill-input hide-selected input-debounce="0" dropdown-icon="search"')
        select.props(f'placeholder="{t(placeholder_key, placeholder_default)}"')
        
        def on_value_change(e):
            value_ref[value_key] = e.value
            if on_change:
                on_change(e.value)
        
        select.on_value_change(on_value_change)
    
    return select


def styled_select(
    options: List[str] | Dict[str, str],
    value: Any = None,
    label: str = '',
    icon: str = 'arrow_drop_down',
    icon_color: str = None,
    placeholder: str = 'Search or select...',
    on_change: Callable = None,
    flex: int = None,
    searchable: bool = True,
):
    """Create a consistent select wrapper that avoids label/value overlap."""
    icon_color = icon_color or COLORS["primary"]
    style = f'flex: {flex};' if flex else ''

    with ui.column().classes('w-full styled-select-container').style(style):
        if label:
            with ui.row().classes('items-center gap-2 q-mb-xs'):
                ui.icon(icon, size='18px')
                ui.label(label).classes('text-caption text-weight-medium').style('color: var(--color-text-secondary);')

        select = ui.select(options=options, value=value, label='').classes('w-full modern-select force-light-bg')
        dropdown_icon = 'search' if searchable else 'arrow_drop_down'
        props = f'dense stack-label dropdown-icon="{dropdown_icon}" placeholder="{placeholder}"'
        if searchable:
            props += ' use-input fill-input hide-selected input-debounce="0"'
        select.props(props)

        if on_change:
            select.on_value_change(lambda e: on_change(e.value))

    return select


def toggle_switch_simple(
    label_key: str,
    value: bool = True,
    on_change: Callable = None,
    label_default: str = None,
):
    """Compact wrapper around the project button-style toggle.

    Returns:
        (switch_element, get_value_fn) tuple
    """
    state = {"value": bool(value)}
    switch = toggle_switch(
        label_key,
        state,
        "value",
        label_default=label_default or label_key,
        on_change=on_change,
    )

    def get_value():
        return state["value"]

    return switch, get_value


def model_selector(
    value_ref: Dict[str, Any],
    value_key: str = 'pretrained_model',
    label_key: str = 'pretrained_model',
    label_default: str = 'Pretrained Model',
    on_change: Callable = None
):
    """
    Create a searchable model selector with common SD models
    """
    # Common model options - can be extended
    model_options = {
        '': t('select_model', 'Select a model...'),
        'runwayml/stable-diffusion-v1-5': 'SD 1.5',
        'stabilityai/stable-diffusion-2-1': 'SD 2.1',
        'stabilityai/stable-diffusion-xl-base-1.0': 'SDXL 1.0',
        'stabilityai/stable-diffusion-xl-refiner-1.0': 'SDXL Refiner',
        'madebyollin/sdxl-vae-fp16-fix': 'SDXL VAE FP16',
        'black-forest-labs/FLUX.2-dev': 'FLUX.2 Dev',
        'black-forest-labs/FLUX.2-schnell': 'FLUX.2 Schnell',
    }
    
    return searchable_select(
        options=model_options,
        value_ref=value_ref,
        value_key=value_key,
        label_key=label_key,
        label_default=label_default,
        placeholder_key='search_model',
        placeholder_default='Search or type model name...',
        on_change=on_change
    )

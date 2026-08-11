"""
Advanced Input Components
Includes: Editable Slider, Toggle Switch, Searchable Dropdown
From sd-scripts/gui with enhancements
"""
import math
import uuid
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Callable, Dict, List

from nicegui import ui
from theme import COLORS
from utils.i18n import get_i18n, t


_USE_TRACK_BOUND = object()
_JS_MAX_SAFE_INTEGER = 2**53 - 1


def _register_bound_control(
    value_ref: Dict[str, Any],
    value_key: str,
    control: Any,
    cleanup: Callable[[], None] | None = None,
) -> None:
    bound_controls = value_ref.setdefault("_bound_controls", {})
    previous_control = bound_controls.get(value_key)
    if previous_control is not None and previous_control is not control:
        previous_dispose = getattr(previous_control, "dispose_form_binding", None)
        if callable(previous_dispose):
            previous_dispose()
    bound_controls[value_key] = control
    disposed = [False]

    def dispose_form_binding() -> None:
        if disposed[0]:
            return
        disposed[0] = True
        if cleanup:
            cleanup()
        if bound_controls.get(value_key) is control:
            bound_controls.pop(value_key, None)

    control.dispose_form_binding = dispose_form_binding
    original_handle_delete = getattr(control, "_handle_delete", None)
    if callable(original_handle_delete):
        def handle_delete() -> None:
            dispose_form_binding()
            original_handle_delete()

        control._handle_delete = handle_delete


def _finite_decimal_or_none(value: Any) -> Decimal | None:
    try:
        numeric_val = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not numeric_val.is_finite():
        return None
    return numeric_val


def _decimal_as_scaled_integer(value: Decimal, exponent: int) -> int:
    sign, digits, value_exponent = value.as_tuple()
    coefficient = int(''.join(str(digit) for digit in digits) or '0')
    if sign:
        coefficient = -coefficient
    return coefficient * 10 ** (value_exponent - exponent)


def _decimal_from_scaled_integer(value: int, exponent: int) -> Decimal:
    sign = 1 if value < 0 else 0
    digits = tuple(int(digit) for digit in str(abs(value))) or (0,)
    return Decimal((sign, digits, exponent))


def _snap_decimal_to_step(value: Decimal, base: Decimal, step: Decimal) -> Decimal:
    """Snap with integer arithmetic so Decimal context precision cannot alter values."""
    exponent = min(value.as_tuple().exponent, base.as_tuple().exponent, step.as_tuple().exponent)
    scaled_value = _decimal_as_scaled_integer(value, exponent)
    scaled_base = _decimal_as_scaled_integer(base, exponent)
    scaled_step = _decimal_as_scaled_integer(step, exponent)
    offset = scaled_value - scaled_base
    quotient, remainder = divmod(abs(offset), scaled_step)
    if remainder * 2 >= scaled_step:
        quotient += 1
    if offset < 0:
        quotient = -quotient
    return _decimal_from_scaled_integer(scaled_base + quotient * scaled_step, exponent)


def _coerce_slider_value(
    value: Any,
    min_val: float,
    max_val: float,
    decimals: int | None,
    fallback: Any = None,
    *,
    step: float | None = None,
    hard_min_val: float | None | object = _USE_TRACK_BOUND,
    hard_max_val: float | None | object = _USE_TRACK_BOUND,
) -> int | float | None:
    numeric_val = _finite_decimal_or_none(value)
    if numeric_val is None:
        if fallback is None:
            return None
        numeric_val = _finite_decimal_or_none(fallback)
        if numeric_val is None:
            numeric_val = _finite_decimal_or_none(min_val)
        if numeric_val is None:
            return None

    hard_min = min_val if hard_min_val is _USE_TRACK_BOUND else hard_min_val
    hard_max = max_val if hard_max_val is _USE_TRACK_BOUND else hard_max_val
    hard_min_numeric = None if hard_min is None else _finite_decimal_or_none(hard_min)
    hard_max_numeric = None if hard_max is None else _finite_decimal_or_none(hard_max)
    if hard_min is not None and hard_min_numeric is None:
        raise ValueError("hard_min_val must be finite or None")
    if hard_max is not None and hard_max_numeric is None:
        raise ValueError("hard_max_val must be finite or None")

    if hard_min_numeric is not None:
        numeric_val = max(hard_min_numeric, numeric_val)
    if hard_max_numeric is not None:
        numeric_val = min(hard_max_numeric, numeric_val)

    if step is not None:
        step_numeric = _finite_decimal_or_none(step)
        step_base = _finite_decimal_or_none(min_val)
        if step_numeric is None or step_numeric <= 0 or step_base is None:
            raise ValueError("step must be a positive finite number")
        numeric_val = _snap_decimal_to_step(numeric_val, step_base, step_numeric)

        if hard_min_numeric is not None:
            numeric_val = max(hard_min_numeric, numeric_val)
        if hard_max_numeric is not None:
            numeric_val = min(hard_max_numeric, numeric_val)

    if decimals == 0:
        return int(numeric_val)
    if decimals is None:
        float_value = float(numeric_val)
        return float_value if math.isfinite(float_value) else None
    if step is None:
        return round(float(numeric_val), decimals)
    quantum = Decimal(1).scaleb(-decimals)
    return float(numeric_val.quantize(quantum, rounding=ROUND_HALF_UP))


def editable_slider(
    label_key: str,
    value_ref: Dict[str, Any],
    value_key: str,
    min_val: float,
    max_val: float,
    step: float = 1,
    decimals: int | None = 0,
    label_default: str = None,
    flex: int = 1,
    on_change: Callable = None,
    hard_min_val: float | None | object = _USE_TRACK_BOUND,
    hard_max_val: float | None | object = _USE_TRACK_BOUND,
    snap_to_step: bool = False,
    allow_empty: bool = False,
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
        decimals: Number of decimal places to round/display, or None to preserve float precision
        label_default: Default label text if translation not found
        flex: Flex grow value for layout
        on_change: Callback when value changes
        hard_min_val: Hard input minimum; defaults to the track minimum
        hard_max_val: Hard input maximum; defaults to the track maximum
        snap_to_step: Snap typed and preset values to the slider step when true
        allow_empty: Preserve an empty string as an optional, unset value
    """
    def is_empty_value(value: Any) -> bool:
        return value is None or (isinstance(value, str) and not value.strip())

    def coerce_value(value: Any, fallback: Any = None) -> int | float | str | None:
        if allow_empty and is_empty_value(value):
            return ""
        return _coerce_slider_value(
            value,
            min_val,
            max_val,
            decimals,
            fallback=fallback,
            step=step if snap_to_step else None,
            hard_min_val=hard_min_val,
            hard_max_val=hard_max_val,
        )

    def slider_proxy_value(value: int | float) -> int | float:
        if abs(value) > _JS_MAX_SAFE_INTEGER:
            return min(max(value, min_val), max_val)
        return value

    def format_numeric_value(value: int | float) -> str:
        return str(value) if decimals in {0, None} else f'{value:.{decimals}f}'

    def format_display_value(value: int | float | str) -> str:
        return "-" if allow_empty and is_empty_value(value) else format_numeric_value(value)

    def format_edit_value(value: int | float | str) -> str:
        return "" if allow_empty and is_empty_value(value) else format_numeric_value(value)

    with ui.element('div').classes('editable-slider').style(
        f'flex: {flex}; margin: 0; padding: 0; min-width: 140px; min-height: 56px;'
    ):
        # Label row with value display
        with ui.row().classes('w-full items-center justify-between no-wrap').style('margin: 0; padding: 0; min-height: 20px;'):
            label_el = ui.label(t(label_key, label_default or label_key)).classes('slider-label').style(
                'min-width: 60px; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin: 0; padding: 0;'
            )

            # Editable value display
            initial_default = "" if allow_empty else min_val
            current_val = coerce_value(value_ref.get(value_key, initial_default), fallback=min_val)
            value_ref[value_key] = current_val
            value_btn = ui.button(format_display_value(current_val)).props('flat dense type="button"').classes('slider-value')
            value_btn.style('padding: 0 4px; min-height: 18px; height: 18px; font-size: 11px; margin: 0;')

            input_id = f'slider-edit-{uuid.uuid4().hex[:8]}'
            edit_container = ui.element('span')
            edit_container.visible = False
            with edit_container:
                edit_input = ui.input(value=format_edit_value(current_val))\
                    .classes('slider-edit-input')\
                    .style('width: 60px;')\
                    .props(f'id="{input_id}"')

        # Register for translation updates
        def update_label():
            try:
                label_el.set_text(t(label_key, label_default or label_key))
            except Exception:
                pass
        i18n = get_i18n()
        i18n.bind(update_label)

        # NiceGUI native slider
        current_slider_val = slider_proxy_value(min_val if is_empty_value(current_val) else current_val)
        track_min = min(min_val, current_slider_val)
        track_max = max(max_val, current_slider_val)
        slider = ui.slider(
            min=track_min,
            max=track_max,
            step=step,
            value=current_slider_val,
        ).classes('w-full').style(
            'margin: 0; padding: 0; min-height: 16px; height: 16px;'
        )
        slider.props('dense')

        editing = [False]
        suppress_slider_event = [False]

        def update_track_range(value: int | float) -> None:
            effective_min = min(min_val, value)
            effective_max = max(max_val, value)
            if slider._props.get('min') == effective_min and slider._props.get('max') == effective_max:
                return
            # String props make Quasar misplace thumbs when the step is greater than one.
            slider._props['min'] = effective_min
            slider._props['max'] = effective_max
            slider.update()

        def apply_value(raw_value: Any, notify: bool = True) -> int | float | str | None:
            new_val = coerce_value(raw_value)
            if new_val is None:
                return None

            previous_val = coerce_value(value_ref.get(value_key), fallback=new_val)
            slider_val = slider_proxy_value(min_val if is_empty_value(new_val) else new_val)
            update_track_range(slider_val)
            if slider.value != slider_val:
                suppress_slider_event[0] = True
                try:
                    slider.set_value(slider_val)
                finally:
                    suppress_slider_event[0] = False

            value_ref[value_key] = new_val
            value_btn.set_text(format_display_value(new_val))
            if editing[0]:
                edit_input.set_value(format_edit_value(new_val))
            if notify and on_change and previous_val != new_val:
                on_change(new_val)
            return new_val

        # Sync value display when slider changes
        def sync_display():
            if suppress_slider_event[0]:
                return
            apply_value(slider.value)

        slider.on_value_change(sync_display)

        def finish_edit():
            if not editing[0]:
                return
            editing[0] = False

            try:
                apply_value(edit_input.value)
            finally:
                edit_container.visible = False
                value_btn.visible = True

        # Keep one hidden editor per slider. Removing a NiceGUI input from its own
        # key event races the component's beforeUnmount value synchronization.
        def start_edit():
            current_val = apply_value(value_ref.get(value_key, initial_default), notify=False)
            edit_input.set_value(format_edit_value(current_val))
            editing[0] = True
            value_btn.visible = False
            edit_container.visible = True

            ui.run_javascript(f'''
                setTimeout(() => {{
                    const input = document.getElementById('{input_id}');
                    if (input) {{
                        input.focus();
                        input.select();
                    }}
                }}, 10);
            ''')

        edit_input.on('blur', finish_edit)
        edit_input.on('keyup.enter', finish_edit)
        value_btn.on_click(start_edit)

        def set_bound_value(new_val: Any):
            apply_value(new_val)

        def get_bound_value() -> Any:
            return value_ref.get(value_key)

        slider.set_bound_value = set_bound_value
        slider.get_bound_value = get_bound_value
        _register_bound_control(
            value_ref,
            value_key,
            slider,
            cleanup=lambda: i18n.unbind(update_label),
        )

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
    i18n = get_i18n()
    i18n.bind(update_toggle_text)

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
    _register_bound_control(
        value_ref,
        value_key,
        btn,
        cleanup=lambda: i18n.unbind(update_toggle_text),
    )
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

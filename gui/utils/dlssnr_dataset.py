"""Dataset-page fields for paired NR inputs and fixed network conditions."""

from decimal import Decimal

NR_CONDITION_DEFAULTS = {
    "nr_style": 0,
    "nr_tone": 1.0,
    "nr_structure": 1.0,
    "nr_skin": -1.0,
    "nr_auto_mask": True,
}
NR_DATASET_FIELDS = ("train_manifest", "validation_manifest", "sequence_manifest", "nr_controls_mode", *NR_CONDITION_DEFAULTS)


def nr_dataset_row_state(dataset):
    return {
        **{name: dataset.get(name, value) for name, value in NR_CONDITION_DEFAULTS.items()},
        **{name: dataset.get(name, "") for name in ("train_manifest", "validation_manifest", "sequence_manifest")},
        "nr_controls_mode": dataset.get("nr_controls_mode", "files" if dataset.get("train_manifest") else "fixed"),
    }


def collect_nr_dataset_fields(state):
    from musubi_tuner.dlssnr.controls import resolve_fixed_controls

    from utils.dlssnr_commands import parse_nr_boolean, validate_nr_numeric_value

    mode = "fixed" if state.get("dataset_source", "directory") == "directory" else state.get("nr_controls_mode", "files")
    if mode not in ("fixed", "files"):
        raise ValueError("nr_controls_mode must be fixed or files")
    fields = {"nr_controls_mode": mode}
    numbers = {}
    for name in ("batch_size", "num_repeats", "resolution_w", "resolution_h"):
        value = state.get(name, "")
        if value is not None and str(value).strip():
            validate_nr_numeric_value(value, name, integer=True, minimum=1)
            numbers[name] = int(Decimal(str(value)))
    for name in ("batch_size", "num_repeats"):
        if name in numbers:
            fields[name] = numbers[name]
    if "resolution_w" in numbers or "resolution_h" in numbers:
        if not {"resolution_w", "resolution_h"} <= numbers.keys():
            raise ValueError("resolution_w and resolution_h must both be specified")
        from musubi_tuner.dlssnr.geometry import resolve_geometry

        resolve_geometry(numbers["resolution_w"], numbers["resolution_h"])
        fields["resolution"] = [numbers["resolution_w"], numbers["resolution_h"]]
    if mode == "fixed":
        values = {key: state.get(key, value) for key, value in NR_CONDITION_DEFAULTS.items()}
        validate_nr_numeric_value(values["nr_style"], "nr_style", integer=True)
        values["nr_style"] = int(Decimal(str(values["nr_style"])))
        for name in ("nr_tone", "nr_structure", "nr_skin"):
            validate_nr_numeric_value(values[name], name, minimum=-1 if name == "nr_skin" else 0, maximum=1)
            values[name] = float(values[name])
        values["nr_auto_mask"] = parse_nr_boolean(values["nr_auto_mask"], "nr_auto_mask")
        fields.update(resolve_fixed_controls(values))
    for name in ("validation_manifest", "sequence_manifest"):
        value = str(state.get(name, "")).strip()
        if value:
            fields[name] = value
    return fields

"""Optimizer choices shared by the training forms."""

import io
import tokenize

OPTIMIZER_TYPES = [
    "AdamW",
    "AdamW8bit",
    "PagedAdamW8bit",
    "AdamW_adv",
    "Prodigy_adv",
    "Adopt_adv",
    "Lion_adv",
    "Lion_Prodigy_adv",
    "Simplified_AdEMAMix",
    "Prodigy",
    "Lion",
    "Lion8bit",
    "PagedLion8bit",
    "adafactor",
    "Sophia",
    "Ranger",
    "Adan",
    "StableAdamW",
    "Tiger",
    "AdEMAMix8bit",
    "PagedAdEMAMix8bit",
    "ademamix",
    "SOAP",
    "sgdsai",
    "adopt",
    "Fira",
    "came",
    "LoRARite",
    "FlashAdamW",
    "DualAdam",
    "ROSE",
    "adammini",
    "adamg",
    "AdaMuon",
    "BCOS",
    "Ano",
    "EmoNavi",
    "EmoFact",
    "EmoLynx",
    "EmoNeco",
    "EmoZeal",
    "DAdaptAdam",
    "DAdaptLion",
    "DAdaptAdan",
    "DAdaptSGD",
    "AdamWScheduleFree",
    "SGDScheduleFree",
]

# The shared backend accepts class paths, not these UI aliases.
OPTIMIZER_CLASS_PATHS = {
    **{
        name.lower(): f"bitsandbytes.optim.{name}"
        for name in (
            "PagedAdamW8bit",
            "Lion8bit",
            "PagedLion8bit",
            "AdEMAMix8bit",
            "PagedAdEMAMix8bit",
        )
    },
    **{
        name.lower(): f"pytorch_optimizer.{name}"
        for name in (
            "DAdaptAdam",
            "DAdaptLion",
            "DAdaptAdan",
            "DAdaptSGD",
        )
    },
    "adamg": "pytorch_optimizer.AdamG",
    "adv_optm.simplifiedademamix": "adv_optm.Simplified_AdEMAMix",
}


def parse_optimizer_args(value: str | None) -> list[str]:
    """Split key=value entries without splitting quoted or grouped Python literals."""
    source = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise ValueError("optimizer_args contains an incomplete Python literal") from exc
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    def position(token):
        return offsets[token.start[0] - 1] + token.start[1]

    cleaned = list(source)
    significant = []
    for token in tokens:
        if token.type == tokenize.COMMENT:
            start = position(token)
            cleaned[start : start + len(token.string)] = " " * len(token.string)
        elif token.type not in {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}:
            significant.append(token)
    source = "".join(cleaned)
    starts = []
    depth = 0
    for index, token in enumerate(significant):
        if token.type == tokenize.OP:
            if token.string in {"(", "[", "{"}:
                depth += 1
            elif token.string in {")", "]", "}"}:
                depth -= 1
        if depth == 0 and token.type == tokenize.NAME and index + 1 < len(significant):
            if significant[index + 1].string == "=":
                starts.append(position(token))
    if not source.strip():
        return []
    if not starts or source[: starts[0]].strip():
        raise ValueError("optimizer_args requires key=value entries, not CLI options")
    ends = [*starts[1:], len(source)]
    return [source[start:end].strip() for start, end in zip(starts, ends)]

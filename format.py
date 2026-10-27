"""Data formats: the dataset row, the prediction row, and the value each card field is scored on.

Dataset row (one per scene)::

    {"eid": "--BfvyPmVMo_20000",
     "audio": <HF Audio or a wav path>,      # for the audio model; never read by the scorer
     "duration": 10.0,
     "sources": [{"type": "speech" | "music" | "sfx", "label": "Hand",
                  "spans": [[3.007, 3.657], [6.374, 7.024]]}, ...],
     "cards": {"speech": {<field>: <value>} | None,
               "music":  {<field>: <value>} | None,
               "sfx":    [{"class": "Hand", <field>: <value>, ...}, ...]}}

Card values are already reduced by :func:`reduce_card`, so a reference card holds exactly
what gets compared. ``cards`` may also be stored as a JSON string.

Prediction row (JSONL, one per scene)::

    {"eid": "--BfvyPmVMo_20000",
     "sources": [{"type": "speech" | "music" | "sound effect", "name": "machine operation",
                  "spans": [[0.0, 7.1]]}, ...],
     "cards": [{"type": "sound effect", "name": "machine operation",
                "card": {<raw card JSON>} | None}, ...]}

``sources`` is the Stage-1 inventory (``parse_inventory`` reads the numbered-list answer);
``cards`` holds one Stage-2 card per source, raw as the model wrote it (``parse_card`` reads
the answer). A card that did not parse is ``None``: it is scored as blank, not dropped.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter, namedtuple
from typing import Any

Loudness = namedtuple("Loudness", "value label")

MODALITIES = ("speech", "music", "sfx")

# Source type as the prompts and predictions name it -> modality.
SOURCE_TYPES = {"speech": "speech", "music": "music", "sound effect": "sfx", "sfx": "sfx"}

LOUDNESS_SCALE = ["Extremely quiet", "Very quiet", "Quiet", "Medium", "Loud", "Very loud"]
BAND_SCALE = ["very-low", "low", "medium-low", "medium", "medium-high", "high", "very-high"]
PLURALITY_SCALE = ["none", "one", "multiple"]
TEMPO_SCALE = ["very slow", "slow", "medium", "fast", "very fast"]
BANDS = ["125", "250", "500", "1000", "2000", "3000", "4000", "6000", "8000", "10000"]
PLURALITY_ALIASES = {"single": "one", "multiple sources": "multiple", "several": "multiple"}

# Robust dispersion of the reference median F0 in semitones (1.4826 * MAD of 12*log2(hz)).
F0_SIGMA = 5.1918783199023695

# Scored fields per modality. kind decides the reduction and the metric; hint is the phrase the
# judge is told the field holds; path reads a sub-field; lemmatize folds inflections.
FIELDS: dict[str, dict[str, dict[str, Any]]] = {
    "speech": {
        "loudness": {"kind": "loudness"},
        "frequency_profile": {"kind": "bands"},
        "median_f0_hz": {"kind": "f0"},
        "intelligibility": {"kind": "tags", "path": "words",
                            "hint": "speech intelligibility descriptors"},
        "emotion": {"kind": "tags", "path": "words", "hint": "speaker emotion"},
        "manner": {"kind": "tags", "hint": "manner-of-speech descriptors"},
        "perceived_audio_quality": {"kind": "tags", "hint": "acoustic-quality descriptors"},
        "short_caption": {"kind": "caption", "hint": "short audio caption", "target_words": 4},
        "descriptive_caption": {"kind": "caption", "hint": "descriptive audio caption"},
    },
    "music": {
        "loudness": {"kind": "loudness"},
        "frequency_profile": {"kind": "bands"},
        "instruments": {"kind": "tags", "hint": "instruments present"},
        "genre": {"kind": "tags", "hint": "music genre tags"},
        "mood": {"kind": "tags", "hint": "musical mood"},
        "tempo": {"kind": "tempo"},
        "music_descriptors": {"kind": "tags", "hint": "musical texture and character descriptors"},
        "perceived_audio_quality": {"kind": "tags", "hint": "acoustic-quality descriptors"},
        "short_caption": {"kind": "caption", "hint": "short audio caption", "target_words": 4},
        "descriptive_caption": {"kind": "caption", "hint": "descriptive audio caption"},
    },
    "sfx": {
        "loudness": {"kind": "loudness"},
        "frequency_profile": {"kind": "bands"},
        "sounding_object": {"kind": "tags", "hint": "sound-source objects"},
        "sounding_action": {"kind": "tags", "hint": "sound-producing actions", "lemmatize": True},
        "temporal_descriptors": {"kind": "tags", "hint": "temporal pattern descriptors"},
        "plurality": {"kind": "plurality"},
        "sound_descriptors": {"kind": "tags", "hint": "sound-quality descriptors"},
        "perceived_audio_quality": {"kind": "tags", "hint": "acoustic-quality descriptors"},
        "short_caption": {"kind": "caption", "hint": "short audio caption", "target_words": 4},
        "descriptive_caption": {"kind": "caption", "hint": "descriptive audio caption"},
    },
}

# Prediction-side key spellings that mean a scored field.
KEY_ALIASES = {
    "perceived_quality": "perceived_audio_quality",
    "short_title": "short_caption",
    "descriptive_title": "descriptive_caption",
}

_EMPTY_STRINGS = {"", "n/a", "none", "unknown", "loudness n/a", "null", "nan"}
_TAG_AXES = frozenset({"fidelity", "space", "frequency balance", "drive", "noise"})
_LEMMA_EXCEPTIONS = {"gas": "gas", "glass": "glass", "grass": "grass", "bass": "bass",
                     "press": "press", "hiss": "hiss", "buzz": "buzz", "wings": "wing"}
_BPM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:bpm|beats per minute)?\s*$", re.I)


# ---------------------------------------------------------------------------
# Model output -> prediction row
# ---------------------------------------------------------------------------

def parse_inventory(text: str) -> list[dict[str, Any]]:
    """The Stage-1 numbered list (``1. <type> | <name> | <start>-<end>; ...``) -> sources."""
    out = []
    for line in (text or "").strip().splitlines():
        m = re.match(r"^\s*\d+[.)]\s*(speech|music|sound effects?|sfx)\s*\|\s*([^|]+?)\s*(?:\|\s*(.*))?$",
                     line.strip(), re.I)
        if not m:
            continue
        source_type = m.group(1).lower()
        source_type = "sound effect" if source_type.startswith(("sound", "sfx")) else source_type
        spans = []
        for a, b in re.findall(r"(\d+\.?\d*)\s*[-–to]+\s*(\d+\.?\d*)", m.group(3) or ""):
            a, b = float(a), float(b)
            if b > a:
                spans.append([max(0.0, a), min(10.0, b)])
        out.append({"type": source_type, "name": m.group(2).strip(), "spans": spans})
    return out


def merge_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The benchmark's counting rule: all speech is one source named "speech", all music one
    named "music", and sound effects sharing a name (case-insensitive) merge. Spans are unioned.
    Stage-2 cards are asked for, and scored, per merged source."""
    out = []
    for source in sources:
        source = {**source, "spans": [list(span) for span in source.get("spans") or []]}
        if source["type"] in ("speech", "music"):
            source["name"] = source["type"]
        out.append(source)
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if a["type"] != b["type"] or a["name"].lower() != b["name"].lower():
                    continue
                a["spans"] += b["spans"]
                del out[j]
                merged = True
                break
            if merged:
                break
    for source in out:
        source["spans"] = merge_spans(source["spans"])
    return out


def merge_spans(spans) -> list[list[float]]:
    """Union overlapping or touching spans."""
    out: list[list[float]] = []
    for start, end in sorted([float(a), float(b)] for a, b in spans):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def parse_card(text: str) -> dict[str, Any] | None:
    """The first JSON object in a model's answer, read ever more forgivingly: strict JSON,
    trailing commas dropped, a Python dict literal, then json_repair if it is installed."""
    blob = _first_object(text or "")
    if blob is None:
        return None
    for attempt in (lambda: json.loads(blob),
                    lambda: json.loads(re.sub(r",\s*([}\]])", r"\1", blob)),
                    lambda: ast.literal_eval(blob)):
        try:
            value = attempt()
        except Exception:
            continue
        if isinstance(value, dict):
            return _json_safe(value)
    try:
        import json_repair
        value = json_repair.loads(blob)
        if isinstance(value, dict) and value:
            return _json_safe(value)
    except Exception:
        pass
    return None


def _first_object(text: str) -> str | None:
    """The first balanced ``{...}``, string-aware; an unterminated one is returned as is."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def _json_safe(value):
    if value is Ellipsis:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def clip_to_prediction_row(clip: dict[str, Any]) -> dict[str, Any]:
    """A per-clip inference JSON (``eid``, ``predicted_sources``, ``cards`` of
    ``source_type`` / ``source_name`` / ``card``) -> a prediction row."""
    return {"eid": clip["eid"],
            "sources": clip.get("predicted_sources", []),
            "cards": [{"type": c["source_type"], "name": c["source_name"], "card": c["card"]}
                      for c in clip["cards"]]}


# ---------------------------------------------------------------------------
# Card -> scored values
# ---------------------------------------------------------------------------

def reduce_card(card: dict[str, Any] | None, modality: str) -> dict[str, Any]:
    """Reduce a raw card to the value each scored field is compared on; ``None`` means blank.

        loudness, plurality  {"label", "index"} on the ordinal scale
        frequency_profile    {band_hz: mean ordinal 0-6 over frames}
        median_f0_hz         {"hz": duration-weighted median, "semitones": 12*log2(hz)}
        tempo                {"label", "index", "bpm", "no_tempo"}
        tag fields           token list
        captions             string
    """
    out: dict[str, Any] = {}
    for field, spec in FIELDS[modality].items():
        raw = field_value(card, field, spec.get("path"))
        kind = spec["kind"]
        if kind == "loudness":
            out[field] = _ordinal(parse_loudness(raw).label, LOUDNESS_SCALE)
        elif kind == "plurality":
            label = None
            if not is_blank(raw, scale=PLURALITY_SCALE):
                token = _canon(str(raw))
                label = PLURALITY_ALIASES.get(token, token)
            out[field] = _ordinal(label, PLURALITY_SCALE)
        elif kind == "bands":
            out[field] = mean_band_ordinals(parse_bands(raw)) or None
        elif kind == "f0":
            hz = parse_f0(raw)
            out[field] = None if hz is None else {"hz": hz, "semitones": 12.0 * math.log2(hz)}
        elif kind == "tempo":
            value, label = parse_tempo(raw)
            out[field] = None if value is None and label is None else {
                "label": label, "index": scale_index(TEMPO_SCALE, label), "bpm": value,
                "no_tempo": value is None and label == "no tempo"}
        elif kind == "tags":
            out[field] = tokenize_tags(raw, lemmatize=spec.get("lemmatize", False)) or None
        elif kind == "caption":
            out[field] = None if is_blank(raw) else str(raw).strip()
    return out


def _ordinal(label: str | None, scale: list[str]) -> dict[str, Any] | None:
    index = scale_index(scale, label)
    return None if index is None else {"label": label, "index": index}


def field_value(card: dict[str, Any] | None, field: str, path: str | None = None) -> Any:
    """One field out of a card, tolerating key spelling drift ("perceived audio quality",
    "Music-Descriptors") and the aliases in ``KEY_ALIASES``."""
    if not isinstance(card, dict):
        return None
    value = card.get(field)
    if value is None:
        wanted = {_key_norm(field)} | {_key_norm(alias) for alias, canonical in KEY_ALIASES.items()
                                       if canonical == field}
        for key, item in card.items():
            if _key_norm(key) in wanted:
                value = item
                break
    if path and isinstance(value, dict):
        value = value.get(path)
    return value


def _key_norm(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def _canon(text: str) -> str:
    return " ".join(text.strip().lower().split())


def canon_label(text: Any) -> str:
    """Hyphens and underscores read as spaces, for scale matching only ("very-loud")."""
    return " ".join(re.sub(r"[-_]+", " ", str(text)).strip().lower().split())


def scale_index(scale: list[str], label: str | None) -> int | None:
    if not label:
        return None
    folded = [canon_label(entry) for entry in scale]
    key = canon_label(label)
    return folded.index(key) if key in folded else None


def is_blank(value: Any, scale: list[str] | None = None) -> bool:
    """No answer. A value on the field's own scale is an answer ("none" for plurality)."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if scale and isinstance(value, str) and _canon(value) in {_canon(s) for s in scale}:
        return False
    if isinstance(value, str):
        return _canon(value) in _EMPTY_STRINGS
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def parse_loudness(value: Any) -> Loudness:
    """Loudness cell -> (mean LUFS, modal label). Accepts ``[[-29.4, "Loud"], ...]``,
    ``[-29.4, "Loud"]``, ``"-37.7, Very quiet"`` and bare numbers; labels off the scale drop."""
    if is_blank(value):
        return Loudness(None, None)
    valid = {canon_label(s) for s in LOUDNESS_SCALE}
    items = value if isinstance(value, (list, tuple)) else [value]
    if (isinstance(value, (list, tuple)) and value
            and any(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
            and any(isinstance(x, str) for x in value)):
        items = [value]
    numbers, labels = [], []
    for item in items:
        number, label = None, None
        if isinstance(item, str):
            if "," in item:
                head, tail = item.rsplit(",", 1)
                try:
                    number = float(head.strip())
                except ValueError:
                    number = None
                label = tail.strip()
            else:
                label = item.strip()
        elif isinstance(item, (list, tuple)) and item:
            if isinstance(item[-1], str):
                label = item[-1].strip()
                numeric = [x for x in item[:-1] if isinstance(x, (int, float))]
            else:
                numeric = [x for x in item if isinstance(x, (int, float))]
            if numeric:
                number = float(numeric[-1])
        elif isinstance(item, (int, float)):
            number = float(item)
        if label is not None and canon_label(label) not in valid:
            label = None
        if number is not None and math.isfinite(number):
            numbers.append(number)
        if label:
            labels.append(label)
    mean = sum(numbers) / len(numbers) if numbers else None
    modal = Counter(labels).most_common(1)[0][0] if labels else None
    return Loudness(mean, modal)


def parse_bands(value: Any) -> list[dict[str, str]]:
    """Frequency profile -> ``[{band_hz: label}, ...]``, one dict per frame."""
    if is_blank(value):
        return []
    frames = value if isinstance(value, (list, tuple)) else [value]
    out = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        labels = frame.get("bands_label")
        if isinstance(labels, dict) and labels:
            out.append({str(k): _canon(str(v)) for k, v in labels.items()})
        elif labels is None and frame and all(isinstance(v, str) for v in frame.values()):
            out.append({str(k): _canon(str(v)) for k, v in frame.items()})
    return out


def mean_band_ordinals(frames: list[dict[str, str]]) -> dict[str, float]:
    """One mean ordinal (0-6) per band over all frames; a band no frame labels is left out."""
    out = {}
    for band in BANDS:
        values = [i for i in (scale_index(BAND_SCALE, frame.get(band)) for frame in frames)
                  if i is not None]
        if values:
            out[band] = sum(values) / len(values)
    return out


def parse_f0(value: Any) -> float | None:
    """``[[start, end, hz], ...]`` -> the duration-weighted median Hz."""
    if is_blank(value):
        return None
    entries = []
    for item in value if isinstance(value, (list, tuple)) else [value]:
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            if item > 0:
                entries.append((float(item), 1.0))
            continue
        if not isinstance(item, (list, tuple)):
            continue
        numeric = [x for x in item if isinstance(x, (int, float))]
        if len(numeric) >= 3:
            hz, weight = float(numeric[2]), max(float(numeric[1]) - float(numeric[0]), 1e-6)
        elif numeric:
            hz, weight = float(numeric[-1]), 1.0
        else:
            continue
        if hz > 0:
            entries.append((hz, weight))
    if not entries:
        return None
    entries.sort()
    half, run = sum(w for _, w in entries) / 2.0, 0.0
    for hz, weight in entries:
        run += weight
        if run >= half:
            return hz
    return entries[-1][0]


def parse_tempo(value: Any) -> tuple[float | None, str | None]:
    """``[111, "medium"]``, ``"100 BPM"``, ``"no tempo"`` -> (bpm, label)."""
    if is_blank(value):
        return None, None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), None
    if isinstance(value, str):
        bpm = _bpm(value)
        return (bpm, None) if bpm is not None else (None, _canon(value))
    if isinstance(value, (list, tuple)):
        number = next((b for b in (_bpm(x) for x in value) if b is not None), None)
        label = next((_canon(x) for x in value
                      if isinstance(x, str) and x.strip() and _bpm(x) is None), None)
        return number, label
    return None, None


def _bpm(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, str):
        m = _BPM_RE.match(x)
        return float(m.group(1)) if m else None
    return None


def tokenize_tags(value: Any, lemmatize: bool = False) -> list[str]:
    """Comma string or list -> lowercased, de-duplicated tokens; an axis heading
    ("Fidelity: clear") is stripped."""
    if is_blank(value):
        return []
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            parts.extend(str(item).split(",") if isinstance(item, str) else [str(item)])
    elif isinstance(value, dict):
        parts = [str(v) for v in value.values()]
    else:
        parts = [str(value)]
    out, seen = [], set()
    for part in parts:
        head, sep, tail = part.partition(":")
        if sep and tail.strip() and head.strip().lower() in _TAG_AXES:
            part = tail
        token = _canon(part)
        if not token or token in _EMPTY_STRINGS:
            continue
        if lemmatize:
            token = " ".join(simple_lemma(word) for word in token.split())
        if token not in seen:
            seen.add(token)
            out.append(token)
    return out


def simple_lemma(word: str) -> str:
    """Rule-based de-inflection for gerunds and plurals (``sounding_action``)."""
    lowered = word.lower()
    if len(lowered) <= 3 or lowered in _LEMMA_EXCEPTIONS:
        return _LEMMA_EXCEPTIONS.get(lowered, lowered)
    if lowered.endswith("ing") and len(lowered) > 5:
        stem = lowered[:-3]
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
            return stem[:-1]
        if stem.endswith(("l", "r", "v", "c", "g", "s", "t", "z", "u")) and not stem.endswith("e"):
            return stem + "e" if stem.endswith(("l", "v", "c", "u")) else stem
        return stem
    if lowered.endswith("ies") and len(lowered) > 4:
        return lowered[:-3] + "y"
    if lowered.endswith(("ches", "shes", "sses", "xes", "zes")):
        return lowered[:-2]
    if lowered.endswith("s") and not lowered.endswith(("ss", "us", "is")):
        return lowered[:-1]
    if lowered.endswith("ed") and len(lowered) > 4:
        stem = lowered[:-2]
        if len(stem) > 2 and stem[-1] == stem[-2]:
            return stem[:-1]
        return stem
    return lowered


def canonical_class(value: Any) -> str:
    """SFX class spelling that joins "Cat Domestic", "cat-domestic" and "cat_domestic"."""
    return re.sub(r"[\s_-]+", "-", str(value).strip()).lower()

"""Scoring: the judge, Stage 1 (what is in the scene, and when) and Stage 2 (the cards).

Stage 1. Predicted sources are paired with reference sources first: speech and music by
identity, sound effects through the judge (every predicted x reference pair is rated for
similarity, then the best one-to-one assignment is kept where it reaches 0.5). The pairs are then
timed as sound event detection -- segment-based F1 on 100 ms segments and event-based F1 with a
1.0 s onset/offset collar. A reference source nothing matched is all misses; a predicted source
that matched nothing is all false alarms, counted against the modality its own name implies.

Stage 2. Each matched source's predicted card is scored against the reference card, field by
field, on a 0-1 scale:

    loudness, plurality  1 - |ordinal distance| / (scale length - 1)
    frequency_profile    mean over bands of 1 - |mean ordinal distance| / 6
    median_f0_hz         max(0, 1 - |semitone error| / (4 * sigma))
    tempo                tempo label as an ordinal; a "no tempo" reference scores 1 only when the
                         prediction also says no tempo
    tag fields           judge-itemized soft F1 (a partial match is worth 0.5)
    captions             judge rubric mean (correctness, relevance, completeness, clarity), times
                         a length factor for short_caption

A blank or unusable prediction scores 0; a field the reference leaves blank is not scored. The
card score is the mean over a card's scored fields, averaged over matched cards.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import prompts
from .format import (BAND_SCALE, BANDS, F0_SIGMA, FIELDS, LOUDNESS_SCALE, MODALITIES,
                     PLURALITY_SCALE, SOURCE_TYPES, TEMPO_SCALE, canonical_class, merge_sources,
                     parse_card, reduce_card)

RESOLUTION = 0.1           # segment-based: 100 ms segments
COLLAR = 1.0               # event-based: onset and offset collar, seconds
MIN_SIMILARITY = 0.5       # a judged pair below this stays unmatched
PARTIAL_CREDIT = 0.5       # tag fields: a broader/narrower/related tag is half a match
CAPTION_CRITERIA = ("correctness", "relevance", "completeness", "clarity")
RESERVED = ("speech", "music")
_EPS = 1e-9


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class VLLMBackend:
    """A judge model loaded in-process with vLLM (greedy decoding)."""

    def __init__(self, model: str, max_model_len: int | None = None, **llm_kwargs: Any) -> None:
        from vllm import LLM
        if max_model_len is None:
            # Judge prompts are short. Cap the KV cache at 16k, but never above the model's own
            # context (Llama-3-8B has 8k and vLLM refuses to start when asked for more).
            max_model_len = 16384
            try:
                from transformers import AutoConfig
                native = getattr(AutoConfig.from_pretrained(model), "max_position_embeddings", None)
                if native:
                    max_model_len = min(max_model_len, int(native))
            except Exception:
                pass
        self.model = model
        self.engine = LLM(model=model, max_model_len=max_model_len, **llm_kwargs)

    def generate(self, batch: list[list[dict]], max_tokens: int) -> list[str]:
        from vllm import SamplingParams
        outputs = self.engine.chat(batch, SamplingParams(temperature=0.0, max_tokens=max_tokens))
        return [output.outputs[0].text for output in outputs]


class OpenAIBackend:
    """Any OpenAI-compatible chat-completions endpoint: OpenAI, Gemini
    (https://generativelanguage.googleapis.com/v1beta/openai), a `vllm serve` server, etc."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", workers: int = 8,
                 timeout: float = 120.0, attempts: int = 4) -> None:
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = os.environ.get(api_key_env, "")
        self.workers = workers
        self.timeout = timeout
        self.attempts = attempts

    def _complete(self, messages: list[dict], max_tokens: int) -> str:
        payload = json.dumps({"model": self.model, "messages": messages,
                              "max_tokens": max_tokens, "temperature": 0.0}).encode()
        request = urllib.request.Request(self.url, data=payload, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        for attempt in range(self.attempts):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())["choices"][0]["message"]["content"] or ""
            except (urllib.error.URLError, TimeoutError) as exc:
                retryable = not isinstance(exc, urllib.error.HTTPError) or exc.code in (429, 500, 502, 503, 504)
                if not retryable or attempt == self.attempts - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def generate(self, batch: list[list[dict]], max_tokens: int) -> list[str]:
        with concurrent.futures.ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda messages: self._complete(messages, max_tokens), batch))


class Judge:
    """Asks a backend for JSON verdicts. An answer that does not parse is asked once more with a
    larger token budget; one that still fails comes back as None. Verdicts are cached by prompt,
    so a repeated question is asked once."""

    def __init__(self, backend, max_tokens: int = 512, retry_max_tokens: int = 1024) -> None:
        self.backend = backend
        self.max_tokens = max_tokens
        self.retry_max_tokens = retry_max_tokens
        self.cache: dict[str, dict | None] = {}
        self.n_calls = 0
        self.n_parse_failures = 0

    def run(self, batch: list[list[dict]]) -> list[dict | None]:
        keys = [json.dumps(messages, sort_keys=True) for messages in batch]
        todo = list({key: messages for key, messages in zip(keys, batch)
                     if key not in self.cache}.items())
        if todo:
            self.n_calls += len(todo)
            parsed = [parse_card(text) for text in
                      self.backend.generate([messages for _, messages in todo], self.max_tokens)]
            retry = [i for i, value in enumerate(parsed) if value is None]
            if retry:
                answers = self.backend.generate([todo[i][1] for i in retry], self.retry_max_tokens)
                for i, text in zip(retry, answers):
                    parsed[i] = parse_card(text)
            self.n_parse_failures += sum(value is None for value in parsed)
            self.cache.update({key: value for (key, _), value in zip(todo, parsed)})
        return [self.cache[key] for key in keys]

    @property
    def stats(self) -> dict[str, Any]:
        return {"judge_model": getattr(self.backend, "model", None),
                "judge_backend": type(self.backend).__name__,
                "judge_calls": self.n_calls, "judge_parse_failures": self.n_parse_failures}


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def source_label(text: str) -> str:
    """speech and music keep their spelling; every other label is a canonical SFX class."""
    lowered = str(text).strip().lower()
    return lowered if lowered in RESERVED else canonical_class(text)


def timeline(sources: Iterable[dict], predicted: bool) -> dict[str, list[tuple[float, float]]]:
    """Sources -> {label: merged spans}. Reference sources are labelled by ``label``, predicted
    ones by ``name``; speech and music by their type. A predicted source with no time span is
    not a Stage-1 answer and is ignored, so it can neither match nor bring its card in."""
    out: dict[str, list[tuple[float, float]]] = {}
    for source in sources:
        modality = SOURCE_TYPES.get(str(source.get("type", "")).strip().lower())
        if modality is None or (predicted and not source.get("spans")):
            continue
        name = modality if modality in RESERVED else source.get("name" if predicted else "label")
        if not name or not str(name).strip():
            continue
        spans = out.setdefault(source_label(name), [])
        for span in source.get("spans") or []:
            start, end = float(span[0]), float(span[1])
            spans.append((min(start, end), max(start, end)))
    merged = {}
    for label, spans in out.items():
        joined: list[list[float]] = []
        for start, end in sorted(spans):
            if joined and start <= joined[-1][1]:
                joined[-1][1] = max(joined[-1][1], end)
            else:
                joined.append([start, end])
        merged[label] = [(start, end) for start, end in joined]
    return merged


def sfx_labels(tl: dict) -> list[str]:
    return sorted(label for label in tl if label not in RESERVED)


def match_sources(ref: dict, pred: dict, similarity: dict[tuple[str, str], float]):
    """(pairs, extra, missed): pairs are (predicted, reference); extra are predicted labels that
    matched nothing, missed are reference labels nothing matched."""
    pairs, extra, missed = [], [], []
    for label in RESERVED:
        if label in ref and label in pred:
            pairs.append((label, label))
        elif label in ref:
            missed.append(label)
        elif label in pred:
            extra.append(label)
    pred_sfx, ref_sfx = sfx_labels(pred), sfx_labels(ref)
    used_p, used_r = set(), set()
    if pred_sfx and ref_sfx:
        matrix = np.array([[similarity[(p, r)] for r in ref_sfx] for p in pred_sfx])
        for row, col in zip(*linear_sum_assignment(matrix, maximize=True)):
            if matrix[row][col] >= MIN_SIMILARITY:
                pairs.append((pred_sfx[row], ref_sfx[col]))
                used_p.add(pred_sfx[row])
                used_r.add(ref_sfx[col])
    extra += [p for p in pred_sfx if p not in used_p]
    missed += [r for r in ref_sfx if r not in used_r]
    return pairs, extra, missed


def active_segments(spans: list[tuple[float, float]], duration: float) -> set[int]:
    """Indices of the 100 ms segments any span overlaps (touching a boundary is not overlap)."""
    n_segments = max(1, round(duration / RESOLUTION))
    active: set[int] = set()
    for start, end in spans:
        if end <= start:
            continue
        first = max(0, math.floor(start / RESOLUTION + _EPS))
        last = min(n_segments - 1, math.ceil(end / RESOLUTION - _EPS) - 1)
        if last >= first:
            active.update(range(first, last + 1))
    return active


def match_events(ref: list[tuple[float, float]], pred: list[tuple[float, float]]) -> int:
    """Greedy one-to-one event matching under the collar, tightest pairs first. Returns hits."""
    candidates = sorted((abs(ps - rs) + abs(pe - re), i, j)
                        for i, (ps, pe) in enumerate(pred) for j, (rs, re) in enumerate(ref)
                        if abs(ps - rs) <= COLLAR and abs(pe - re) <= COLLAR)
    used_p, used_r, hits = set(), set(), 0
    for _, i, j in candidates:
        if i not in used_p and j not in used_r:
            used_p.add(i)
            used_r.add(j)
            hits += 1
    return hits


def modality_of(label: str) -> str:
    return label if label in RESERVED else "sfx"


def stage1_counts(ref: dict, pred: dict, pairs, extra, missed, duration: float) -> dict:
    """{"segment"|"event": {modality: [tp, fp, fn]}} for one scene. A pair counts toward the
    reference label's modality, an unmatched prediction toward its own label's."""
    counts = {basis: {m: [0, 0, 0] for m in MODALITIES} for basis in ("segment", "event")}

    def add(basis, label, tp, fp, fn):
        bucket = counts[basis][modality_of(label)]
        bucket[0] += tp
        bucket[1] += fp
        bucket[2] += fn

    for p, r in pairs:
        ref_active, pred_active = active_segments(ref[r], duration), active_segments(pred[p], duration)
        add("segment", r, len(ref_active & pred_active), len(pred_active - ref_active),
            len(ref_active - pred_active))
        hits = match_events(ref[r], pred[p])
        add("event", r, hits, len(pred[p]) - hits, len(ref[r]) - hits)
    for r in missed:
        add("segment", r, 0, 0, len(active_segments(ref[r], duration)))
        add("event", r, 0, 0, len(ref[r]))
    for p in extra:
        add("segment", p, 0, len(active_segments(pred[p], duration)), 0)
        add("event", p, 0, len(pred[p]), 0)
    return counts


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _similarity(verdict: dict | None) -> float:
    try:
        return max(0.0, min(1.0, float(verdict["similarity"])))
    except (KeyError, TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def _ordinal_score(ref: dict | None, pred: dict | None, scale: list[str]) -> float | None:
    if ref is None:
        return None
    if pred is None:
        return 0.0
    return 1.0 - abs(pred["index"] - ref["index"]) / (len(scale) - 1)


def score_field(kind: str, ref: Any, pred: Any) -> float | None:
    """One deterministic field. None when the reference leaves the field blank (not scored)."""
    if kind == "tempo":
        if ref is None:
            return None
        if ref["no_tempo"]:
            return 1.0 if pred is not None and pred["no_tempo"] else 0.0
        if ref["index"] is None:
            return None
        if pred is None or pred["index"] is None:
            return 0.0
        return 1.0 - abs(pred["index"] - ref["index"]) / (len(TEMPO_SCALE) - 1)
    if kind == "loudness":
        return _ordinal_score(ref, pred, LOUDNESS_SCALE)
    if kind == "plurality":
        return _ordinal_score(ref, pred, PLURALITY_SCALE)
    if ref is None:
        return None
    if pred is None:
        return 0.0
    if kind == "bands":
        shared = [band for band in BANDS if band in ref and band in pred]
        if not shared:
            return 0.0
        span = len(BAND_SCALE) - 1
        return float(np.mean([1.0 - abs(pred[band] - ref[band]) / span for band in shared]))
    if kind == "f0":
        return max(0.0, 1.0 - abs(pred["semitones"] - ref["semitones"]) / (4.0 * F0_SIGMA))
    raise ValueError(f"not a deterministic field kind: {kind}")


def _count(verdict: dict, key: str, limit: int) -> int | None:
    try:
        return max(0, min(limit, int(round(float(verdict[key])))))
    except (KeyError, TypeError, ValueError):
        return None


def tag_score(verdict: dict | None, n_ref: int, n_pred: int) -> float:
    """Soft F1 from the judge's alignment counts; an unusable verdict scores 0."""
    if not isinstance(verdict, dict):
        return 0.0
    ref_exact, pred_exact = _count(verdict, "ref_exact", n_ref), _count(verdict, "pred_exact", n_pred)
    if ref_exact is None or pred_exact is None:
        return 0.0
    ref_partial = _count(verdict, "ref_partial", n_ref - ref_exact) or 0
    pred_partial = _count(verdict, "pred_partial", n_pred - pred_exact) or 0
    recall = (ref_exact + PARTIAL_CREDIT * ref_partial) / n_ref
    precision = (pred_exact + PARTIAL_CREDIT * pred_partial) / n_pred
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def caption_score(verdict: dict | None, prediction: str, target_words: int | None) -> float:
    """Rubric mean, times ``max(0, 1 - |words - target| / target)`` when a length is set."""
    try:
        values = [max(0.0, min(1.0, float(verdict[name]))) for name in CAPTION_CRITERIA]
    except (KeyError, TypeError, ValueError):
        return 0.0
    score = float(np.mean(values))
    if target_words:
        n_words = len(prediction.split())
        score *= max(0.0, 1.0 - abs(n_words - target_words) / target_words)
    return score


def _reference_cards(scene: dict) -> dict[tuple[str, str], dict]:
    """{(modality, key): reduced reference card}; key is the modality or the canonical class."""
    cards = scene.get("cards") or {}
    if isinstance(cards, str):
        cards = json.loads(cards)
    out = {}
    for modality in RESERVED:
        if cards.get(modality):
            out[(modality, modality)] = cards[modality]
    for card in cards.get("sfx") or []:
        out[("sfx", canonical_class(card["class"]))] = card
    return out


def _predicted_cards(prediction: dict) -> dict[tuple[str, str], tuple[str, dict | None]]:
    """{(modality, key): (name, raw card)}. A card may be a dict, the model's raw text, or None."""
    out = {}
    for entry in prediction.get("cards") or []:
        modality = SOURCE_TYPES.get(str(entry.get("type", "")).strip().lower())
        if modality is None:
            continue
        name = modality if modality in RESERVED else str(entry.get("name") or "")
        card = entry.get("card")
        if isinstance(card, str):
            card = parse_card(card)
        out[(modality, source_label(name))] = (name, card if isinstance(card, dict) else None)
    return out


# ---------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------

def evaluate(scenes: Iterable[dict], predictions: Iterable[dict], judge: Judge
             ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score predictions against the dataset. Returns (results, per-card records).

    A scene with no prediction row is scored as an empty prediction: every reference source is
    missed. A prediction row for a scene outside the dataset is ignored.
    """
    by_eid = {row["eid"]: row for row in predictions}
    scenes = list(scenes)
    items = []
    for scene in scenes:
        prediction = by_eid.get(scene["eid"]) or {}
        merged = merge_sources([{**s, "type": str(s.get("type", "")).strip().lower(),
                                 "name": str(s.get("name") or "")}
                                for s in prediction.get("sources") or []])
        items.append((scene, prediction, timeline(scene.get("sources") or [], predicted=False),
                      timeline(merged, predicted=True)))

    # Stage 1 -- rate every predicted x reference SFX label pair in one judge batch.
    pair_keys = sorted({(p, r) for _, _, ref, pred in items
                        for p in sfx_labels(pred) for r in sfx_labels(ref)})
    verdicts = judge.run([prompts.judge_messages(prompts.PAIR_SIMILARITY, reference=r, prediction=p)
                          for p, r in pair_keys])
    similarity = {key: _similarity(verdict) for key, verdict in zip(pair_keys, verdicts)}

    totals = {basis: {m: [0, 0, 0] for m in MODALITIES} for basis in ("segment", "event")}
    matched_cards = []                     # (scene eid, modality, reference key, predicted name, ref, raw pred)
    n_reference = dict.fromkeys(MODALITIES, 0)
    n_predicted = dict.fromkeys(MODALITIES, 0)
    for scene, prediction, ref_tl, pred_tl in items:
        pairs, extra, missed = match_sources(ref_tl, pred_tl, similarity)
        counts = stage1_counts(ref_tl, pred_tl, pairs, extra, missed, float(scene.get("duration") or 10.0))
        for basis, per_modality in counts.items():
            for modality, (tp, fp, fn) in per_modality.items():
                bucket = totals[basis][modality]
                bucket[0] += tp
                bucket[1] += fp
                bucket[2] += fn

        ref_cards, pred_cards = _reference_cards(scene), _predicted_cards(prediction)
        for modality, _ in ref_cards:
            n_reference[modality] += 1
        for modality, _ in pred_cards:
            n_predicted[modality] += 1
        # Speech and music join on the scene; sound effects through the Stage-1 pairs.
        joins = [(m, m) for m in RESERVED] + [(p, r) for p, r in pairs if r not in RESERVED]
        for p, r in joins:
            modality = modality_of(r)
            ref_card, pred_entry = ref_cards.get((modality, r)), pred_cards.get((modality, p))
            if ref_card is None or pred_entry is None:
                continue
            matched_cards.append((scene["eid"], modality, r, pred_entry[0], ref_card, pred_entry[1]))

    # Stage 2 -- deterministic fields now, judged fields in one batch.
    records, requests = [], []
    for eid, modality, ref_key, pred_name, ref_card, raw_pred in matched_cards:
        pred_card = reduce_card(raw_pred, modality)
        scores = {}
        for field, spec in FIELDS[modality].items():
            ref, pred = ref_card.get(field), pred_card.get(field)
            if spec["kind"] in ("tags", "caption"):
                if ref is None:
                    scores[field] = None
                elif pred is None:
                    scores[field] = 0.0
                else:
                    prompt = prompts.TAG_SET if spec["kind"] == "tags" else prompts.CAPTION_RUBRIC
                    requests.append((len(records), field, spec, ref, pred,
                                     prompts.judge_messages(prompt, hint=spec["hint"],
                                                            reference=ref, prediction=pred)))
            else:
                scores[field] = score_field(spec["kind"], ref, pred)
        records.append({"eid": eid, "modality": modality, "reference": ref_key,
                        "prediction": pred_name, "scores": scores})

    for (index, field, spec, ref, pred, _), verdict in zip(
            requests, judge.run([request[-1] for request in requests])):
        records[index]["scores"][field] = (
            tag_score(verdict, len(ref), len(pred)) if spec["kind"] == "tags"
            else caption_score(verdict, pred, spec.get("target_words")))

    stage2 = {}
    for modality in MODALITIES:
        cards = [r["scores"] for r in records if r["modality"] == modality]
        card_means = [float(np.mean(scored)) for scored in
                      ([v for v in scores.values() if v is not None] for scores in cards) if scored]
        fields = {}
        for field in FIELDS[modality]:
            values = [scores[field] for scores in cards if scores.get(field) is not None]
            fields[field] = {"score": float(np.mean(values)) if values else None, "n": len(values)}
        stage2[modality] = {"card_score": float(np.mean(card_means)) if card_means else None,
                            "n_reference_cards": n_reference[modality],
                            "n_predicted_cards": n_predicted[modality],
                            "n_matched_cards": len(cards), "fields": fields}

    stage1 = {}
    for basis, per_modality in totals.items():
        stage1[basis] = {modality: prf(*counts) for modality, counts in per_modality.items()}
        stage1[basis]["all"] = prf(*(sum(c[i] for c in per_modality.values()) for i in range(3)))

    results = {"n_scenes": len(scenes),
               "n_scenes_without_prediction": sum(1 for s in scenes if s["eid"] not in by_eid),
               "stage1": stage1, "stage2": stage2,
               "judge": judge.stats, "prompt_version": prompts.VERSION,
               "prompt_fingerprint": prompts.fingerprint()}
    return results, records

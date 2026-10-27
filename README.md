# MixCardBench

Prompts and scoring for **MixCardBench** from the paper **MixCards: Evaluating Structured Descriptions of Multi-Source Sound Scenes**.

A model hears a 10-second scene mixing speech, music and sound effects, and answers in two stages:

- **Stage 1 (sound event detection):** list every distinct sound source and the time spans it is audible.
- **Stage 2 (MixCards):** for each source, fill a card of fixed fields (loudness, a ten-band frequency
  profile, tags, captions, and per-modality fields such as median F0 or tempo).

This package gives you the prompts and scores the answers against the reference.

## Install

Python 3.12 or later.

**With the in-process vLLM judge** (default; how the reported numbers were produced). Needs an
NVIDIA GPU and driver >= 580: vLLM 0.24.0 installs torch 2.11 built for CUDA 13.0, so no system
CUDA toolkit is required.

```bash
conda env create -f environment.yml        # python 3.12 + requirements-vllm.txt + this package
conda activate mixcards-eval
```

or with any Python 3.12 environment:

```bash
pip install -r requirements-vllm.txt
pip install -e .
```

**With an API judge only** (OpenAI, Gemini, or a `vllm serve` endpoint elsewhere). No GPU:

```bash
pip install -r requirements.txt
pip install -e .
```

## 1. Run your model

```python
from mixcards_eval import merge_sources, parse_card, parse_inventory, stage1_messages, stage2_messages

answer = my_model(audio, stage1_messages())                  # attach the scene audio your model's way
sources = merge_sources(parse_inventory(answer))              # all speech -> one source, all music -> one
cards = [{"type": s["type"], "name": s["name"],
          "card": parse_card(my_model(audio, stage2_messages(s["type"], s["name"])))}
         for s in sources]
row = {"eid": eid, "sources": sources, "cards": cards}        # one line of predictions.jsonl
```

Send the full scene with every call. The messages carry no audio; place it the way your model
expects.

### Prediction row

```json
{"eid": "--BfvyPmVMo_20000",
 "sources": [{"type": "speech" | "music" | "sound effect", "name": "machine operation",
              "spans": [[0.0, 7.1]]}],
 "cards": [{"type": "sound effect", "name": "machine operation", "card": {...} | null}]}
```

`card` is the model's card as a dict, its raw text (parsed for you), or `null` if it did not parse.
A null card is scored as blank, not dropped. A source with no time span is ignored.

## 2. Score

```bash
mixcards-eval --dataset <hf id | local dataset dir | rows.jsonl> --split final_4884 \
    --predictions predictions.jsonl --out results.json [--per-card per_card.jsonl]
```

The judge defaults to `meta-llama/Meta-Llama-3-8B-Instruct` on vLLM. Any OpenAI-compatible endpoint should work (need tests):

```bash
--judge-backend openai --judge gpt-4o-mini                                      # OPENAI_API_KEY
```

### Dataset row

```json
{"eid": "--BfvyPmVMo_20000", "audio": <audio>, "duration": 10.0,
 "sources": [{"type": "speech" | "music" | "sfx", "label": "Hand", "spans": [[3.007, 3.657]]}],
 "cards": {"speech": {<field>: <value>} | null, "music": {...} | null,
           "sfx": [{"class": "Hand", <field>: <value>}]}}
```

Reference cards hold the values the scorer compares (`mixcards_eval.format.reduce_card`).

## What is scored

**Stage 1.** Speech and music pair by identity. Each predicted sound effect is rated against each
reference class by the judge, and the best one-to-one assignment is kept where the similarity
reaches 0.5. Matched sources are timed as sound event detection: segment-based F1 over 100 ms
segments, and event-based F1 with a 1.0 s onset/offset collar. An unmatched reference source is
all misses; an unmatched prediction is all false alarms. F1 is reported per modality and overall.

**Stage 2.** Each matched card is scored field by field, 0 to 1:

| field | score |
|---|---|
| loudness, plurality | 1 - ordinal distance / (scale length - 1) |
| frequency_profile | mean over bands of 1 - ordinal distance / 6 |
| median_f0_hz | max(0, 1 - semitone error / (4 x 5.19)) |
| tempo | tempo label as an ordinal (very slow ... very fast) |
| tag fields | judge-itemized soft F1; a broader, narrower or related tag counts half |
| captions | judge rubric mean (correctness, relevance, completeness, clarity); short_caption also scaled by a 4-word length target |

A blank or unusable answer scores 0; a field the reference leaves blank is not scored. The card
score is the mean of a card's scored fields, averaged over matched cards.

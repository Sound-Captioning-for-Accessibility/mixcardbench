"""mixcards-eval: score a predictions file against the MixCards dataset.

    mixcards-eval --dataset <hf id | local dir | file.jsonl> [--split final_4884] \\
        --predictions predictions.jsonl --out results.json [--per-card per_card.jsonl] \\
        [--judge meta-llama/Meta-Llama-3-8B-Instruct]

The judge runs in-process with vLLM by default. Any OpenAI-compatible endpoint works instead:

    --judge-backend openai --judge gpt-4o-mini                              (OPENAI_API_KEY)
    --judge-backend openai --judge gemini-2.0-flash \\
        --base-url https://generativelanguage.googleapis.com/v1beta/openai \\
        --api-key-env GEMINI_API_KEY
    --judge-backend openai --judge meta-llama/Meta-Llama-3-8B-Instruct \\
        --base-url http://localhost:8000/v1                                  (vllm serve)
"""

from __future__ import annotations

import argparse
import json
import pathlib

from .evaluate import Judge, OpenAIBackend, VLLMBackend, evaluate

DEFAULT_JUDGE = "meta-llama/Meta-Llama-3-8B-Instruct"


def load_scenes(dataset: str, split: str):
    """Dataset rows without decoding audio: a local .json/.jsonl file, a local dataset dir or a
    Hub id."""
    import datasets
    if dataset.endswith((".json", ".jsonl")):
        ds = datasets.load_dataset("json", data_files=dataset, split="train")
    elif pathlib.Path(dataset).is_dir() and (pathlib.Path(dataset) / "dataset_dict.json").exists():
        ds = datasets.load_from_disk(dataset)[split]
    else:
        ds = datasets.load_dataset(dataset, split=split)
    if "audio" in ds.column_names:
        ds = ds.remove_columns("audio")
    return list(ds)


def load_predictions(path: str) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="HF dataset id, local dataset dir, or .jsonl")
    parser.add_argument("--split", default="final_4884", help="final_4884 or validated_235")
    parser.add_argument("--predictions", required=True, help="predictions JSONL, one row per scene")
    parser.add_argument("--out", required=True, help="results JSON")
    parser.add_argument("--per-card", help="optional JSONL of per-card field scores")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help=f"judge model (default {DEFAULT_JUDGE})")
    parser.add_argument("--judge-backend", default="vllm", choices=("vllm", "openai"))
    parser.add_argument("--base-url", default="https://api.openai.com/v1",
                        help="openai backend: endpoint base URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="openai backend: environment variable holding the API key")
    parser.add_argument("--workers", type=int, default=8, help="openai backend: parallel requests")
    args = parser.parse_args(argv)

    scenes = load_scenes(args.dataset, args.split)
    predictions = load_predictions(args.predictions)
    print(f"{len(scenes)} scenes, {len(predictions)} prediction rows", flush=True)

    if args.judge_backend == "vllm":
        backend = VLLMBackend(args.judge)
    else:
        backend = OpenAIBackend(args.judge, base_url=args.base_url, api_key_env=args.api_key_env,
                                workers=args.workers)
    results, records = evaluate(scenes, predictions, Judge(backend))
    results["dataset"], results["split"], results["predictions"] = args.dataset, args.split, args.predictions

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    if args.per_card:
        with open(args.per_card, "w") as handle:
            handle.writelines(json.dumps(record) + "\n" for record in records)

    for basis in ("segment", "event"):
        row = results["stage1"][basis]
        print(f"Stage 1 {basis:7s} F1  " + "  ".join(
            f"{m} {row[m]['f1']:.3f}" for m in ("speech", "music", "sfx", "all")))
    print("Stage 2 card score  " + "  ".join(
        f"{m} {results['stage2'][m]['card_score'] or 0:.3f} ({results['stage2'][m]['n_matched_cards']} cards)"
        for m in ("speech", "music", "sfx")))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

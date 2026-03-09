"""Analyze tokenizer vocabulary differences between RoBERTa-base and SecureBERT.

For each MySQL keyword and function, counts the number of subword pieces each
tokenizer produces, using a leading space to replicate mid-sentence BPE context.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from transformers import RobertaTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.constants import mysql_functions, mysql_keywords

logger = logging.getLogger(__name__)

SQL_KEYWORDS = sorted(mysql_keywords | mysql_functions)

MODELS = {
    "roberta-base": "roberta-base",
    "securebert": "ehsanaghaei/SecureBERT",
}


def analyze_keywords(tokenizers: dict, keywords: list[str]) -> pd.DataFrame:
    """Count subword pieces each tokenizer produces per SQL keyword.

    Uses a leading space before each keyword to replicate mid-sentence BPE context,
    which is how keywords appear inside a SQL query string.
    """
    rows = []
    for kw in keywords:
        mid_sentence = " " + kw  # leading space → mid-sentence BPE merges
        row = {"keyword": kw}
        for model_name, tok in tokenizers.items():
            pieces = tok.tokenize(mid_sentence)
            # Strip the leading Ġ/Ċ marker from the first piece for display clarity
            display = [p.lstrip("Ġ") if i == 0 else p for i, p in enumerate(pieces)]
            row[f"n_tokens_{model_name}"] = len(pieces)
            row[f"pieces_{model_name}"] = "|".join(display)
        rows.append(row)
    return pd.DataFrame(rows)


def print_sample_tokenizations(
    tokenizers: dict, df: pd.DataFrame, output_dir: Path, n: int = 10
) -> None:
    normal = df[df["label"] == 0].sample(
        n=min(n, (df["label"] == 0).sum()), random_state=2
    )
    attack = df[df["label"] == 1].sample(
        n=min(n, (df["label"] == 1).sum()), random_state=2
    )

    lines = []
    for label_str, subset in [("NORMAL", normal), ("ATTACK", attack)]:
        header = f"\n{'='*60} {label_str} {'='*60}"
        print(header)
        lines.append(header)
        for _, row in subset.iterrows():
            user_input = row["user_inputs"]
            entry = f"\nInput: {user_input}"
            print(entry)
            lines.append(entry)
            for model_name, tok in tokenizers.items():
                pieces = tok.tokenize(user_input)
                line = f"  [{model_name}] ({len(pieces)} tokens): {' | '.join(pieces)}"
                print(line)
                lines.append(line)

    out_path = output_dir / "sample_tokenizations.txt"
    out_path.write_text("\n".join(lines))
    print(f"\nSaved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tokenizer differences between RoBERTa-base and SecureBERT for SQL keywords"
    )
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        default=None,
        help="Dataset label and CSV path (repeatable, e.g. --dataset A data/A.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/experiments/h3_tokenizer",
        help="Output directory for results",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizers ...")
    tokenizers = {
        name: RobertaTokenizerFast.from_pretrained(hf_id)
        for name, hf_id in MODELS.items()
    }

    print("\nAnalyzing SQL keyword fragmentation ...")
    kw_df = analyze_keywords(tokenizers, SQL_KEYWORDS)
    kw_path = output_dir / "keyword_fragmentation.csv"
    kw_df[["keyword", "pieces_roberta-base", "pieces_securebert"]].to_csv(
        kw_path, index=False
    )
    print(kw_df.to_string(index=False))
    print(f"\nSaved {kw_path}")

    print("\nAverage subword pieces per keyword:")
    for model_name in MODELS:
        avg = kw_df[f"n_tokens_{model_name}"].mean()
        print(f"  {model_name}: {avg:.3f}")

    if args.dataset:
        frames = []
        for label, path in args.dataset:
            print(f"\nLoading dataset {label} from {path} ...")
            frame = pd.read_csv(
                path, usecols=["user_inputs", "label", "split"], low_memory=False
            )
            frame["dataset"] = label
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
        df = df[df["split"] == "test"]
        print_sample_tokenizations(tokenizers, df, output_dir)


if __name__ == "__main__":
    main()

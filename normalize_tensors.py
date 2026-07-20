import json
import shutil
from pathlib import Path


def normalize_merges(tokenizer_path: str, output_path: str = None, backup: bool = True):
    """
    Normalize the `merges` field in a tokenizer.json file.

    Newer tokenizer.json files store merges as pairs, e.g. ["t", "h"].
    Older consumers (like some transformers.js BPE implementations)
    expect merges as space-joined strings, e.g. "t h".

    Converts pair-format merges to string format; leaves string-format
    merges untouched. Only writes/backs up if a change is actually made.
    """
    tokenizer_path = Path(tokenizer_path)
    output_path = Path(output_path) if output_path else tokenizer_path

    with open(tokenizer_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = data.get("model", {})
    merges = model.get("merges")

    if merges is None:
        print("No 'merges' field found under data['model']['merges']. Nothing to do.")
        return

    if not merges:
        print("'merges' is empty. Nothing to do.")
        return

    sample = merges[0]

    if isinstance(sample, str):
        print("Merges are already in string format. No changes needed.")
        return

    if not isinstance(sample, list):
        raise TypeError(f"Unrecognized merge entry type: {type(sample)}")

    normalized = []
    for i, pair in enumerate(merges):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(
                f"Unexpected merge entry at index {i}: {pair!r} "
                f"(expected a 2-element list)"
            )
        normalized.append(" ".join(pair))

    # Only back up right before we actually overwrite the original file
    if backup and output_path == tokenizer_path:
        backup_path = tokenizer_path.with_suffix(tokenizer_path.suffix + ".bak")
        shutil.copy2(tokenizer_path, backup_path)
        print(f"Backup written to {backup_path}")

    model["merges"] = normalized
    data["model"] = model

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(normalized)} merges from pair format to string format.")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Normalize tokenizer.json merges to space-joined string format."
    )
    parser.add_argument("tokenizer_path", help="Path to tokenizer.json")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output path (defaults to overwriting input, with a .bak backup)"
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Skip creating a .bak backup when overwriting in place"
    )
    args = parser.parse_args()

    normalize_merges(args.tokenizer_path, args.output, backup=not args.no_backup)
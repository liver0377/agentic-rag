"""Export data from Langfuse Dataset to JSON file.

This script exports evaluation data from a Langfuse Dataset, including:
- query, contexts, answer (from dataset items)
- pass_rate (from dataset item scores)

Usage:
    python scripts/export_from_langfuse_dataset.py --dataset my-eval-dataset
    python scripts/export_from_langfuse_dataset.py --dataset my-dataset -o eval_data/dataset_export.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import load_settings
from src.evaluation.data_collector import create_data_collector


def main():
    parser = argparse.ArgumentParser(description="Export data from Langfuse Dataset")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Name of the Langfuse dataset to export",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: eval_data/dataset_{name}_{timestamp}.json)",
    )
    parser.add_argument(
        "--format",
        choices=["full", "ragas"],
        default="full",
        help="Output format: full (with metadata) or ragas (compatible with RAGAS)",
    )

    args = parser.parse_args()

    settings = load_settings()

    if not settings.langfuse.enabled:
        print("Error: LangFuse is not enabled in config/settings.yaml", file=sys.stderr)
        sys.exit(1)

    print(f"Exporting from Langfuse dataset: {args.dataset}")

    collector = create_data_collector(settings.langfuse)
    test_set = collector.collect_from_dataset(args.dataset)

    if not test_set.test_cases:
        print(f"No items found in dataset '{args.dataset}'")
        sys.exit(0)

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("eval_data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = args.dataset.replace("/", "_").replace("\\", "_")
        output_path = output_dir / f"dataset_{safe_name}_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "ragas":
        data = {
            "test_cases": [tc.to_ragas_format() for tc in test_set.test_cases],
            "created_at": test_set.created_at,
            "source": test_set.source,
            "description": test_set.description,
        }
    else:
        data = test_set.to_dict()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    pass_count = sum(1 for tc in test_set.test_cases if tc.pass_rate == "Pass")
    fail_count = sum(1 for tc in test_set.test_cases if tc.pass_rate == "Fail")
    unlabeled = len(test_set.test_cases) - pass_count - fail_count

    print(f"Exported {len(test_set.test_cases)} test cases to: {output_path}")
    if pass_count + fail_count > 0:
        print(f"  Pass: {pass_count}, Fail: {fail_count}, Unlabeled: {unlabeled}")


if __name__ == "__main__":
    main()

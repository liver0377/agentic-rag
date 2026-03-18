"""Export traces from LangFuse to JSON file.

Usage:
    python scripts/export_from_langfuse.py --days 7 --limit 100
    python scripts/export_from_langfuse.py --days 30 --output eval_data/langfuse_export.json
    python scripts/export_from_langfuse.py --tags rag,production --limit 500
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
    parser = argparse.ArgumentParser(description="Export traces from LangFuse")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to look back (default: 7)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of traces to export (default: 100)",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Filter by tags (e.g., --tags rag production)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: eval_data/langfuse_YYYYMMDD_HHMMSS.json)",
    )
    parser.add_argument(
        "--format",
        choices=["native", "simple"],
        default="native",
        help="Output format: native (full TestSet) or simple (array of test cases)",
    )

    args = parser.parse_args()

    settings = load_settings()

    if not settings.langfuse.enabled:
        print("Error: LangFuse is not enabled in config/settings.yaml", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to LangFuse: {settings.langfuse.host}")
    print(f"Fetching traces from last {args.days} days (limit: {args.limit})...")

    collector = create_data_collector(settings.langfuse)
    test_set = collector.collect_from_langfuse(
        days=args.days,
        limit=args.limit,
        tags=args.tags,
    )

    if not test_set.test_cases:
        print("No traces found matching the criteria.")
        sys.exit(0)

    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path("eval_data/raw")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"langfuse_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "simple":
        data = [tc.to_dict() for tc in test_set.test_cases]
    else:
        data = test_set.to_dict()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(test_set.test_cases)} test cases to: {output_path}")

    valid_count = sum(1 for tc in test_set.test_cases if tc.query and tc.answer)
    if valid_count < len(test_set.test_cases):
        print(f"Note: {valid_count} test cases have both query and answer")


if __name__ == "__main__":
    main()

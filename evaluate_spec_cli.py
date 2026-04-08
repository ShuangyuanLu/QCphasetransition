import argparse
import json
from pathlib import Path

from periodic_mc import evaluate_spec


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a JSON Monte Carlo spec and write a JSON result."
    )
    parser.add_argument(
        "input",
        help="Path to the input JSON spec file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output JSON result file. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--include-final-state",
        action="store_true",
        help="Include the final lattice state in the JSON result.",
    )
    parser.add_argument(
        "--no-measurements",
        action="store_true",
        help="Omit raw measurement arrays from the JSON result.",
    )
    parser.add_argument(
        "--tail-start-fraction",
        type=float,
        default=0.5,
        help="Fraction of the time series to discard before the tail summary. Default: 0.5.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent level for JSON output. Default: 2.",
    )
    return parser.parse_args()


def _load_spec(input_path):
    with open(input_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(result, output_path=None, indent=2):
    payload = json.dumps(result, indent=indent, sort_keys=True)
    if output_path is None:
        print(payload)
        return

    output_path = Path(output_path)
    output_path.write_text(payload + "\n", encoding="utf-8")


def main():
    args = _parse_args()
    spec = _load_spec(args.input)
    result = evaluate_spec(
        spec,
        include_measurements=not args.no_measurements,
        include_final_state=args.include_final_state,
        tail_start_fraction=args.tail_start_fraction,
    )
    _write_json(result, output_path=args.output, indent=args.indent)


if __name__ == "__main__":
    main()

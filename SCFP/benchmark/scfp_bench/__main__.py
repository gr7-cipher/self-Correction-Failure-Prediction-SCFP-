"""python -m scfp_bench <stage> --config configs/<file>.yaml"""

import argparse

import yaml

from . import stages

STAGES = ["holdout", "filter", "generate", "label", "split", "export", "agree", "validate"]


def main():
    ap = argparse.ArgumentParser(prog="scfp_bench")
    ap.add_argument("stage", choices=STAGES + ["all-auto"],
                    help="all-auto runs holdout..split (everything before human annotation)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--annotations", help="Label Studio JSON export or annotations.jsonl (agree)")
    ap.add_argument("--allow-mock", action="store_true",
                    help="validate a mock smoke-test run; never use for a release")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    todo = ["holdout", "filter", "generate", "label", "split"] if args.stage == "all-auto" else [args.stage]
    for s in todo:
        print(f"== {s}")
        if s == "agree":
            if not args.annotations:
                ap.error("agree needs --annotations")
            stages.stage_agree(cfg, args.annotations)
        elif s == "validate":
            stages.stage_validate(cfg, allow_mock=args.allow_mock)
        else:
            getattr(stages, f"stage_{s}")(cfg)


if __name__ == "__main__":
    main()

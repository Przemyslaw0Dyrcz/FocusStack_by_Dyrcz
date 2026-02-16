from focusstack.main import build_parser, run
import sys

def main() -> int:
    args = build_parser().parse_args()
    return run(args)

if __name__ == "__main__":
    sys.exit(main())

# convenience wrapper if you want to call programmatically
from focusstack.main import build_parser, run

def run_pipeline(args):
    # args: argparse.Namespace or object with same attributes
    return run(args)

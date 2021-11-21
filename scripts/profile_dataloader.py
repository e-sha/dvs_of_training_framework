import sys


try:
    from train_flownet import get_trainset_params
except ImportError:
    from pathlib import Path

    script_path = Path(__file__).resolve().parent
    sys.path.append(str(script_path.parent))
    from train_flownet import get_trainset_params


try:
    from train_flownet import get_dataloader, parse_args
    from utils.performance import get_iterable_performance
except ImportError:
    raise


def main(args):
    loader = get_dataloader(get_trainset_params(args))
    loader_perf = get_iterable_performance(loader)
    print(f'An average dataloader performance is {loader_perf} '
          'seconds per iteration')


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)

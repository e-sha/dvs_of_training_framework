import sys


try:
    from train_flownet import get_trainset_params
except ImportError:
    from pathlib import Path

    script_path = Path(__file__).resolve().parent
    sys.path.append(str(script_path.parent))


try:
    from train_flownet import get_dataset, parse_args
    from utils.performance import get_iterable_performance
except ImportError:
    pass


def main(args):
    loader = get_dataset(get_trainset_params(args))
    loader_perf = get_iterable_performance(loader, 100, 400)
    print(f'An average dataloader performance is {loader_perf} '
          'seconds per iteration')


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)

from time import perf_counter


def get_iterable_performance(loader,
                             start: int = 20,
                             num_iters: int = 50):
    """Returns average dataloader performance

    Args:
        loader:
            An iterable object to evaluate.
        start:
            A number of iteration to skip before evaluation.
        num_iters:
            A number of iterations for evaluation.

    Returns:
        An average time required for a single iteration of the loader.
    """
    assert num_iters > 0
    t0 = None
    t1 = None
    for i, _ in enumerate(loader):
        if i == start:
            t0 = perf_counter()
        elif i == start + num_iters:
            t1 = perf_counter()
            break
    assert t0 is not None and t1 is not None
    return (t1 - t0) / num_iters

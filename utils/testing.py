import numpy as np
import yaml
import itertools
from types import SimpleNamespace

from .eval import estimate_corresponding_gt_flow, flow_error_dense
from .data import get_count_image, frame_generator


def evaluate(of,
             events,
             frames,
             gt,
             event_preproc_fun=None,
             pred_postproc_fun=None,
             gt_proc_fun=None,
             is_car=False,
             log=False):
    ''' Evaluate quality of optical flow on the specified dataset

    Args:
        of (callable): of generator. That process a batch of events
        events (list): a list of sorted events in form of [x, y, t, p],
                       where each component is an array
        frames (list): a list of timestamp pairs that generate frame in
                       a form of [(start_ts, stop_ts)]
        gt (dict): a dictionary with keys: 'timestamps',
                   'x_flow_dist', 'y_flow_dist'
        event_preproc_fun (callable): a function to preprocess a set of
                                      events. For example, it may restrict
                                      events to some bounding box inside the
                                      image. Default: None
        pred_postproc_fun (callable): a function to postprocess predicted
                                      flow. For example it may crop box from
                                      the flow map. Default: None
        gt_proc_fun (callable): a function to preprocess true flow.
                                For example it may crop box from the flow map.
                                Default: None
        is_car (bool): Indicator of sequence captured from car. Default: False
        log (bool): the function prints intermediate statistics if True.
                    Default: False
    '''

    def ev_pre_fun(x):
        if event_preproc_fun is None:
            return x
        return event_preproc_fun(x)

    def fl_post_fun(x):
        if pred_postproc_fun is None:
            return x
        return pred_postproc_fun(x)

    def gt_post_fun(x):
        if gt_proc_fun is None:
            return x
        return gt_proc_fun(x)

    AEE_sum = 0
    percent_AEE_sum = 0

    max_flow_sum = 0
    min_flow_sum = 0
    for i, (e, start, stop) in enumerate(frame_generator(events, frames)):
        # preprocessing
        e = ev_pre_fun(np.array(e).T).T
        # prediction. Batch size is 1
        flow = of([e], [start], [stop])[0]
        # postprocessing
        flow = fl_post_fun(flow)

        # compute statistics
        max_flow_sum += np.max(flow)
        min_flow_sum += np.min(flow)

        # construct GT
        U_gt, V_gt = estimate_corresponding_gt_flow(gt['x_flow_dist'],
                                                    gt['y_flow_dist'],
                                                    gt['timestamps'],
                                                    start,
                                                    stop)
        gt_flow = np.dstack((U_gt, V_gt))
        # postprocess GT
        gt_flow = gt_post_fun(gt_flow)

        # compute errors
        event_count_image = get_count_image(e, gt_flow.shape[:2])
        AEE, percent_AEE, n_points = flow_error_dense(gt_flow,
                                                      flow,
                                                      event_count_image,
                                                      is_car)
        AEE_sum += AEE
        percent_AEE_sum += percent_AEE

        n = i + 1
        if log and n % 100 == 0:
            print('-------------------------------')
            print(f'Iter: {n}')
            print(f'Mean max flow: {max_flow_sum/n:.2f}, '
                  f'mean min flow: {min_flow_sum/n:.2f}')
            print(f'Mean AEE: {AEE_sum/n:.2f}, mean %AEE: '
                  f'{percent_AEE_sum/n:.2f}, #pts: {n_points},')

    res = (float(AEE_sum) / n, percent_AEE_sum / n)
    if log:
        print('Testing done.')
        print(f'Mean AEE: {res[0]:.6f}, mean %AEE: {res[1]:.6f}')
    return res


def read_config(filename):
    with open(str(filename), 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def option2list(option):
    if type(option) == list:
        return option
    return [option]


def shape2list(option):
    assert type(option) == list
    if type(option[0]) == list:
        return option
    return [option]


def ravel_config(config):
    cfg = {k: option2list(config[k])
           for k in ['start', 'stop', 'step', 'crop_type', 'is_car']}
    cfg['test_shape'] = shape2list(config['test_shape'])
    for (start,
            stop,
            step,
            test_shape,
            crop_type,
            is_car) in itertools.product(cfg['start'],
                                         cfg['stop'],
                                         cfg['step'],
                                         cfg['test_shape'],
                                         cfg['crop_type'],
                                         cfg['is_car']):
        yield SimpleNamespace(start=start,
                              stop=stop,
                              step=step,
                              test_shape=test_shape,
                              crop_type=crop_type,
                              is_car=is_car)

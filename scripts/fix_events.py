from argparse import ArgumentParser
import numpy as np
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


try:
    import tensorflow as tf
except ImportError:
    raise


def parse_args():
    parser = ArgumentParser(description='Removes incorrect events from event '
                                        'files. Correct event file should '
                                        'have ascending sequence of x values '
                                        'for each tag')
    parser.add_argument('input',
                        help='path to a directory with input events',
                        type=Path)
    parser.add_argument('output',
                        help='path to a directory with output events',
                        type=Path)
    return parser.parse_args()


def read_file(path):
    result = {}

    serialized_examples = tf.data.TFRecordDataset(str(path))
    try:
        for serialized_example in tqdm(serialized_examples, desc=path.name):
            event = tf.core.util.event_pb2 \
                      .Event.FromString(serialized_example.numpy())
            for v in event.summary.value:
                if v.tag not in result:
                    result[v.tag] = {'t': [], 'x': [], 'y': []}
                result[v.tag]['t'].append(event.wall_time)
                result[v.tag]['y'].append(v.simple_value)
                result[v.tag]['x'].append(event.step)
    except KeyboardInterrupt:
        raise
    except Exception:
        pass
    return result


def combine_events(events):
    tags = set([t for e in events for t in e])
    result = {t: {'t': [], 'x': [], 'y': []} for t in tags}
    for t in tags:
        for e in events:
            seq = e.pop(t, {'t': [], 'x': [], 'y': []})
            for k in result[t]:
                result[t][k] += seq[k]
    return result


def read_data(path):
    events = [read_file(f) for f in path.glob('**/events*')]
    return combine_events(events)


def reorder_events(events):
    result = {}
    for t, seq in events.items():
        idx = np.argsort(seq['t'])
        result[t] = {k: np.array(seq[k])[idx] for k in 'txy'}
    return result


def fix_events(events):
    result = {}
    for t, seq in events.items():
        v = np.minimum.accumulate(seq['x'][::-1])[::-1]
        mask = seq['x'] == v
        seq = {k: seq[k][mask] for k in 'txy'}
        mask = np.ones(seq['x'].size, dtype=np.bool)
        mask[:-1] = seq['x'][:-1] < seq['x'][1:]
        result[t] = {k: seq[k][mask] for k in 'txy'}
    return result


def write_events(path, events):
    flat = {'k': [], 't': [], 'x': [], 'y': []}
    for k, seq in events.items():
        flat['k'] += [np.full(len(seq['x']), k)]
        for n in 'xyt':
            flat[n] += [seq[n]]
    for k in flat:
        flat[k] = np.hstack(flat[k])
    idx = np.argsort(flat['t'])
    for k in flat:
        flat[k] = flat[k][idx]

    writer = SummaryWriter(path)
    for k, t, x, y in tqdm(zip(flat['k'], flat['t'], flat['x'], flat['y']),
                           desc='output',
                           total=flat['t'].size):
        writer.add_scalar(k, y, x, t)
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d '
                               '%(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('read event')
    events = read_data(args.input)
    logging.info('reorder events')
    events = reorder_events(events)
    logging.info('fix events')
    events = fix_events(events)
    logging.info('write events')
    write_events(args.output, events)
    logging.info('finished')

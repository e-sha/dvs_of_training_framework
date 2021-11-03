import h5py
from pathlib import Path
import tempfile
import torch
import time
from types import SimpleNamespace

from tests.utils import test_path, data_path, compare
from train_flownet import init_model, construct_train_tools
from utils.dataset import encode_batch, decode_batch, join_batches
from utils.dataset import select_encoded_ranges, read_encoded_batch
from utils.dataset import write_encoded_batch, PreprocessedDataloader
from utils.dataset import DatasetImpl, collate_wrapper
from utils.loss import Losses
from utils.timer import FakeTimer
from utils.training import train


class TestDatasetEncoding:
    def setup_class(self):
        self.decoded = {
            'events': {
                'x': torch.tensor([1, 2, 2, 1, 4, 6, 7], dtype=torch.long),
                'y': torch.tensor([2, 1, 3, 4, 1, 6, 8], dtype=torch.long),
                'timestamp': torch.tensor([0.02, 0.06, 0.07, 0.015, 0.01,
                                           0.05, 0.07], dtype=torch.float32),
                'polarity': torch.tensor([-1, 1, -1, 1, 1, 1, -1],
                                         dtype=torch.long),
                'element_index': torch.tensor([0, 1, 1, 0, 0, 2, 3],
                                              dtype=torch.long),
                'sample_index': torch.tensor([0, 0, 0, 1, 2, 2, 2],
                                             dtype=torch.long)},
            'timestamps': torch.tensor([0, 0.04, 0.08, 0, 0.03, 0,
                                        0.02, 0.04, 0.06, 0.08],
                                       dtype=torch.float32),
            'sample_idx': torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 2],
                                       dtype=torch.long),
            'images': torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8],
                                   dtype=torch.float32).view(-1, 1, 1, 1)
                                                       .tile(1, 1, 10, 10),
            'augmentation_params': {
                'idx': torch.tensor([0, 1, 2], dtype=torch.long),
                'sequence_length': torch.tensor([2, 1, 4], dtype=torch.short),
                'collapse_length': torch.tensor([1, 2, 3], dtype=torch.short),
                'box': torch.tensor([[0, 0, 10, 10],
                                     [0, 1, 10, 10],
                                     [1, 0, 10, 10]], dtype=torch.long),
                'angle': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
                'is_flip': torch.tensor([True, False, True])},
            'size': 3}
        self.encoded = {
            'events': {'x': torch.tensor([1, 2, 2, 1, 4, 6, 7],
                                         dtype=torch.short),
                       'y': torch.tensor([2, 1, 3, 4, 1, 6, 8],
                                         dtype=torch.short),
                       'timestamp': torch.tensor([0.02, 0.06, 0.07, 0.015,
                                                  0.01, 0.05, 0.07],
                                                 dtype=torch.float32),
                       'polarity': torch.tensor([False, True, False,
                                                 True, True, True, False]),
                       'events_per_element': torch.tensor([1, 2, 1, 1,
                                                           0, 1, 1],
                                                          dtype=torch.long),
                       'elements_per_sample': torch.tensor([2, 1, 4],
                                                           dtype=torch.short)},
            'timestamps': torch.tensor([0, 0.04, 0.08, 0, 0.03, 0,
                                        0.02, 0.04, 0.06, 0.08],
                                       dtype=torch.float32),
            'images': torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8],
                                   dtype=torch.uint8).view(-1, 1, 1, 1)
                                                     .tile(1, 1, 10, 10),
            'augmentation_params': {
                'idx': torch.tensor([0, 1, 2], dtype=torch.long),
                'sequence_length': torch.tensor([2, 1, 4], dtype=torch.short),
                'collapse_length': torch.tensor([1, 2, 3], dtype=torch.short),
                'box': torch.tensor([[0, 0, 10, 10],
                                     [0, 1, 10, 10],
                                     [1, 0, 10, 10]], dtype=torch.long),
                'angle': torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
                'is_flip': torch.tensor([True, False, True])}}

        self.encoded_parts = [
            {'events': {'x': torch.tensor([1, 2, 2, 1],
                                          dtype=torch.short),
                        'y': torch.tensor([2, 1, 3, 4],
                                          dtype=torch.short),
                        'timestamp': torch.tensor([0.02, 0.06, 0.07, 0.015],
                                                  dtype=torch.float32),
                        'polarity': torch.tensor([False, True, False,
                                                  True]),
                        'events_per_element': torch.tensor([1, 2, 1],
                                                           dtype=torch.long),
                        'elements_per_sample':
                        torch.tensor([2, 1], dtype=torch.short)},
             'timestamps': torch.tensor([0, 0.04, 0.08, 0, 0.03],
                                        dtype=torch.float32),
             'images': torch.tensor([0, 1, 2, 3, 4],
                                    dtype=torch.uint8).view(-1, 1, 1, 1)
                                                      .tile(1, 1, 10, 10),
             'augmentation_params': {
                 'idx': torch.tensor([0, 1], dtype=torch.long),
                 'sequence_length': torch.tensor([2, 1], dtype=torch.short),
                 'collapse_length': torch.tensor([1, 2], dtype=torch.short),
                 'box': torch.tensor([[0, 0, 10, 10],
                                      [0, 1, 10, 10]], dtype=torch.long),
                 'angle': torch.tensor([0.1, 0.2], dtype=torch.float32),
                 'is_flip': torch.tensor([True, False])}},
            {'events': {'x': torch.tensor([4, 6, 7],
                                          dtype=torch.short),
                        'y': torch.tensor([1, 6, 8],
                                          dtype=torch.short),
                        'timestamp': torch.tensor([0.01, 0.05, 0.07],
                                                  dtype=torch.float32),
                        'polarity': torch.tensor([True, True, False]),
                        'events_per_element': torch.tensor([1, 0, 1, 1],
                                                           dtype=torch.long),
                        'elements_per_sample':
                        torch.tensor([4], dtype=torch.short)},
             'timestamps': torch.tensor([0, 0.02, 0.04, 0.06, 0.08],
                                        dtype=torch.float32),
             'images': torch.tensor([5, 6, 7, 8],
                                    dtype=torch.uint8).view(-1, 1, 1, 1)
                                                      .tile(1, 1, 10, 10),
             'augmentation_params': {
                 'idx': torch.tensor([2], dtype=torch.long),
                 'sequence_length': torch.tensor([4], dtype=torch.short),
                 'collapse_length': torch.tensor([3], dtype=torch.short),
                 'box': torch.tensor([[1, 0, 10, 10]], dtype=torch.long),
                 'angle': torch.tensor([0.3], dtype=torch.float32),
                 'is_flip': torch.tensor([True])}}
            ]

    def test_encode(self):
        encoded = encode_batch(**self.decoded)
        compare(encoded, self.encoded)

    def test_decoded(self):
        decoded = decode_batch(self.encoded)
        compare(decoded, self.decoded)

    def test_join(self):
        joined = join_batches(self.encoded_parts)
        compare(joined, self.encoded)

    def test_batch_selection_indices(self):
        begin = 0
        end = 1
        gt = {'events': {'x': {'begin': 0, 'end': 3},
                         'y': {'begin': 0, 'end': 3},
                         'timestamp': {'begin': 0, 'end': 3},
                         'polarity': {'begin': 0, 'end': 3},
                         'events_per_element': {'begin': 0, 'end': 2},
                         'elements_per_sample': {'begin': 0, 'end': 1}},
              'timestamps': {'begin': 0, 'end': 3},
              'images': {'begin': 0, 'end': 3},
              'augmentation_params': {
                  'idx': {'begin': 0, 'end': 1},
                  'sequence_length': {'begin': 0, 'end': 1},
                  'collapse_length': {'begin': 0, 'end': 1},
                  'box': {'begin': 0, 'end': 1},
                  'angle': {'begin': 0, 'end': 1},
                  'is_flip': {'begin': 0, 'end': 1}}}
        prediction = select_encoded_ranges(
                self.encoded['events']['events_per_element'],
                self.encoded['events']['elements_per_sample'], begin, end)
        compare(prediction, gt)
        begin = 1
        end = 2
        gt = {'events': {'x': {'begin': 3, 'end': 4},
                         'y': {'begin': 3, 'end': 4},
                         'timestamp': {'begin': 3, 'end': 4},
                         'polarity': {'begin': 3, 'end': 4},
                         'events_per_element': {'begin': 2, 'end': 3},
                         'elements_per_sample': {'begin': 1, 'end': 2}},
              'timestamps': {'begin': 3, 'end': 5},
              'images': {'begin': 3, 'end': 5},
              'augmentation_params': {
                  'idx': {'begin': 1, 'end': 2},
                  'sequence_length': {'begin': 1, 'end': 2},
                  'collapse_length': {'begin': 1, 'end': 2},
                  'box': {'begin': 1, 'end': 2},
                  'angle': {'begin': 1, 'end': 2},
                  'is_flip': {'begin': 1, 'end': 2}}}
        prediction = select_encoded_ranges(
                self.encoded['events']['events_per_element'],
                self.encoded['events']['elements_per_sample'], begin, end)
        compare(prediction, gt)
        begin = 2
        end = 3
        gt = {'events': {'x': {'begin': 4, 'end': 7},
                         'y': {'begin': 4, 'end': 7},
                         'timestamp': {'begin': 4, 'end': 7},
                         'polarity': {'begin': 4, 'end': 7},
                         'events_per_element': {'begin': 3, 'end': 7},
                         'elements_per_sample': {'begin': 2, 'end': 3}},
              'timestamps': {'begin': 5, 'end': 10},
              'images': {'begin': 5, 'end': 10},
              'augmentation_params': {
                  'idx': {'begin': 2, 'end': 3},
                  'sequence_length': {'begin': 2, 'end': 3},
                  'collapse_length': {'begin': 2, 'end': 3},
                  'box': {'begin': 2, 'end': 3},
                  'angle': {'begin': 2, 'end': 3},
                  'is_flip': {'begin': 2, 'end': 3}}}
        prediction = select_encoded_ranges(
                self.encoded['events']['events_per_element'],
                self.encoded['events']['elements_per_sample'], begin, end)
        compare(prediction, gt)
        begin = 0
        end = 2
        gt = {'events': {'x': {'begin': 0, 'end': 4},
                         'y': {'begin': 0, 'end': 4},
                         'timestamp': {'begin': 0, 'end': 4},
                         'polarity': {'begin': 0, 'end': 4},
                         'events_per_element': {'begin': 0, 'end': 3},
                         'elements_per_sample': {'begin': 0, 'end': 2}},
              'timestamps': {'begin': 0, 'end': 5},
              'images': {'begin': 0, 'end': 5},
              'augmentation_params': {
                  'idx': {'begin': 0, 'end': 2},
                  'sequence_length': {'begin': 0, 'end': 2},
                  'collapse_length': {'begin': 0, 'end': 2},
                  'box': {'begin': 0, 'end': 2},
                  'angle': {'begin': 0, 'end': 2},
                  'is_flip': {'begin': 0, 'end': 2}}}
        prediction = select_encoded_ranges(
                self.encoded['events']['events_per_element'],
                self.encoded['events']['elements_per_sample'], begin, end)
        compare(prediction, gt)
        begin = 1
        end = 3
        gt = {'events': {'x': {'begin': 3, 'end': 7},
                         'y': {'begin': 3, 'end': 7},
                         'timestamp': {'begin': 3, 'end': 7},
                         'polarity': {'begin': 3, 'end': 7},
                         'events_per_element': {'begin': 2, 'end': 7},
                         'elements_per_sample': {'begin': 1, 'end': 3}},
              'timestamps': {'begin': 3, 'end': 10},
              'images': {'begin': 3, 'end': 10},
              'augmentation_params': {
                  'idx': {'begin': 1, 'end': 3},
                  'sequence_length': {'begin': 1, 'end': 3},
                  'collapse_length': {'begin': 1, 'end': 3},
                  'box': {'begin': 1, 'end': 3},
                  'angle': {'begin': 1, 'end': 3},
                  'is_flip': {'begin': 1, 'end': 3}}}
        prediction = select_encoded_ranges(
                self.encoded['events']['events_per_element'],
                self.encoded['events']['elements_per_sample'], begin, end)
        compare(prediction, gt)
        begin = 0
        end = 3
        gt = {'events': {'x': {'begin': 0, 'end': 7},
                         'y': {'begin': 0, 'end': 7},
                         'timestamp': {'begin': 0, 'end': 7},
                         'polarity': {'begin': 0, 'end': 7},
                         'events_per_element': {'begin': 0, 'end': 7},
                         'elements_per_sample': {'begin': 0, 'end': 3}},
              'timestamps': {'begin': 0, 'end': 10},
              'images': {'begin': 0, 'end': 10},
              'augmentation_params': {
                  'idx': {'begin': 0, 'end': 3},
                  'sequence_length': {'begin': 0, 'end': 3},
                  'collapse_length': {'begin': 0, 'end': 3},
                  'box': {'begin': 0, 'end': 3},
                  'angle': {'begin': 0, 'end': 3},
                  'is_flip': {'begin': 0, 'end': 3}}}
        prediction = select_encoded_ranges(
                self.encoded['events']['events_per_element'],
                self.encoded['events']['elements_per_sample'], begin, end)
        compare(prediction, gt)

    def test_read_prepared_batch(self):
        with tempfile.NamedTemporaryFile(suffix='.hdf5') as f:
            filename = f.name
        write_encoded_batch(filename, self.encoded)
        with h5py.File(filename, 'r') as f:
            elements_per_sample = \
                torch.tensor(f['events']['elements_per_sample'])
            events_per_element = \
                torch.tensor(f['events']['events_per_element'])
            batch = read_encoded_batch(f, events_per_element,
                                       elements_per_sample, 0, 2)
        compare(batch, self.encoded_parts[0])
        with h5py.File(filename, 'r') as f:
            elements_per_sample = \
                torch.tensor(f['events']['elements_per_sample'])
            events_per_element = \
                torch.tensor(f['events']['events_per_element'])
            batch = read_encoded_batch(f, events_per_element,
                                       elements_per_sample, 2, 3)
        Path(filename).unlink()
        compare(batch, self.encoded_parts[1])

    def test_preprocessed_dataloader(self):
        with tempfile.TemporaryDirectory() as dirname:
            dirname = Path(dirname)
            for i, part in enumerate(self.encoded_parts):
                write_encoded_batch(dirname/f'{i}.hdf5', part)
            dataloader = PreprocessedDataloader(dirname, 2)
            batch = next(dataloader)
            compare(batch, decode_batch(self.encoded_parts[0]))

            dataloader = PreprocessedDataloader(dirname, 1)
            dataloader.set_index(2)
            batch = next(dataloader)
            compare(batch, decode_batch(self.encoded_parts[1]))

            dataloader = PreprocessedDataloader(dirname, 3)
            batch = next(dataloader)
            compare(batch, decode_batch(join_batches(self.encoded_parts)))

            dataloader = PreprocessedDataloader(dirname, 5)
            batch = next(dataloader)
            compare(batch, decode_batch(join_batches(
                self.encoded_parts + [self.encoded_parts[0]])))

    def test_training_preprocessed(self):
        shape = [256, 256]
        dataset = DatasetImpl(path=data_path,
                              shape=shape,
                              augmentation=True,
                              collapse_length=1,
                              is_raw=True)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  collate_fn=collate_wrapper,
                                                  batch_size=2,
                                                  pin_memory=True,
                                                  shuffle=False)
        args = SimpleNamespace(wdw=0.01,
                               training_steps=1,
                               rs=0,
                               optimizer='ADAM',
                               lr=0.01,
                               half_life=1,
                               device=torch.device('cpu'),
                               num_warmup_steps=0)
        model = init_model(
                SimpleNamespace(flownet_path=test_path.parent/'EV_FlowNet',
                                mish=False, sp=None, prefix_length=0,
                                suffix_length=0, max_sequence_length=1,
                                dynamic_sample_length=False,
                                event_representation_depth=9),
                device=args.device)
        optimizer, scheduler = construct_train_tools(args, model)
        batch_size = 4
        evaluator = Losses([tuple(map(lambda x: x // 2 ** i, shape))
                            for i in range(4)][::-1], batch_size, args.device)
        with tempfile.TemporaryDirectory() as dirname:
            dirname = Path(dirname)
            batch = next(iter(data_loader))
            write_encoded_batch(dirname/'0.hdf5', encode_batch(**batch))
            dataloader = PreprocessedDataloader(dirname, batch_size)

            logger = torch.utils.tensorboard.SummaryWriter(log_dir=dirname)
            train(model=model, device=args.device, loader=dataloader,
                  optimizer=optimizer, num_steps=args.training_steps,
                  scheduler=scheduler, logger=logger, evaluator=evaluator,
                  timers=FakeTimer())
            del logger
        time.sleep(1)

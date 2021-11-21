from argparse import ArgumentParser
import copy
import h5py
import sys
import torch
from tqdm import tqdm


try:
    from train_flownet import get_trainset_params
except ImportError:
    from pathlib import Path

    script_path = Path(__file__).resolve().parent
    sys.path.append(str(script_path.parent))
    from train_flownet import get_trainset_params


try:
    from train_flownet import get_dataloader
    from utils.common import write_execution_info, collect_execution_info
    from utils.common import check_execution_info
    from utils.dataset import encode_quantized_batch, write_encoded_batch
    from utils.dataset import join_batches
    from utils.dataloader import choose_data_path
    from utils.model import init_model
    from utils.options import validate_dataset_args, add_dataset_arguments
    from utils.options import add_dataset_preprocessing_arguments
    from utils.options import add_dataloader_arguments, add_common_arguments
    from utils.options import add_model_arguments, validate_quantization_args
    from utils.options import add_preprocessed_dataset_arguments
except ImportError:
    raise


def parse_args(args, is_write=True):
    parser = ArgumentParser()
    parser = add_common_arguments(parser)
    parser = add_dataset_arguments(parser)
    parser = add_dataloader_arguments(parser)
    parser = add_model_arguments(parser)
    parser = add_dataset_preprocessing_arguments(parser)
    parser = add_preprocessed_dataset_arguments(parser)
    args = parser.parse_args(args)
    args = validate_dataset_args(args)
    args = validate_quantization_args(args)

    args.output.mkdir(exist_ok=True, parents=True)
    args = choose_data_path(args)

    execution_info = collect_execution_info(args)
    check_execution_info(args.output, execution_info, args)
    args.output.mkdir(exist_ok=True, parents=True)
    if is_write:
        write_execution_info(args.output, execution_info)
    return args


def main(args):
    model = init_model(args, device=args.device)
    args.output.mkdir(exist_ok=True)
    written_files = list(args.output.glob('*.hdf5'))
    written_indices = [int(f.stem) for f in written_files]
    num_written = 0
    for filename in written_files:
        with h5py.File(filename, 'r') as f:
            num_written += len(f['elements_per_sample'])
    loader = get_dataloader(get_trainset_params(args), sample_idx=num_written)
    num_batches_per_write = (args.samples_per_file - 1) // args.mbs + 1
    encoded_batches = []
    j = 0
    initial = num_written // args.mbs
    total = (args.size - num_written) // args.mbs + initial
    for i, batch in tqdm(enumerate(loader), initial=initial, total=total):
        if num_written >= args.size:
            break
        imsize = batch['images'].size()[-2:]
        quantized_batch = copy.deepcopy(batch)
        del quantized_batch['events']
        with torch.no_grad():
            quantized_batch['data'] = model.quantize(batch['events'],
                                                     batch['timestamps'],
                                                     batch['sample_idx'],
                                                     imsize)
        del batch
        encoded_batches.append(encode_quantized_batch(quantized_batch))
        del quantized_batch
        num_written += \
            len(encoded_batches[-1]['elements_per_sample'])
        is_last = num_written >= args.size
        if (i + 1) % num_batches_per_write == 0 or is_last:
            joined_batches = join_batches(encoded_batches)
            del encoded_batches
            while j in written_indices:
                j += 1
            write_encoded_batch(args.output/f'{j}.hdf5', joined_batches)
            j += 1
            del joined_batches
            encoded_batches = []
        if is_last:
            break


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)

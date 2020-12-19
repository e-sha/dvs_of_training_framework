import io
from parse import parse
from pathlib import Path
import logging
import torch


def _sure_N_args_string(template : str, N: int, err_msg : str):
    try:
        res = template.format(*([0]*N))
        if N != 0 and res == template:
            raise IndexError
    except IndexError as e:
        raise logging.error(f'{err_msg} But ' + template + ' is given')


def remove_tree(path):
    if path.is_file():
        path.unlink()
        return
    to_remove = [path]
    while len(to_remove) > 0:
        el = to_remove[0]
        assert el.is_dir()
        items = [x for x in el.iterdir()]
        [x.unlink() for x in items if x.is_file()]
        dirs2remove = [x for x in items if x.is_dir()]
        if len(dirs2remove) > 0:
            to_remove = dirs2remove + to_remove
        else:
            el.rmdir()
            to_remove = to_remove[1:]

class Serializer:
    def __init__(self,
                 path : Path,                          # path to write checkpoints
                 keep_checkpoints_max : int,           # number of last checkpoints to keep
                 permanent_checkpoint_interval : int,  # interval between permanent checkpoints. Don't store permanent checkpoints if value is 0
                 name_template='step_{}.pt'            # template for a model id
                 ):
        self._path = Path(path)
        self._history_size = keep_checkpoints_max
        self._permanent_interval = permanent_checkpoint_interval
        self._permanent_checkpoints = dict()
        self._temporal_checkpoints = dict()
        _sure_N_args_string(name_template, 1,
                            'checkpoint name template for the serializer '
                            'has to use exactly one argument -- checkpoint id.')
        self._name_template = name_template
        self._find_checkpoints()

    def _remove_old(self):
        if self._history_size <= 0:
            return
        temporal_steps = sorted(list(self._temporal_checkpoints.keys()), key=lambda x: -x)
        for step in temporal_steps[self._history_size:]:
            remove_tree(self._path/self._temporal_checkpoints.pop(step))
            logging.info(f'Checkpoint with ID={step} is removed')

    def _find_checkpoints(self):
        names = list(map(lambda x: x.name, self._path.iterdir()))
        keys = [parse(self._name_template, str(name)) for name in names]
        known_checkpoints = {int(step[0]): name for step, name in zip(keys, names)
                             if step is not None and step[0].isdigit()}
        if self._permanent_interval > 0:
            self._permanent_checkpoints = {s: n for s, n in known_checkpoints.items() if
                                           s % self._permanent_interval == 0}
        self._temporal_checkpoints = {s: n for s, n in known_checkpoints.items() if
                                      s not in self._permanent_checkpoints}

    def _id2path(self, global_step):
        return self._path/self._name_template.format(global_step)

    def checkpoint_model(self, model, optimizer, global_step, **kwargs):
        """Utility function for checkpointing model + optimizer dictionaries
           The main purpose for this is to be able to resume training from that instant again
        """
        path = self._id2path(global_step)
        if self._permanent_interval > 0 and global_step % self._permanent_interval == 0:
            self._permanent_checkpoints[global_step] = path.name
        else:
            self._temporal_checkpoints[global_step] = path.name

        checkpoint_state_dict = {'model': model.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'global_step': global_step}
        # Add extra kwargs too
        checkpoint_state_dict.update(kwargs)

        # write model and optimizer
        torch.save(checkpoint_state_dict, path)

        self._remove_old()
        status_msg = f'checkpointing: PATH={path.parent}, ckpt_id={path.name}'

    def has_checkpoints(self):
        return len(self._temporal_checkpoints) + len(self._permanent_checkpoints) > 0

    def list_known_steps(self):
        steps = list(self._temporal_checkpoints.keys()) + list(self._permanent_checkpoints.keys())
        return sorted(steps)

    def load_checkpoint(self, model, global_step, optimizer=None, device=None):
        """Utility function for checkpointing model + optimizer dictionaries
           The main purpose for this is to be able to resume training from that instant again
        """
        if global_step not in self._temporal_checkpoints and global_step not in self._permanent_checkpoints:
            raise ValueError(f'Checkpoint for step {global_step} not found')
        path = self._id2path(global_step)
        checkpoint_state_dict = torch.load(path, map_location=device)
        global_step = checkpoint_state_dict['global_step']
        model.load_state_dict(checkpoint_state_dict['model'])
        if optimizer:
            optimizer.load_state_dict(checkpoint_state_dict['optimizer'])
        checkpoint_state_dict.pop('global_step', None)
        checkpoint_state_dict.pop('model', None)
        checkpoint_state_dict.pop('optimizer', None)
        return global_step, checkpoint_state_dict

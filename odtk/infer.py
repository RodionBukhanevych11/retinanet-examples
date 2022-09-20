import os
import json
import tempfile
from contextlib import redirect_stdout
import torch
from apex import amp
from apex.parallel import DistributedDataParallel as ADDP
from torch.nn.parallel import DistributedDataParallel
from pycocotools.cocoeval import COCOeval
import numpy as np
from tqdm import tqdm
from .data import DataIterator, RotatedDataIterator
from .dali import DaliDataIterator
from .model import Model
from copy import deepcopy
from .utils import Profiler, rotate_box, get_metrics


def infer(model, path, detections_file, resize, max_size, batch_size, mixed_precision=True, world=0,
          annotations=None, with_apex=False, use_dali=True, is_validation=False, verbose=True, rotated_bbox=False, nc = 3):
    'Run inference on images from path'

    DDP = DistributedDataParallel if not with_apex else ADDP
    backend = 'pytorch' if isinstance(model, Model) or isinstance(model, DDP) else 'tensorrt'

    stride = model.module.stride if isinstance(model, DDP) else model.stride

    # Create annotations if none was provided
    if not annotations:
        annotations = tempfile.mktemp('.json')
        images = [{'id': i, 'file_name': f} for i, f in enumerate(os.listdir(path))]
        json.dump({'images': images}, open(annotations, 'w'))

    # TensorRT only supports fixed input sizes, so override input size accordingly
    if backend == 'tensorrt': max_size = max(model.input_size)

    # Prepare dataset
    if verbose: print('Preparing dataset...')
    if rotated_bbox:
        if use_dali: raise NotImplementedError("This repo does not currently support DALI for rotated bbox.")
        data_iterator = RotatedDataIterator(path, resize, max_size, batch_size, stride,
                                            world, annotations, training=False)
    else:
        data_iterator = (DaliDataIterator if use_dali else DataIterator)(
            path, resize, max_size, batch_size, stride,
            world, annotations, training=False)

    # Prepare model
    if backend == 'pytorch':
        # If we are doing validation during training,
        # no need to register model with AMP again
        if not is_validation:
            if torch.cuda.is_available(): model = model.to(memory_format=torch.channels_last).cuda()
            if with_apex:
                model = amp.initialize(model, None,
                                    opt_level='O2' if mixed_precision else 'O0',
                                    keep_batchnorm_fp32=True,
                                    verbosity=0)

        model.eval()

    if verbose:
        print('   backend: {}'.format(backend))
        print('    device: {} {}'.format(
            world, 'cpu' if not torch.cuda.is_available() else 'GPU' if world == 1 else 'GPUs'))
        print('     batch: {}, precision: {}'.format(batch_size,
                                                     'unknown' if backend == 'tensorrt' else 'mixed' if mixed_precision else 'full'))
        print(' BBOX type:', 'rotated' if rotated_bbox else 'axis aligned')
        print('Running inference...')

    results = []
    profiler = Profiler(['infer', 'fw'])
    with torch.no_grad():
        for data, target, ids, ratio in tqdm(data_iterator):
            # Forward pass
            if backend=='pytorch': data = data.contiguous(memory_format=torch.channels_last)
            scores, boxes, classes = model(data, rotated_bbox) #Need to add model size (B, 3, W, H)
            for sample_i in range(len(boxes)):
                results.append([scores[sample_i], boxes[sample_i], classes[sample_i], target[sample_i]])
    # Gather results from all devices
    metrics_dict = {'tp':0,'tn':0,'fp':0,'fn':0}
    metrics_05 = []
    for _ in range(nc):
        metrics_05.append(deepcopy(metrics_dict))
    f1_05 = [0 for i in range(nc)]
    for result in results:
        scores, boxes, classes, targets = result
        keep = (scores >= 0).nonzero(as_tuple=False)
        scores = scores[keep].view(-1)
        boxes = boxes[keep, :].view(-1, 4)
        classes = classes[keep].view(-1).int()
        metrics_05 = get_metrics(boxes, classes, targets, metrics_05, 0.4)
    for i in range(nc):
        pr_05 = metrics_05[i]['tp'] / (metrics_05[i]['tp']+metrics_05[i]['fp'] + 1e-9)
        recall_05 = metrics_05[i]['tp'] / (metrics_05[i]['tp']+metrics_05[i]['fn'] + 1e-9)
        f1_05[i] = 2 * pr_05 * recall_05/(pr_05 + recall_05 + 1e-9)    
        
    return f1_05

import os.path
import time
import json
import warnings
import signal
from datetime import datetime
from contextlib import contextmanager
from PIL import Image, ImageDraw
import requests
import numpy as np
import math
import torch
from torch import nn


def bbox_iou_numpy(bb1, bb2):
    bb1[2], bb1[3] = bb1[0] + bb1[2], bb1[1] + bb1[3]
    bb2[2], bb2[3] = bb2[0] + bb2[2], bb2[1] + bb2[3]
    print(bb1, bb2)
    if bb1[0] >= bb1[2] or bb1[1] >= bb1[3] or bb2[0] >= bb2[2] or bb2[1] >= bb2[3]:
        return 0.0

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_metrics_per_class(pr_bboxes,pr_labels,targets,metrics,iou_th):
    for pr_index in range(len(pr_bboxes)):
        predicted_label = int(pr_labels[pr_index])
        predicted_box = pr_bboxes[pr_index]
        if len(targets) == 0:
            metrics[predicted_label]['fp']+=1
        else:
            iou_all = np.array([bbox_iou_numpy(target[:4],predicted_box) for target in targets])
            if (iou_all >= iou_th).any():
                matched_targets_index = np.where(iou_all>=iou_th)[0]
                matched_targets = targets[matched_targets_index][:,-1]
                for target in matched_targets:
                    target_label = int(target[-1])
                    if predicted_label == target_label:
                        metrics[predicted_label]['tp']+=1
                    else:
                        metrics[predicted_label]['fn']+=1
                        metrics[target_label]['fp']+=1
            else:
                metrics[predicted_label]['fn']+=1
                
    return metrics


def get_metrics(pr_bboxes, pr_labels, targets, metrics, iou_th):
    pr_bboxes = pr_bboxes.detach().cpu().numpy()
    pr_labels = pr_labels.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    if len(pr_bboxes)==0:
        for target in temp_targets:
            target_label = target[-1]
            if target_label>-1:
                metrics[label]['fn']+=1
    else:
        metrics = get_metrics_per_class(pr_bboxes,pr_labels,targets,metrics,iou_th)
    return metrics


def order_points(pts):
    pts_reorder = []

    for idx, pt in enumerate(pts):
        idx = torch.argsort(pt[:, 0])
        xSorted = pt[idx, :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        leftMost = leftMost[torch.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        D = torch.cdist(tl[np.newaxis], rightMost)[0]
        (br, tr) = rightMost[torch.argsort(D, descending=True), :]
        pts_reorder.append(torch.stack([tl, tr, br, bl]))

    return torch.stack([p for p in pts_reorder])

def rotate_boxes(boxes, points=False):
    '''
    Rotate target bounding boxes

    Input:
        Target boxes (xmin_ymin, width_height, theta)
    Output:
        boxes_axis (xmin_ymin, xmax_ymax, theta)
        boxes_rotated (xy0, xy1, xy2, xy3)
    '''

    u = torch.stack([torch.cos(boxes[:,4]), torch.sin(boxes[:,4])], dim=1)
    l = torch.stack([-torch.sin(boxes[:,4]), torch.cos(boxes[:,4])], dim=1)
    R = torch.stack([u, l], dim=1)

    if points:
        cents = torch.stack([(boxes[:,0]+boxes[:,2])/2, (boxes[:,1]+boxes[:,3])/2],1).transpose(1,0)
        boxes_rotated = torch.stack([boxes[:,0],boxes[:,1],
            boxes[:,2], boxes[:,1],
            boxes[:,2], boxes[:,3],
            boxes[:,0], boxes[:,3],
            boxes[:,-2],
            boxes[:,-1]],1)

    else:
        cents = torch.stack([boxes[:,0]+(boxes[:,2])/2, boxes[:,1]+(boxes[:,3])/2],1).transpose(1,0)
        boxes_rotated = torch.stack([boxes[:,0],boxes[:,1],
            (boxes[:,0]+boxes[:,2]), boxes[:,1],
            (boxes[:,0]+boxes[:,2]), (boxes[:,1]+boxes[:,3]),
            boxes[:,0], (boxes[:,1]+boxes[:,3]),
            boxes[:,-2],
            boxes[:,-1]],1)

    xy0R = torch.matmul(R,boxes_rotated[:,:2].transpose(1,0) - cents) + cents
    xy1R = torch.matmul(R,boxes_rotated[:,2:4].transpose(1,0) - cents) + cents
    xy2R = torch.matmul(R,boxes_rotated[:,4:6].transpose(1,0) - cents) + cents
    xy3R = torch.matmul(R,boxes_rotated[:,6:8].transpose(1,0) - cents) + cents

    xy0R = torch.stack([xy0R[i,:,i] for i in range(xy0R.size(0))])
    xy1R = torch.stack([xy1R[i,:,i] for i in range(xy1R.size(0))])
    xy2R = torch.stack([xy2R[i,:,i] for i in range(xy2R.size(0))])
    xy3R = torch.stack([xy3R[i,:,i] for i in range(xy3R.size(0))])

    boxes_axis = torch.cat([boxes[:, :2], boxes[:, :2] + boxes[:, 2:4] - 1,
        torch.sin(boxes[:,-1, None]), torch.cos(boxes[:,-1, None])], 1)
    boxes_rotated = order_points(torch.stack([xy0R,xy1R,xy2R,xy3R],dim = 1)).view(-1,8)

    return boxes_axis, boxes_rotated


def rotate_box(bbox):
    xmin, ymin, width, height, theta = bbox

    xy1 = xmin, ymin
    xy2 = xmin, ymin + height - 1
    xy3 = xmin + width - 1, ymin + height - 1
    xy4 = xmin + width - 1, ymin

    cents = np.array([xmin + (width - 1) / 2, ymin + (height - 1) / 2])

    corners = np.stack([xy1, xy2, xy3, xy4])

    u = np.stack([np.cos(theta), -np.sin(theta)])
    l = np.stack([np.sin(theta), np.cos(theta)])
    R = np.vstack([u, l])

    corners = np.matmul(R, (corners - cents).transpose(1, 0)).transpose(1, 0) + cents

    return corners.reshape(-1).tolist()


def show_detections(detections):
    'Show image with drawn detections'

    for image, detections in detections.items():
        im = Image.open(image).convert('RGBA')
        overlay = Image.new('RGBA', im.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        detections.sort(key=lambda d: d['score'])
        for detection in detections:
            box = detection['bbox']
            alpha = int(detection['score'] * 255)
            draw.rectangle(box, outline=(255, 255, 255, alpha))
            draw.text((box[0] + 2, box[1]), '[{}]'.format(detection['class']),
                      fill=(255, 255, 255, alpha))
            draw.text((box[0] + 2, box[1] + 10), '{:.2}'.format(detection['score']),
                      fill=(255, 255, 255, alpha))
        im = Image.alpha_composite(im, overlay)
        im.show()


def save_detections(path, detections):
    print('Writing detections to {}...'.format(os.path.basename(path)))
    with open(path, 'w') as f:
        json.dump(detections, f)


@contextmanager
def ignore_sigint():
    handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, handler)


class Profiler(object):
    def __init__(self, names=['main']):
        self.names = names
        self.lasts = {k: 0 for k in names}
        self.totals = self.lasts.copy()
        self.counts = self.lasts.copy()
        self.means = self.lasts.copy()
        self.reset()

    def reset(self):
        last = time.time()
        for name in self.names:
            self.lasts[name] = last
            self.totals[name] = 0
            self.counts[name] = 0
            self.means[name] = 0

    def start(self, name='main'):
        self.lasts[name] = time.time()

    def stop(self, name='main'):
        self.totals[name] += time.time() - self.lasts[name]
        self.counts[name] += 1
        self.means[name] = self.totals[name] / self.counts[name]

    def bump(self, name='main'):
        self.stop(name)
        self.start(name)


def post_metrics(url, metrics):
    try:
        for k, v in metrics.items():
            requests.post(url,
                          data={'time': int(datetime.now().timestamp() * 1e9),
                                'metric': k, 'value': v})
    except Exception as e:
        warnings.warn('Warning: posting metrics failed: {}'.format(e))

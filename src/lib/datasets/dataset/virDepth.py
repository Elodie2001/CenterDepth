from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import torch

import torch.utils.data as data


class virDepth(data.Dataset):
    num_classes = 3
    default_resolution = [512, 512]
    mean = np.array([0.61068779, 0.60996849, 0.58054898],
                    dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0.29232583, 0.29280022, 0.30315563],
                   dtype=np.float32).reshape((1, 1, 3))

    def __init__(self, opt, split):
        super(virDepth, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'Town01')
        self.img_dir = os.path.join(self.data_dir, 'images')
        self.seg_img_dir = os.path.join(self.data_dir, 'segimg')
        self.depth_img_dir = os.path.join(self.data_dir, 'depthimg')
        if split == 'val':
            self.annot_path = os.path.join(
                self.data_dir, 'annotations', 'val_town01_depth.json')
        else:
            if opt.task == 'exdet':
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations', 'train_town01_depth.json')
            if split == 'test':
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations', 'test_town01_depth.json')
            else:
                self.annot_path = os.path.join(
                    self.data_dir, 'annotations', 'train_town01_depth.json')
        self.max_objs = 128
        self.class_name = ['pedestrian', 'car', 'truck']
        self._valid_ids = [1, 2, 3]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        print('==> initializing carla {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        print('Loaded {} {} samples'.format(split, self.num_samples))

    @staticmethod
    def _to_float(x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        # import pdb; pdb.set_trace()
        detections = []
        for image_id in all_bboxes:
            for cls_ind in all_bboxes[image_id]:
                category_id = self._valid_ids[cls_ind - 1]
                for bbox in all_bboxes[image_id][cls_ind]:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = list(map(self._to_float, bbox[0:4]))

                    detection = {
                        "image_id": int(image_id),
                        "category_id": int(category_id),
                        "bbox": bbox_out,
                        "score": float("{:.2f}".format(score))
                    }
                    if len(bbox) > 5:
                        extreme_points = list(map(self._to_float, bbox[5:13]))
                        detection["extreme_points"] = extreme_points
                    detections.append(detection)
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()




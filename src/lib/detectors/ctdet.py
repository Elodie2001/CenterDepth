from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
  from external.nms import soft_nms
except:
  print('NMS not imported! If you need it,'
        ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctdet_decode
from models.utils import flip_tensor
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector

class CtdetDetector(BaseDetector):
  def __init__(self, opt):
    super(CtdetDetector, self).__init__(opt)
  
  def process(self, images, return_time=False):
    with torch.no_grad():
      output = self.model(images)[-1]
      hm = output['hm'].sigmoid_()
      wh = output['wh']
      dis = output['dis']
      reg = output['reg'] if self.opt.reg_offset else None
      if self.opt.flip_test:
        hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
        wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
        reg = reg[0:1] if reg is not None else None
      torch.cuda.synchronize()
      forward_time = time.time()
      dets, distances = ctdet_decode(hm, wh, dis, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
      # print(distances)
    if return_time:
      return output, dets, distances, forward_time
    else:
      return output, dets, distances
    

  def post_process(self, dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], self.opt.num_classes)
    for j in range(1, self.num_classes + 1):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]

  def merge_outputs(self, detections, distances):
    results = {}
    angles_dict = {}
    x_coords_dict = {}
    y_coords_dict = {}
    dis_pred = {}
    for j in range(1, self.num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)
        if len(self.scales) > 1 or self.opt.nms:
            soft_nms(results[j], Nt=0.5, method=2)

    image_width = 1920
    camera_fov = 90
    for j in range(1, self.num_classes + 1):
        bboxes = results[j]
        centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2

        angles = np.arctan(centers_x / image_width) * camera_fov
        angles = 45 - angles
        angles_dict[j] = angles
    
    scores = np.hstack(
        [results[j][:, 4] for j in range(1, self.num_classes + 1)])
    if len(scores) > self.max_per_image:
        kth = len(scores) - self.max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, self.num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
    angles = []
    dis = []
    for j in range(1, self.num_classes + 1):
        angle = angles_dict[j]
        angles.extend(angle)
        dis = distances[j]
    x_coords = dis * np.sin(np.radians(angles))
    y_coords = dis * np.cos(np.radians(angles))
    start_idx = 0
    for j in range(1, self.num_classes + 1):
      angles = angles_dict[j]
      end_idx = start_idx + len(angles)
      dis_pred[j] = dis[start_idx:end_idx]
      x_coords_dict[j] = x_coords[start_idx:end_idx]
      y_coords_dict[j] = y_coords[start_idx:end_idx]
      start_idx = end_idx
      # print(dis_pred)
    return results, x_coords_dict, y_coords_dict

  def debug(self, debugger, images, dets, output, scale=1):
    detection = dets.detach().cpu().numpy().copy()
    detection[:, :, :4] *= self.opt.down_ratio
    for i in range(1):
      img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
      img = ((img * self.std + self.mean) * 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm_{:.1f}'.format(scale))
      debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
      for k in range(len(dets[i])):
        if detection[i, k, 4] > self.opt.center_thresh:
          debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                 detection[i, k, 4], 
                                 img_id='out_pred_{:.1f}'.format(scale))

  # def show_results(self, debugger, image, results):
  #   debugger.add_img(image, img_id='ctdet')
  #   for j in range(1, self.num_classes + 1):
  #     for bbox in results[j]:
  #       if bbox[4] > self.opt.vis_thresh:
  #         debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
  #   debugger.show_all_imgs(pause=self.pause)
  def show_results(self, debugger, image, results, x, y):
    debugger.add_img(image, img_id='ctdet')
    for j in range(1, self.num_classes + 1):
        for idx, bbox in enumerate(results[j]):
            if bbox[4] > self.opt.vis_thresh:
                debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
                debugger.add_bev_points(x, y)
    debugger.show_all_imgs(pause=self.pause)

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import cv2
import numpy as np
import onnxruntime as ort
from config import *
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool

try:
    import sahi
    from sahi.slicing import slice_image
except Exception as e:
    print(
        'sahi not found, plaese install sahi. '
        'for example: `pip install sahi`, see https://github.com/obss/sahi.'
    )
    raise e

def multiclass_nms(bboxs, num_classes, match_threshold=0.6, match_metric='ios'):
    final_boxes = []
    for c in range(num_classes):
        idxs = bboxs[:, 0] == c
        if np.count_nonzero(idxs) == 0: continue
        r = nms(bboxs[idxs, 1:], match_threshold, match_metric)
        final_boxes.append(np.concatenate([np.full((r.shape[0], 1), c), r], 1))
    return final_boxes


def nms(dets, match_threshold=0.6, match_metric='iou'):
    """ Apply NMS to avoid detecting too many overlapping bounding boxes.
        Args:
            dets: shape [N, 5], [score, x1, y1, x2, y2]
            match_metric: 'iou' or 'ios'
            match_threshold: overlap thresh for match metric.
    """
    if dets.shape[0] == 0:
        return dets[[], :]
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            if match_metric == 'iou':
                union = iarea + areas[j] - inter
                match_value = inter / union
            elif match_metric == 'ios':
                smaller = min(iarea, areas[j])
                match_value = inter / smaller
            else:
                raise ValueError()
            if match_value >= match_threshold:
                suppressed[j] = 1
    keep = np.where(suppressed == 0)[0]
    dets = dets[keep, :]
    return dets

class PicoDet():
    def __init__(self,
                 model_pb_path,
                 label_path,
                 prob_threshold=0.4,
                 nms_threshold=0.3):
        # 读取类别文件，获取类别列表
        self.classes = list(
            map(lambda x: x.strip(), open(label_path, 'r').readlines()))
        self.num_classes = len(self.classes)
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold

        # 均值、标准差，用于归一化
        self.mean = np.array(
            [103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(
            [57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)

        # 实例化onnx推理session
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model_pb_path, so)

        # 根据网络结构，获取输入名称和尺寸
        inputs_name = [a.name for a in self.net.get_inputs()]
        inputs_shape = {
            k: v.shape
            for k, v in zip(inputs_name, self.net.get_inputs())
        }
        self.input_shape = inputs_shape['image'][2:]

    def _normalize(self, img):
        '''
        图像归一化
        '''
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean / 255.0) / (self.std / 255.0)
        return img

    def resize_image(self, srcimg, keep_ratio=False):
        '''
        图像缩放

        Args:
            srcimg 原始输入图片
        Returns:
            keep_ratio 是否保持原图宽高比
        '''
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        origin_shape = srcimg.shape[:2]
        im_scale_y = newh / float(origin_shape[0])
        im_scale_x = neww / float(origin_shape[1])
        img_shape = np.array([
            [float(self.input_shape[0]), float(self.input_shape[1])]
        ]).astype('float32')
        scale_factor = np.array([[im_scale_y, im_scale_x]]).astype('float32')
        img = cv2.resize(srcimg,  tuple(self.input_shape), interpolation=2)

        return img, img_shape, scale_factor

    def preprocess(self,srcimg):
        '''
        数据预处理
        '''
        # 缩放到推理尺寸
        img, im_shape, scale_factor = self.resize_image(srcimg)
        img = img[:,:,::-1]
        # 归一化
        img = self._normalize(img)
        # BGR转RGB
        img = img[:,:,::-1]
        # 维度转置+添加维度
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
        inputs_dict = {
            'im_shape': im_shape,
            'image': blob,
            'scale_factor': scale_factor
        }
        inputs_name = [a.name for a in self.net.get_inputs()]
        net_inputs = {k: inputs_dict[k] for k in inputs_name}
        return net_inputs
    
    def slice_preprocess(self,srcimg):
        '''
        数据预处理（没有归一化这一步）
        '''
        # 缩放到推理尺寸
        img, im_shape, scale_factor = self.resize_image(srcimg)
        # 维度转置+添加维度
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
        inputs_dict = {
            'im_shape': im_shape,
            'image': blob,
            'scale_factor': scale_factor
        }
        inputs_name = [a.name for a in self.net.get_inputs()]
        net_inputs = {k: inputs_dict[k] for k in inputs_name}
        return net_inputs
    
    def det_onnx(self, srcimg):
        '''
        目标检测模型推理接口

        Args:
            srcimg 原始数据
        Returns:
            result_list 检测结果列表
        '''
        net_inputs = self.preprocess(srcimg)

        result_list = []
        try:
            outs = self.net.run(None, net_inputs)
            outs = np.array(outs[0])
            # 过滤检测结果：置信度大于阈值，索引大于-1
            expect_boxes = (outs[:, 1] > self.prob_threshold) & (outs[:, 0] > -1)
            result_boxes = outs[expect_boxes, :]
            
            for i in range(result_boxes.shape[0]):
                class_id, conf = int(result_boxes[i, 0]), result_boxes[i, 1]
                class_name = self.classes[class_id]
                xmin, ymin, xmax, ymax = int(result_boxes[i, 2]), int(result_boxes[
                    i, 3]), int(result_boxes[i, 4]), int(result_boxes[i, 5])
                result = {"classid":class_id,
                        "classname":class_name,
                        "confidence":conf,
                        "box":[xmin,ymin,xmax,ymax]}
                result_list.append(result)
        except Exception as e:
            print(e)

        return result_list

    def run_slice(self,srcimg):
        '''
        多线程的原子操作
        '''
        net_inputs = self.slice_preprocess(srcimg)
        outs = self.net.run(None, net_inputs)
        boxes = np.array(outs[0])
        return boxes

    def det_onnx_slice(self,
                        srcimg,
                        slice_size=[640,640],
                        overlap_ratio=[0.25,0.25]):
        ''' 
        目标检测模型推理接口(自动切图拼图)

        Args:
            srcimg 原始数据
            slice_size: 裁剪尺寸
            overlap_ratio: 块之间的重叠比例
        Returns:
            result_list 检测结果列表
        '''
        # 切分原图
        slice_image_result = sahi.slicing.slice_image(
                image=Image.fromarray(srcimg),
                slice_height=slice_size[0],
                slice_width=slice_size[1],
                overlap_height_ratio=overlap_ratio[0],
                overlap_width_ratio=overlap_ratio[1])
        sub_img_num = len(slice_image_result)
        batch_image_list = [slice_image_result.images[_ind] for _ind in range(sub_img_num)]

        result_list = []
        try:
            # 多块并行检测
            trs = time.time()
            pool = ThreadPool(processes = process_num)
            pred_boxes = pool.map(self.run_slice, batch_image_list)
            pool.close()
            pool.join()
            tre = time.time()

            print("det run cost: ", tre - trs)

            # 坐标映射
            merged_bboxs = []
            for _ind in range(sub_img_num):
                # 保留置信度大于阈值的检测框
                expect_boxes = (pred_boxes[_ind][:, 1] > self.prob_threshold) & (pred_boxes[_ind][:, 0] > -1)
                pred_boxes[_ind] = pred_boxes[_ind][expect_boxes, :]

                shift_amount = slice_image_result.starting_pixels[_ind]
                pred_boxes[_ind][:, 2:4] = pred_boxes[_ind][:, 2:4] + shift_amount
                pred_boxes[_ind][:, 4:6] = pred_boxes[_ind][:, 4:6] + shift_amount
                merged_bboxs.append(pred_boxes[_ind])
            # nms后处理
            final_boxes = multiclass_nms(np.concatenate(merged_bboxs),
                                        self.num_classes,
                                        match_threshold= self.nms_threshold,
                                        match_metric="ios"
                                        )
            outs = np.concatenate(final_boxes)
            print(len(outs))
            print("det nms cost: ", time.time()-tre)

            # 过滤检测结果：置信度大于阈值，索引大于-1
            expect_boxes = (outs[:, 1] > self.prob_threshold) & (outs[:, 0] > -1)
            result_boxes = outs[expect_boxes, :]
            
            for i in range(result_boxes.shape[0]):
                class_id, conf = int(result_boxes[i, 0]), result_boxes[i, 1]
                # 因为切图拼图模型训练时，0是ico,1是text,所以这里需要反过来
                class_id = 1- class_id
                class_name = self.classes[class_id]
                xmin, ymin, xmax, ymax = int(result_boxes[i, 2]), int(result_boxes[
                    i, 3]), int(result_boxes[i, 4]), int(result_boxes[i, 5])
                result = {"classid": class_id, 
                        "classname":class_name,
                        "confidence":conf,
                        "box":[xmin,ymin,xmax,ymax]}
                result_list.append(result)
        except Exception as e:
            print(e)
            return []

        return result_list

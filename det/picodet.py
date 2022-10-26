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

from time import time
import cv2
import numpy as np
import onnxruntime as ort
from config import *

class PicoDet():
    def __init__(self,
                 model_pb_path,
                 label_path,
                 prob_threshold=0.4,
                 iou_threshold=0.3):
        # 读取类别文件，获取类别列表
        self.classes = list(
            map(lambda x: x.strip(), open(label_path, 'r').readlines()))
        self.num_classes = len(self.classes)
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold

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

        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_shape[0], int(self.input_shape[1] /
                                                      hw_scale)
                img = cv2.resize(
                    srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_shape[1] - neww) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    0,
                    0,
                    left,
                    self.input_shape[1] - neww - left,
                    cv2.BORDER_CONSTANT,
                    value=0)  # add border
            else:
                newh, neww = int(self.input_shape[0] *
                                 hw_scale), self.input_shape[1]
                img = cv2.resize(
                    srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_shape[0] - newh) * 0.5)
                img = cv2.copyMakeBorder(
                    img,
                    top,
                    self.input_shape[0] - newh - top,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0)
        else:
            img = cv2.resize(
                srcimg,  tuple(self.input_shape), interpolation=2)

        return img, img_shape, scale_factor

    def detect_onnx(self, srcimg,modle_with_nms = True):
        '''
        目标检测模型推理接口

        Args:
            srcimg 原始数据
            modle_with_nms 模型是否内置NMS
        Returns:
            result_list 检测结果列表
        '''

        # 缩放到推理尺寸
        img, im_shape, scale_factor = self.resize_image(srcimg)
        # 归一化
        img = self._normalize(img)
        # BGR转BGR
        img = img[:,:,::-1]
        # 维度转置+添加维度
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        inputs_dict = {
            'im_shape': im_shape,
            'image': blob,
            'scale_factor': scale_factor
        }
        inputs_name = [a.name for a in self.net.get_inputs()]
        net_inputs = {k: inputs_dict[k] for k in inputs_name}

        t1 = time()
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

   
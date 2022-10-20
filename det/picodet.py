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
import paddle
import paddle._C_ops as C_ops
from config import *
paddle.set_device('cpu')

def multiclass_nms(bboxes,
                   scores,
                   score_threshold,
                   nms_top_k,
                   keep_top_k,
                   nms_threshold=0.3,
                   normalized=True,
                   nms_eta=1.,
                   background_label=-1,
                   return_index=False,
                   return_rois_num=True,
                   rois_num=None,
                   name=None):
    """
    This operator is to do multi-class non maximum suppression (NMS) on
    boxes and scores.
    In the NMS step, this operator greedily selects a subset of detection bounding
    boxes that have high scores larger than score_threshold, if providing this
    threshold, then selects the largest nms_top_k confidences scores if nms_top_k
    is larger than -1. Then this operator pruns away boxes that have high IOU
    (intersection over union) overlap with already selected boxes by adaptive
    threshold NMS based on parameters of nms_threshold and nms_eta.
    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.
    Args:
        bboxes (Tensor): Two types of bboxes are supported:
                           1. (Tensor) A 3-D Tensor with shape
                           [N, M, 4 or 8 16 24 32] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           2. (LoDTensor) A 3-D Tensor with shape [M, C, 4]
                           M is the number of bounding boxes, C is the
                           class number
        scores (Tensor): Two types of scores are supported:
                           1. (Tensor) A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is
                           number of bounding boxes. For each category there
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes.
                           2. (LoDTensor) A 2-D LoDTensor with shape [M, C].
                           M is the number of bbox, C is the class number.
                           In this case, input BBoxes should be the second
                           case with shape [M, C, 4].
        background_label (int): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score. If not provided,
                                 consider all boxes.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        nms_threshold (float): The threshold to be used in NMS. Default: 0.3
        nms_eta (float): The threshold to be used in NMS. Default: 1.0
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        normalized (bool): Whether detections are normalized. Default: True
        return_index(bool): Whether return selected index. Default: False
        rois_num(Tensor): 1-D Tensor contains the number of RoIs in each image. 
            The shape is [B] and data type is int32. B is the number of images.
            If it is not None then return a list of 1-D Tensor. Each element 
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default.
        name(str): Name of the multiclass nms op. Default: None.
    Returns:
        A tuple with two Variables: (Out, Index) if return_index is True,
        otherwise, a tuple with one Variable(Out) is returned.
        Out: A 2-D LoDTensor with shape [No, 6] represents the detections.
        Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
        or A 2-D LoDTensor with shape [No, 10] represents the detections.
        Each row has 10 values: [label, confidence, x1, y1, x2, y2, x3, y3,
        x4, y4]. No is the total number of detections.
        If all images have not detected results, all elements in LoD will be
        0, and output tensor is empty (None).
        Index: Only return when return_index is True. A 2-D LoDTensor with
        shape [No, 1] represents the selected index which type is Integer.
        The index is the absolute value cross batches. No is the same number
        as Out. If the index is used to gather other attribute such as age,
        one needs to reshape the input(N, M, 1) to (N * M, 1) as first, where
        N is the batch size and M is the number of boxes.
    Examples:
        .. code-block:: python

            import paddle
            from ppdet.modeling import ops
            boxes = paddle.static.data(name='bboxes', shape=[81, 4],
                                      dtype='float32', lod_level=1)
            scores = paddle.static.data(name='scores', shape=[81],
                                      dtype='float32', lod_level=1)
            out, index = ops.multiclass_nms(bboxes=boxes,
                                            scores=scores,
                                            background_label=0,
                                            score_threshold=0.5,
                                            nms_top_k=400,
                                            nms_threshold=0.3,
                                            keep_top_k=200,
                                            normalized=False,
                                            return_index=True)
    """
   
    attrs = ('background_label', background_label, 'score_threshold',
                score_threshold, 'nms_top_k', nms_top_k, 'nms_threshold',
                nms_threshold, 'keep_top_k', keep_top_k, 'nms_eta', nms_eta,
                'normalized', normalized)
    output, index, nms_rois_num = C_ops.multiclass_nms3(bboxes, scores,
                                                        rois_num, *attrs)
    if not return_index:
        index = None
    return output, nms_rois_num, index

class MultiClassNMS(object):
    def __init__(self,
                 score_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 nms_threshold=nmsThreshold,
                 normalized=True,
                 nms_eta=1.0,
                 return_index=False,
                 return_rois_num=True,
                 trt=False):
        super(MultiClassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.return_index = return_index
        self.return_rois_num = return_rois_num
        self.trt = trt

    def __call__(self, bboxes, score, background_label=-1):
        """
        bboxes (Tensor|List[Tensor]): 1. (Tensor) Predicted bboxes with shape 
                                         [N, M, 4], N is the batch size and M
                                         is the number of bboxes
                                      2. (List[Tensor]) bboxes and bbox_num,
                                         bboxes have shape of [M, C, 4], C
                                         is the class number and bbox_num means
                                         the number of bboxes of each batch with
                                         shape [N,] 
        score (Tensor): Predicted scores with shape [N, C, M] or [M, C]
        background_label (int): Ignore the background label; For example, RCNN
                                is num_classes and YOLO is -1. 
        """
        kwargs = self.__dict__.copy()
        if isinstance(bboxes, tuple):
            bboxes, bbox_num = bboxes
            kwargs.update({'rois_num': bbox_num})
        if background_label > -1:
            kwargs.update({'background_label': background_label})
        kwargs.pop('trt')
        
        return multiclass_nms(bboxes, score, **kwargs)

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
        self.nms = MultiClassNMS()

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
                srcimg, self.input_shape, interpolation=2)

        return img, img_shape, scale_factor

    def detect_onnx(self, srcimg,modle_with_nms = False):
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
            # print("run cost: ",time()-t1)
            # 模型内部不带NMS，需自己加上NMS
            if not modle_with_nms:
                pred_bboxes,pred_scores = paddle.to_tensor(outs[0],place=paddle.CPUPlace()),paddle.to_tensor(outs[1],place=paddle.CPUPlace())
                scale_factor_t = paddle.to_tensor(scale_factor,place=paddle.CPUPlace())
                scale_y, scale_x = paddle.split(scale_factor_t, 2, axis=-1)
                scale_factor_t = paddle.concat(
                    [scale_x, scale_y, scale_x, scale_y],
                    axis=-1).reshape([-1, 1, 4])
                # scale bbox to origin image size.
                pred_bboxes /= scale_factor_t
                bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
                outs=[np.array(bbox_pred)]
            
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

   
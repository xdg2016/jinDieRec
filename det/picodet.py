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
import uuid
import cv2
import numpy as np
import onnxruntime as ort
from config import *
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool
try:
    from openvino.runtime import Core
except Exception as e:
    print("The current platform does not support this library !")
    pass

try:
    import sahi
    from sahi.slicing import slice_image
except Exception as e:
    print(
        'sahi not found, plaese install sahi. '
        'for example: `pip install sahi`, see https://github.com/obss/sahi.'
    )
    raise e

def multiclass_nms_(bboxs, num_classes, match_threshold=0.6, match_metric='ios'):
    '''
    多目标nms，用于切图拼图合并
    '''
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

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]      # 取剩余的检测框
        rest_boxes = boxes[indexes, :]

        # 根据边界过滤掉一些框，减少IOU的判断时间
        r_thresh = 1/2
        br = rest_boxes[...,0] > (current_box[...,0] + current_box[...,2])*(1-r_thresh) # 右侧框
        bl = rest_boxes[...,2] < (current_box[...,0] + current_box[...,2])*r_thresh     # 左侧框
        bt = rest_boxes[...,1] > (current_box[...,1] + current_box[...,3])*(1-r_thresh) # 下侧框
        bb = rest_boxes[...,3] < (current_box[...,1] + current_box[...,3])*r_thresh     # 上侧框
        mask = ( br | bl | bt | bb )
        good_indexes = indexes[mask]    # 已经符合条件的index
        rest_indexes = indexes[(1 - mask).astype(np.bool)]
        rest_boxes = boxes[rest_indexes]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(
                current_box, axis=0), )
        # indexes = rest_indexes[iou <= iou_threshold]
        indexes = np.concatenate([good_indexes, rest_indexes[iou <= iou_threshold]])

    return box_scores[picked, :]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


class PicoDetNMS(object):
    """
    Args:
        input_shape (int): network input image size
        scale_factor (float): scale factor of ori image
    """

    def __init__(self,
                 score_threshold=0.2,
                 nms_threshold=0.7,
                 nms_top_k=1000,
                 keep_top_k=300):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k

    def __call__(self, decode_boxes, select_scores):
        batch_size = 1
        out_boxes_list = []
        for batch_id in range(batch_size):
            # nms
            bboxes = np.concatenate(decode_boxes, axis=0)
            confidences = np.concatenate(select_scores, axis=0)
            picked_box_probs = []
            picked_labels = []
            for class_index in range(0, confidences.shape[1]):
                probs = confidences[:, class_index]
                mask = probs > self.score_threshold
                probs = probs[mask]
                if probs.shape[0] == 0:
                    continue
                subset_boxes = bboxes[mask, :]
                box_probs = np.concatenate(
                    [subset_boxes, probs.reshape(-1, 1)], axis=1)
                box_probs = hard_nms(
                    box_probs,
                    iou_threshold=self.nms_threshold,
                    top_k=self.keep_top_k,
                    candidate_size= 1000 )
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.shape[0])

            if len(picked_box_probs) == 0:
                out_boxes_list.append(np.empty((0, 4)))

            else:
                picked_box_probs = np.concatenate(picked_box_probs)
                # clas score box
                out_boxes_list.append(
                    np.concatenate(
                        [
                            np.expand_dims(
                                np.array(picked_labels),
                                axis=-1), np.expand_dims(
                                    picked_box_probs[:, 4], axis=-1),
                            picked_box_probs[:, :4]
                        ],
                        axis=1))

        out_boxes_list = np.concatenate(out_boxes_list, axis=0)
        return out_boxes_list


def get_uuid():
    '''
    获取uuid
    '''
    get_timestamp_uuid = uuid.uuid1()  # 根据 时间戳生成 uuid , 保证全球唯一
    return get_timestamp_uuid

def merge_boxes2(ori_boxes):
    '''
    合并检测框
    args:
        ori_boxes: 检测框结果
    return:
        new_boxes: 合并后的检测框
    '''
    new_boxes = []
    delta = 5
    h_delta = 10
    wh_r_th = 1.5
    merged_boxes_array = None
    merged_boxes = []
    mask = np.ones(len(ori_boxes)).astype(np.bool)
    for i in range(len(ori_boxes)):
        mask[i] = False
        box = ori_boxes[i]
        l,r,t,b = box[2],box[4],box[3],box[5]
        w,h = r-l+1, b-t+1
        c = box[0]

        # 加上新合并完的框，也参与判断
        boxes = ori_boxes[mask]
        if merged_boxes:
            boxes = np.concatenate((boxes,merged_boxes_array))
        boxes_c = boxes[:,0]
        boxes_l = boxes[:,2]
        boxes_r = boxes[:,4]
        boxes_t = boxes[:,3]
        boxes_b = boxes[:,5]
        boxes_h = boxes_b - boxes_t + 1
        boxes_w = boxes_r - boxes_l + 1
        
        idxs_r = set(np.where((boxes_l < r) & (boxes_l > l))[0])                                                        # 右侧相邻框
        idxs_l = set(np.where((boxes_r > l) & (boxes_r < r))[0])                                                        # 左侧相邻框
        idxs_samel = set(np.where(((abs(b - boxes_b) < h_delta) | (abs(t - boxes_t) < h_delta)) 
                        & (abs(h - boxes_h) < h_delta))[0])                                                             # 同行框
        idxs_cover = set(np.where((l>=boxes_l) & (r <= boxes_r))[0])                                                    # 覆盖框
        # idxs_text = set(np.where( boxes_whr > wh_r_th)[0])                                                            # 宽高比大于阈值，认为是文字行
        idxs_samec = set(np.where(c==boxes_c)[0])                                                                          # 类别相同

        idxs = (idxs_l | idxs_r | idxs_cover)  & idxs_samel & idxs_samec
        if len(idxs) == 0:                                  # 没有同行左或右相邻的框
            new_boxes.append(tuple(list(box))) 
        else:
            tmp_boxes = np.concatenate((boxes[np.array(list(idxs))],box[np.newaxis]))
            for idx in idxs:
                if tuple(boxes[idx,:]) in new_boxes:     # 如果在合并后的框集合中
                    new_boxes.remove(tuple(boxes[idx]))
            xmin = tmp_boxes[:,2].min()
            ymin = tmp_boxes[:,3].min()
            xmax = tmp_boxes[:,4].max()
            ymax = tmp_boxes[:,5].max()
            conf = tmp_boxes[:,1].max()
            class_id = tmp_boxes[:,0][np.argmax(conf)]
            new_boxes.append((class_id,conf,xmin,ymin,xmax,ymax))
            merged_boxes.append((class_id,conf,xmin,ymin,xmax,ymax))
            merged_boxes_array = np.array([list(box) for box in set(merged_boxes)])
    new_boxes = np.array([list(box) for box in set(new_boxes)])
    return new_boxes

class PicoDet():
    def __init__(self,
                 model_pb_path,
                 slice_model_pb_path,
                 label_path,
                 prob_threshold=0.4,
                 nms_threshold=0.3):
        # 读取类别文件，获取类别列表
        self.classes = list(
            map(lambda x: x.strip(), open(label_path, 'r').readlines()))
        self.num_classes = len(self.classes)
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold
        # self.nms = multiclass_nms()
        self.nms = PicoDetNMS(score_threshold=self.prob_threshold,
                            nms_threshold=self.nms_threshold,
                            nms_top_k=1000,
                            keep_top_k=300)

        # 均值、标准差，用于归一化(BGR)
        self.mean = np.array(
            [103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(
            [57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)

        # 初始化onnx推理
        self.net = self.onnx_init(model_pb_path)
        self.net_slice = self.onnx_init(slice_det_model_path)

        # 初始化openvino推理
        self.net_vino = self.openvino_init(model_pb_path)        
        self.net_vino_slice = self.openvino_init(slice_model_pb_path)
        # 根据网络结构，获取输入名称和尺寸
        inputs_name = [a.name for a in self.net.get_inputs()]
        inputs_shape = {
            k: v.shape
            for k, v in zip(inputs_name, self.net.get_inputs())
        }
        self.input_shape = inputs_shape['image'][2:]
        # 切图模型输入
        inputs_name = [a.name for a in self.net_slice.get_inputs()]
        inputs_shape = {
            k: v.shape
            for k, v in zip(inputs_name, self.net_slice.get_inputs())
        }
        self.input_shape_slice = inputs_shape['image'][2:]


    def onnx_init(self,model_path):
        '''
        onnx模型初始化
        '''
        so = ort.SessionOptions()
        so.log_severity_level = 3
        try:
            net = ort.InferenceSession(model_path, so)
        except Exception as ex:
            print(ex)
            net = None
        return net

    def openvino_init(self,model_path):
        '''
        openvino模型初始化
        '''
        try:
            ie = Core()
            net = ie.read_model(model_path)
            input_layer = net.input(0)
            input_shape = input_layer.partial_shape
            # 输入batch改为1
            input_shape[0] = 1
            net.reshape({input_layer: input_shape})
            compiled_model = ie.compile_model(net, 'CPU')
        except Exception as ex:
            print(ex)
            compiled_model = None
        return compiled_model

    def _normalize(self, img):
        '''
        图像归一化
        '''
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean / 255.0) / (self.std / 255.0)
        # img = img / 255.0
        return img

    def resize_image(self, srcimg, slice=False):
        '''
        图像缩放

        Args:
            srcimg 原始输入图片
        Returns:
            keep_ratio 是否保持原图宽高比
        '''
        origin_shape = srcimg.shape[:2]
        neww,newh = det_w,det_h
        if origin_shape[0] > default_imgh or origin_shape[1] > default_imgw:
            newh = int((origin_shape[0]*resize_ratio_h)/32) * 32
            neww = int((origin_shape[1]*resize_ratio_w)/32) * 32

        if slice:
             newh, neww = self.input_shape_slice[0], self.input_shape_slice[1]
        
        im_scale_y = newh / float(origin_shape[0])
        im_scale_x = neww / float(origin_shape[1])
        img_shape = np.array([
            [float(newh), float(neww)]
        ]).astype('float32')
        scale_factor = np.array([[im_scale_y, im_scale_x]]).astype('float32')
        img = cv2.resize(srcimg,  (neww,newh), interpolation=2)

        return img, img_shape, scale_factor

    def preprocess(self,srcimg):
        '''
        数据预处理
        '''
        # 缩放到推理尺寸
        img, im_shape, scale_factor = self.resize_image(srcimg)
        img = img[:,:,::-1]
        # 归一化
        # img = self._normalize(img)
        # img = img/255
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
        切图拼图检测流程的数据预处理
        '''
        # 缩放到推理尺寸
        img, im_shape, scale_factor = self.resize_image(srcimg,slice=True)
        # 转成BGR
        img = img[:,:,::-1]
        # img = self._normalize(img)        # ppyoloe不用这句，picodet需要
        # 转回RGB
        img = img[:,:,::-1]
        # 维度转置+添加维度
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
        inputs_dict = {
            'im_shape': im_shape,
            'image': blob,
            'scale_factor': scale_factor
        }
        inputs_name = [a.name for a in self.net_slice.get_inputs()]
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
            ts = time.time()
            t = 1
            for i in range(t):
                outs = self.net.run(None, net_inputs)
            print("infer cost:",(time.time()-ts)/t)

            # nms
            num_outs = int(len(outs) / 2)
            decode_boxes = []
            select_scores = []
            for out_idx in range(num_outs):
                decode_boxes.append(outs[out_idx])
                select_scores.append(outs[out_idx + num_outs])
            outs = self.nms(decode_boxes, select_scores)
            t3 = time.time()
            print("infer+nms cost:",t3-ts)

            
            # 过滤检测结果：置信度大于阈值，索引大于-1
            # outs = np.array(outs[0])                                                # 模型内部不带nms时，需注释这一句
            expect_boxes = (outs[:, 1] > self.prob_threshold) & (outs[:, 0] > -1)
            result_boxes = outs[expect_boxes, :]
            t_m = time.time()
            result_boxes = merge_boxes2(result_boxes)
            print("merge cost:",time.time()-t_m)
            
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

    def vino_preprocess(self,img,net):
        # n,c, H,W = list(net.inputs[0].shape)
        origin_shape = img.shape[:2]
        neww,newh = det_w,det_h
        if origin_shape[0] > default_imgh or origin_shape[1] > default_imgw:
            newh = int((origin_shape[0]*resize_ratio_h)/32) * 32
            neww = int((origin_shape[1]*resize_ratio_w)/32) * 32
        im_scale_y = newh / float(img.shape[0])
        im_scale_x = neww / float(img.shape[1])
        scale_factor = np.array([[im_scale_y, im_scale_x]]).astype('float32')
        img = cv2.resize(img, (neww, newh), interpolation=2)
        # 转成BGR
        img = img[:,:,::-1]
        # img = self._normalize(img)  # ppyoloe模型不需要normalize,但是picodet需要
        # 转回RGB
        img = img[:,:,::-1]
        # 维度转置+添加维度
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
        return blob,scale_factor

    def det_vino(self,srcimg):
        '''
        目标检测模型推理接口(基于openvino库推理)
        '''
        result_list = []
        img,scale_factor = self.vino_preprocess(srcimg,self.net_vino)
        try:
            t1 = time.time()
            t = 1
            for i in range(t):
                output = self.net_vino.infer_new_request({0: img,1:scale_factor})   # 这里要加上缩放因子
            t2 = time.time()
            print("infer cost:",(t2-t1)/t)
            outs = list(output.values())

            # # nms ,模型内部不带nms时，需执行这一段
            num_outs = int(len(outs) / 2)
            decode_boxes = []
            select_scores = []
            for out_idx in range(num_outs):
                decode_boxes.append(outs[out_idx])
                select_scores.append(outs[out_idx + num_outs])
            outs = self.nms(decode_boxes, select_scores)
            t3 = time.time()
            print("infer+nms cost:",t3-t1)
            
            # 过滤检测结果：置信度大于阈值，索引大于-1
            # outs = np.array(outs[0])                                                # 模型内部不带nms时，需注释这一句
            expect_boxes = (outs[:, 1] > self.prob_threshold) & (outs[:, 0] > -1)
            result_boxes = outs[expect_boxes, :]
            t3 = time.time()
            # 合并检测框
            result_boxes = merge_boxes2(result_boxes)
            print("merge cost:",time.time()-t3)
            
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

    def det_vino_slice(self,
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
        t1 = time.time()
        # 切分原图
        slice_image_result = sahi.slicing.slice_image(
                image=Image.fromarray(srcimg),
                slice_height=slice_size[0],
                slice_width=slice_size[1],
                overlap_height_ratio=overlap_ratio[0],
                overlap_width_ratio=overlap_ratio[1])
        sub_img_num = len(slice_image_result)
        batch_image_list = [slice_image_result.images[_ind] for _ind in range(sub_img_num)]
        imgs_list = [self.vino_preprocess(img,self.net_vino_slice) for img in batch_image_list]
        t2 = time.time()
        print("slice img cost:",t2-t1)


        result_list = []    # 保存返回结果
        try:
            pred_bboxes_all = []
            pred_scores_all = []
            for _ind,(img,scale_factor) in enumerate(imgs_list):
                output = self.net_vino_slice.infer_new_request({0: img,1:scale_factor})
                outs = list(output.values())

                # 带nms时的操作
                outs = np.array(outs[0])
                # 按置信度过滤
                expect_boxes = (outs[:, 1] > self.prob_threshold) & (outs[:, 0] > -1)
                result_boxes = outs[expect_boxes, :]
                # 坐标从子图映射回原图
                shift_amount = slice_image_result.starting_pixels[_ind]
                result_boxes[:, 2:4] = result_boxes[:, 2:4] + shift_amount
                result_boxes[:, 4:] = result_boxes[:, 4:] + shift_amount  
                pred_bboxes_all.append(result_boxes)
            
                # 不带nms时的操作
                # pred_boxes,pred_scores = outs[0][0],outs[1][0]  # 1,3598,4  1,c,3598
                # cls_ids = np.argmax(pred_scores,0)
                # pred_scores_max = np.max(pred_scores,0)
                # # 保留置信度大于阈值的检测框
                # expect_boxes = (pred_scores_max > self.prob_threshold)
                # pred_boxes = pred_boxes[expect_boxes, :]
                # pred_scores = pred_scores[:,expect_boxes]
                # cls_ids = cls_ids[expect_boxes]
                # scale_y, scale_x = scale_factor[0][0] , scale_factor[0][1]
                # scale_factor_t = np.array([scale_x, scale_y, scale_x, scale_y]).reshape([-1, 4])
                # # 缩放回原始尺寸
                # pred_boxes /= scale_factor_t
                # # 相对坐标映射
                # shift_amount = slice_image_result.starting_pixels[_ind]
                # pred_boxes[:, :2] = pred_boxes[:, :2] + shift_amount
                # pred_boxes[:, 2:] = pred_boxes[:, 2:] + shift_amount     
                # # 子图不做nms,
                # pred_bboxes_all.append(np.concatenate([cls_ids.reshape(-1,1),pred_scores[0].reshape(-1,1),pred_boxes],1))

            
            t3 = time.time()
            print("slice infer cost:",t3-t2)
            
            # 检测框合并
            result_boxes = merge_boxes2(np.concatenate(pred_bboxes_all))
            t4 = time.time()
            print("slice merge cost:",t4-t3)
 
            # nms后处理去重(耗时较多,建议使用合并)
            # final_boxes = multiclass_nms_(np.concatenate(pred_bboxes_all),
            #                             self.num_classes,
            #                             match_threshold= self.nms_threshold,
            #                             match_metric="ios"
            #                             )
            # result_boxes = np.concatenate(final_boxes)
            # print(len(result_boxes))
            # t4 = time.time()
            # print("slice nms cost:",t4-t3)

            for i in range(result_boxes.shape[0]):
                class_id, conf = int(result_boxes[i, 0]), result_boxes[i, 1]
                # 因为切图拼图模型训练时，0是ico,1是text,所以这里需要反过来
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

    def run_slice(self,srcimg):
        '''
        多线程的原子操作
        '''
        net_inputs = self.slice_preprocess(srcimg)
        t1 = time.time()
        outs = self.net_slice.run(None, net_inputs)
        print("infer cost:",time.time()-t1)
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
        uuid = get_uuid()
        # for i,img in enumerate(batch_image_list):
        #     Image.fromarray(img).save(f"F:/Datasets/securety/PageRec/test_data/test_sliced/{uuid}_{i}.jpg")


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
            # nms后处理,合并
            final_boxes = multiclass_nms_(np.concatenate(merged_bboxs),
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

    def infer(self,img):
        # self.net_vino = None
        if self.net_vino is None:
            print("infer by onnx ...")
            det_results = self.det_onnx(img)
        else:
            print("infer by openvino ...")
            det_results = self.det_vino(img)
        return det_results

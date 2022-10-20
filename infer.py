from .picodet.picodet import PicoDet
from talkapprec.config import *

# 初始化模型
net = PicoDet(model_pb_path = onnx_model_path,
                label_path = label_path,
                prob_threshold = confThreshold,
                iou_threshold=nmsThreshold)

def infer(img):
    '''
    推理
    Args: 
        img: 待检测图像
    Returns
        det_results: 检测结果列表，格式为：
                    [{"classid":class_id,
                      "classname":class_name,
                      "confidence":conf,
                      "box":[xmin,ymin,xmax,ymax]},
                      ...,
                      ...]
    '''
    det_results = net.detect_onnx(img,True)
    return det_results
    

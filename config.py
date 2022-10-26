# 通用参数设置
r = 1               # 缩放比例
score_th = 0.8        # 置信度阈值，用于划分文字和图标
merge_box = False     # 是否合并文本检测框
use_mp = True         # 是否使用多线程
process_num = 10      # 线程数


# OCR参数

infer_h = 32
batch = 1
#----------------------------------- paddle系列 -------------------------------------------#
# pprec_v2 官方
model_path = "models/pprec_v2.onnx"
in_names = "x"
out_names = ["save_infer_model/scale_0.tmp_1"]
keys_txt_path = "models/ppocr_keys_v1.txt"

# 文本图标检测
# 模型路径（pp-yolo-E）
# det_model_path = "models/ppyoloe_crn_s_300e_coco_text_ico_1021_350e_960.onnx"       # 推理尺寸960x960
# det_model_path = "models/ppyoloe_crn_s_300e_coco_text_ico_1021_350e_768.onnx"     # 推理尺寸768x768
det_model_path = "models/ppyoloe_crn_s_300e_coco_text_ico_1021_350e.onnx"     # 推理尺寸640x640

# 模型路径（pp-yolo-E+）
# det_model_path = "models/ppyoloe_plus_crn_s_80e_coco_text_ico_1021_300e.onnx"     # 推理尺寸640x640
# 类别列表
label_path = "det/label_list_text_ico.txt"
# 显示置信度阈值
confThreshold = 0.5
# nms阈值
nmsThreshold = 0.3
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
# model_path = "models/pprec_v2.onnx"
# in_names = "x"
# out_names = ["save_infer_model/scale_0.tmp_1"]
# keys_txt_path = "models/ppocr_keys_v1.txt"

# pprec_v3 官方
# infer_h = 48
# in_names = "x"
# out_names = ["softmax_5.tmp_0"]
# keys_txt_path = "models/ppocr_keys_v1.txt"

#ppocr_v2_me 自己训练
model_path = "models/pprec_v2_me.onnx"
in_names = "x"
out_names = ["softmax_0.tmp_0"]
keys_txt_path = "models/keys.txt"

#----------------------------------- pytorch系列 -------------------------------------------#

# pytorch-crnn
# model_path = "models/crnn_lite_lstm.onnx"
# in_names = "input"
# out_names = ["out"]

# paddel对应的官方pytorchOCR
# model_path = "models/rec.onnx"
# in_names = "in"
# out_names = ["out"]

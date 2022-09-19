import crnn.model as model


# 通用参数设置
r = 3/4               # 缩放比例
score_th = 0.8        # 置信度阈值，用于划分文字和图标
merge_box = False     # 是否合并文本检测框
use_mp = True         # 是否使用多线程
process_num = 10      # 线程数


# OCR参数设置

# pprec
ocr_handle = model.OcrHandle("models/pprec_v2.onnx",32,1)           # 初始化OCR识别模型
# ocr_handle = model.OcrHandle("models/pprec_v3.onnx",48,1)
ocr_predict = ocr_handle.PPRecWithBox

# pytorch-crnn
# ocr_handle = model.OcrHandle("models/crnn_lite_lstm.onnx",32,1)
# ocr_predict = ocr_handle.crnnRecWithBox

# import crnn.model as model

# OCR参数设置
infer_h = 32
batch = 1
#----------------------------------- paddle系列 -------------------------------------------#
# pprec_v2 官方
# ocr_handle = model.OcrHandle("models/pprec_v2.onnx",32,1)           # 初始化OCR识别模型
model_path = "models/pprec_v2.onnx"
in_names = "x"
out_names = ["save_infer_model/scale_0.tmp_1"]
keys_txt_path = "models/ppocr_keys_v1.txt"

# pprec_v3 官方
# ocr_handle = model.OcrHandle("models/pprec_v3.onnx",48,1)
# in_names = "x"
# out_names = ["softmax_5.tmp_0"]
# keys_txt_path = "models/ppocr_keys_v1.txt"

#ppocr_v2_me 自己训练
# ocr_handle = model.OcrHandle("models/pprec_v2_me.onnx",32,1)  
# in_names = "x"
# out_names = ["softmax_0.tmp_0"]
# keys_txt_path = "models/keys.txt"

# ocr_predict = ocr_handle.PPRecWithBox

#----------------------------------- pytorch系列 -------------------------------------------#

# pytorch-crnn
# ocr_handle = model.OcrHandle("models/crnn_lite_lstm.onnx",32,1)
# ocr_predict = ocr_handle.crnnRecWithBox
# in_names = "input"
# out_names = ["out"]

# paddel对应的官方pytorchOCR
# ocr_handle = model.OcrHandle("models/rec.onnx",32,1)
# ocr_predict = ocr_handle.PPRecWithBox
# in_names = "in"
# out_names = ["out"]
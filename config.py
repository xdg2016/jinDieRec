import os



filt_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(filt_path) + os.path.sep + ".")

# dbnet 参数
dbnet_max_size = 3000 #长边最大长度
pad_size = 0 #检测是pad尺寸，有些文档文字充满整个屏幕检测有误，需要pad
model_path = os.path.join(father_path, "models/DBnet.onnx")
# model_path = os.path.join(father_path, "models/PPOCR_det.onnx")  # 长边960

# crnn参数
crnn_lite = True
is_rgb = True
batch_num = 8
crnn_model_path = os.path.join(father_path, "models/crnn_lite_lstm.onnx")



# angle
angle_detect = False
angle_detect_num = 10
angle_net_path = os.path.join(father_path, "models/angle_net.onnx")


max_post_time = 100 # ip 访问最大次数

from crnn.keys import alphabetChinese as alphabet


white_ips = [] #白名单

version = 'api/v1'

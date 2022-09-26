import time
import numpy as np
from .keys import alphabetChinese as alphabet
import os,sys
import onnxruntime as rt
from .util import strLabelConverter, resizeNormalize

converter = strLabelConverter(''.join(alphabet))

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


class PPrecHandle:
    '''
    百度的识别模型
    '''
    def __init__(self, model_path,keys_txt_path,in_names,out_names):
        self.keys_txt_path = keys_txt_path
        self.in_names = in_names
        self.out_names = out_names
        
        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = 16
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = rt.InferenceSession(model_path,sess_options= sess_options)
 
        self.character_str = []
        character_dict_path = keys_txt_path
        use_space_char = True
        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        th = 0.15
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token
            # 不使用置信度阈值过滤
            # char_list = [
            #     self.character[text_id]
            #     for idx,text_id in enumerate(text_index[batch_idx][selection])
            # ]
            # 使用置信度阈值过滤
            char_list = [
                self.character[text_id]
                for idx,text_id in enumerate(text_index[batch_idx][selection])
                if text_prob[batch_idx][selection][idx] > th
            ]

            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank

    def predict_rbg(self, im):
        """
        预测
        """
        t1 = time.time()
        # pprec v2
        # preds = self.sess.run(["save_infer_model/scale_0.tmp_1"], {"x": im.astype(np.float32)}) # 12ms
        # pprec v3
        # preds = self.sess.run(["softmax_5.tmp_0"], {"x": im.astype(np.float32)})  
        # paddle对应的pytorchOCR版本 
        # preds = self.sess.run(["out"], {"in": im.astype(np.float32)})  
        # rec_me
        preds = self.sess.run(self.out_names, {self.in_names: im.astype(np.float32)}) # 12ms
        # print("run cost: ",time.time()-t1)

        preds = preds[0]

        # preds = softmax(preds)
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)

        return text


class CRNNHandle:
    def __init__(self, model_path,keys_txt_path,in_names,out_names):
        self.keys_txt_path = keys_txt_path
        self.in_names = in_names
        self.out_names = out_names
        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = 16
        sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = rt.InferenceSession(model_path,sess_options= sess_options)
        # self.sess = rt.InferenceSession(model_path)

    def predict_rbg(self, im):
        """
        预测
        """
        preds = self.sess.run(self.out_names, {self.in_names: im.astype(np.float32)})
        preds = preds[0]

        # pytorch crnn
        length  = preds.shape[0]
        batch = preds.shape[1]

        if batch > 1:
            batch_preds = []
            for i in range(batch):
                batch_pred = preds[:,i,:].reshape(length,-1)
                batch_pred = softmax(batch_pred)

                # 取出最大索引和概率最大值
                preds_idxs = batch_pred.argmax(axis=1)
                preds_probs = batch_pred.max(axis=1)
                
                text, prob = converter.decode(preds_idxs,preds_probs, length, raw=False)
                batch_preds.append((text,prob))
            return batch_preds
        else:
            preds = preds.reshape(length,-1)
            preds = softmax(preds)

            # 取出最大索引和概率最大值
            preds_idxs = preds.argmax(axis=1)
            preds_probs = preds.max(axis=1)
        
            text, prob = converter.decode(preds_idxs,preds_probs, length, raw=False)

            return [(text, prob)] 

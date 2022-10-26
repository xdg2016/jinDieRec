# PageItemRec
通用界面元素分析,包括文字和图标，思路流程如下：

![页面元素（文字、图标）检测_第二期](./imgs/页面元素（文字、图标）检测_第二期.png)

## 算法思路

1、用目标检测网络出文本和图标集合。

2、对文本框集合调用OCR识别模块做文本内容识别。

3、返回识别后的文本框和图标框集合。



## 版本更新

- 2022.09.24发布 pageitemrec v1.0

  - 检测算法：Robert算子边缘检测，得到二值图
  - 连通域检测，去除大框和横竖长线条，只留下文字和图标
  - 闭运算，合并文字间隙
  - 外轮廓检测，计算轮廓外接矩形
  - OCR识别：使用pprec_v2算法，多线程+onnxruntime推理加速
  - 根据OCR识别置信度，区分是文字还是图标，返回文字和图标集合。

  **耗时统计**

  | 测试集 | 缩放比例 | 非线程池 |
  | ------ | -------- | -------- |
  | 金蝶   | 1        | 0.186 s  |
  | chrome | 3/4      | 0.391 s  |

- 2022.10.26发布 pageitemrec v2.0

  优化内容：用深度学习方法替代传统检测算法+分类的方法，一次性可以区分文本和图标，以及对应的位置，且可以适应任意场景的文本和图标检测。

  - 检测算法：pp-yoloe算法，一次性检测出图标和文本。
  - OCR识别：pprec_v2算法，多线程+onnxruntime推理加速。

  **耗时统计**

  | 测试集             | 耗时（onnx+多线程） |
  | ------------------ | ------------------- |
  | 71张各场景混合图片 | 0.272s              |


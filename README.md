## 脚本功能&核心依赖


| 脚本文件               | 核心依赖库  | 特点说明                                               |
|------------------------|-------------|--------------------------------------------------------|
| `dlib_face_detector.py`  | dlib        | 基于dlib实现人脸检测，安装需编译环境                   |
| `insightface_det.py`     | insightface | 基于insightface实现人脸检测                            |
| `insightface_compare.py` | insightface | 基于insightface实现人脸对比，需要一张对比照片 `test.jpg` |


## insightface 说明

模型下载后的存放位置：`C:\Users\<username>\.insightface\models\buffalo_l`

```shell
uv pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/
```

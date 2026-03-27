# python_Keras

基于 Flask + YOLO + InsightFace 的人脸识别 Web 应用。

## 功能

- 上传视频/图片，自动进行人脸检测与识别
- 使用 YOLOv8 做目标检测，InsightFace 做人脸特征提取与比对
- MediaPipe 辅助关键点检测
- Web 界面展示识别结果，支持日志记录

## 技术栈

| 组件 | 库 |
|------|----|
| Web 框架 | Flask |
| 目标检测 | YOLOv8 (ultralytics) |
| 人脸识别 | InsightFace |
| 关键点检测 | MediaPipe |
| 图像处理 | OpenCV, NumPy |

## 目录结构

```
├── app.py              # 基础版本
├── app_improved.py     # 改进版本（推荐）
├── src/                # 核心模块
├── static/             # 静态资源（上传/结果图片）
│   ├── uploads/
│   ├── results/
│   └── faces/
└── templates/          # HTML 模板
```

## 快速开始

```bash
pip install flask opencv-python mediapipe ultralytics insightface numpy
python app_improved.py
# 访问 http://localhost:5000
```

## 环境要求

- Python 3.8+
- CUDA（可选，有 GPU 时推理更快）

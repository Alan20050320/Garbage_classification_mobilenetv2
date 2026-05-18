
# Garbage_classification_mobilenetv2
基于 MobileNetV2 的垃圾分类图像识别项目

## 项目说明
本项目使用 PyTorch 框架，基于 MobileNetV2 预训练模型做迁移学习微调，实现生活垃圾图像分类。
全程在 **Mac M1 芯片** 设备上完成训练与推理，支持单张/批量图片分类，可输出预测结果与可视化报告。

## 训练硬件与环境
- 设备：MacBook Air（M1 芯片）
- 加速：PyTorch MPS 加速
- 依赖：torch、torchvision、matplotlib

## 模型效果示例
模型可对输入图片自动分类，并输出类别与置信度，同时生成对比可视化结果。
![预测效果示例](assets/predict_results/11_prediction.jpg)
![预测效果示例](assets/predict_results/15_prediction.jpg)
![预测效果示例](assets/predict_results/14_prediction.jpg)
![预测效果示例](assets/predict_results/16_prediction.jpg)
![预测效果示例](assets/predict_results/18_prediction.jpg)




## 快速使用
1. 克隆项目并进入目录
```bash
git clone https://github.com/Alan20050320/Garbage_classification_mobilenetv2.git
cd Garbage_classification_mobilenetv2
```

2. 安装依赖
```bash
pip install torch torchvision matplotlib
```

3. 训练模型
```bash
python train.py
```
脚本会自动启用 MPS 加速，按 8:2 划分训练/验证集，训练完成后保存最优模型 `best_model_v2.pth`。

4. 图片预测
```bash
# 默认路径批量预测
python predict.py

# 自定义图片文件夹批量预测
python predict.py 你的图片文件夹路径
```
预测结果自动保存到 `predict_results` 目录，包含可视化图片与分类报告。

## 支持分类类别
1. 纸板（cardboard）
2. 玻璃（glass）
3. 金属（metal）
4. 纸张（paper）
5. 塑料（plastic）
6. 其他垃圾（trash）

## 文件说明
- `train.py`：训练脚本，含数据增强、分层解冻、标签平滑、早停、余弦退火学习率
- `predict.py`：预测脚本，支持单张/批量预测、结果可视化、报告生成
- `best_model_v2.pth`：训练好的模型权重文件
- `predict_results/`：预测结果输出目录

## 备注
项目适配 Mac M1 设备，无需独立 GPU 即可完成训练与推理，代码可直接复现运行。

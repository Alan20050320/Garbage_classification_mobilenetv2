import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from datetime import datetime

# 设置中文字体（如果你需要中文标签的话）
# 如果不需要中文，可以注释掉这部分
try:
    # 尝试使用Mac已有的中文字体
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'Heiti SC']
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass  # 如果字体设置失败，继续使用默认字体

# 1. 类别名称(与训练时一致)
class_names = ['cardboard(纸板)', 'glass(玻璃)', 'metal(金属)',
               'paper(纸张)', 'plastic(塑料)', 'trash(其他)']

# 2. 加载模型
print("正在加载模型...")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

# 检查模型文件是否存在
model_path = 'best_model_v2.pth'
if not os.path.exists(model_path):
    print(f"错误: 找不到模型文件 '{model_path}'")
    print("请先运行训练脚本 train.py 来训练模型")
    exit(1)

model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
print("模型加载完成!")

# 3. 图片预处理(与验证集一致)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict_image(image_path, save_results=True, output_dir="predict_results"):
    """
    预测单张图片并保存结果

    参数:
    - image_path: 图片文件路径
    - save_results: 是否保存预测结果图片
    - output_dir: 结果保存目录

    返回:
    - 预测结果字典
    """
    try:
        # 打开图片
        img = Image.open(image_path).convert('RGB')
        original_size = img.size

        # 预处理
        tensor = transform(img).unsqueeze(0)  # 增加batch维度

        # 预测
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]  # 转为概率
            pred = probs.argmax().item()
            confidence = probs[pred].item()

        # 获取所有类别的概率
        all_probs = {class_names[i]: float(probs[i].item()) for i in range(len(class_names))}

        # 创建结果字典
        result = {
            'image_path': image_path,
            'prediction': class_names[pred],
            'confidence': confidence,
            'all_probabilities': all_probs,
            'original_size': f"{original_size[0]}x{original_size[1]}"
        }

        # 保存带预测结果的图片
        if save_results:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 创建带标注的图片
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 左侧: 原始图片
            ax1.imshow(img)
            ax1.set_title(f"原始图片\n{Path(image_path).name}", fontsize=12)
            ax1.axis('off')

            # 右侧: 预测结果和概率分布
            # 显示类别和概率
            # 使用Set3颜色映射，如果不存在则使用tab20c
            try:
                colors = plt.cm.Set3(range(len(class_names)))
            except AttributeError:
                colors = plt.cm.tab20c(range(len(class_names)))

            y_pos = list(range(len(class_names)))
            probs_list = [all_probs[cn] for cn in class_names]

            bars = ax2.barh(y_pos, probs_list, color=colors)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(class_names, fontsize=10)
            ax2.set_xlabel('概率')
            ax2.set_title(f'预测结果: {class_names[pred]}\n置信度: {confidence:.2%}')
            ax2.set_xlim(0, 1)

            # 在柱状图上显示概率值
            for i, (bar, prob) in enumerate(zip(bars, probs_list)):
                width = bar.get_width()
                ax2.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{prob:.2%}', ha='left', va='center', fontsize=9)

            # 高亮预测结果
            bars[pred].set_color('red')

            plt.tight_layout()

            # 保存图片
            output_filename = f"{Path(image_path).stem}_prediction.jpg"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            result['output_path'] = output_path

        return result

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return None


def batch_predict(folder_path, output_dir="predict_results"):
    """
    批量预测文件夹中的所有图片

    参数:
    - folder_path: 包含图片的文件夹路径
    - output_dir: 结果保存目录
    """
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return None

    # 获取所有图片文件
    image_files = []
    for ext in image_extensions:
        for file_path in Path(folder_path).glob(f"*{ext}"):
            image_files.append(file_path)
        for file_path in Path(folder_path).glob(f"*{ext.upper()}"):
            image_files.append(file_path)

    if not image_files:
        print(f"在文件夹 '{folder_path}' 中没有找到支持的图片文件")
        print(f"支持的格式: {', '.join(image_extensions)}")
        return None

    print(f"找到 {len(image_files)} 张图片")
    print("开始批量预测...")

    # 批量预测
    results = []
    successful = 0
    failed = 0

    for i, img_path in enumerate(image_files, 1):
        print(f"处理图片 {i}/{len(image_files)}: {img_path.name}")

        result = predict_image(str(img_path), save_results=True, output_dir=output_dir)

        if result:
            results.append(result)
            successful += 1
            print(f"  ✓ 预测结果: {result['prediction']}, 置信度: {result['confidence']:.2%}")
        else:
            failed += 1
            print(f"  ✗ 处理失败")

    # 生成报告
    if results:
        generate_report(results, output_dir, total=len(image_files), successful=successful, failed=failed)

    return results


def generate_report(results, output_dir, total, successful, failed):
    """
    生成预测结果报告
    """
    if not results:
        print("没有生成任何预测结果")
        return

    # 创建数据框以便分析
    data = []
    for res in results:
        row = {
            '图片文件': Path(res['image_path']).name,
            '预测结果': res['prediction'],
            '置信度': f"{res['confidence']:.2%}",
            '结果图片': Path(res.get('output_path', '')).name if res.get('output_path') else 'N/A',
            '原始尺寸': res['original_size']
        }
        data.append(row)

    # 保存为TXT（易读格式）
    txt_path = os.path.join(output_dir, "predictions_report.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("垃圾分类预测结果报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理图片总数: {total}\n")
        f.write(f"成功预测: {successful}\n")
        f.write(f"失败: {failed}\n\n")

        f.write("预测结果统计:\n")
        f.write("-" * 40 + "\n")
        for class_name in class_names:
            count = sum(1 for res in results if res['prediction'] == class_name)
            if count > 0:
                percentage = count / successful * 100
                f.write(f"{class_name}: {count} 张 ({percentage:.1f}%)\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("详细预测结果:\n")
        f.write("=" * 60 + "\n\n")

        for i, res in enumerate(results, 1):
            f.write(f"{i}. {Path(res['image_path']).name}\n")
            f.write(f"   预测结果: {res['prediction']}\n")
            f.write(f"   置信度: {res['confidence']:.2%}\n")

            # 显示前3个最可能的类别
            all_probs = res['all_probabilities']
            sorted_probs = sorted(all_probs.items(),
                                  key=lambda x: x[1], reverse=True)[:3]
            f.write(f"   前3个可能类别:\n")
            for class_name, prob in sorted_probs:
                f.write(f"     - {class_name}: {prob:.2%}\n")

            output_path = res.get('output_path', '')
            if output_path:
                f.write(f"   结果保存: {Path(output_path).name}\n")
            f.write("\n")

    print(f"\n{'=' * 60}")
    print("批量预测完成!")
    print(f"{'=' * 60}")
    print(f"处理图片总数: {total}")
    print(f"成功预测: {successful}")
    print(f"失败: {failed}")
    print(f"\n结果保存目录: {output_dir}")
    print(f"详细报告: {txt_path}")

    # 显示统计信息
    print(f"\n预测结果统计:")
    print("-" * 40)
    for class_name in class_names:
        count = sum(1 for res in results if res['prediction'] == class_name)
        if count > 0:
            percentage = count / successful * 100
            print(f"{class_name}: {count} 张 ({percentage:.1f}%)")


def main():
    """
    主函数
    """
    print("垃圾分类图片批量预测工具")
    print("=" * 50)

    # 设置图片文件夹路径
    # 你可以修改这里为你的文件夹路径
    image_folder = "/Users/macbookair/Desktop/code/Deeplearning/test_photo/garbage_photo"

    # 或者通过命令行参数获取
    import sys
    if len(sys.argv) > 1:
        image_folder = sys.argv[1]

    # 设置结果保存目录
    output_dir = "predict_results"

    # 执行批量预测
    results = batch_predict(image_folder, output_dir)

    if results:
        print(f"\n所有处理完成! 共处理了 {len(results)} 张图片。")
    else:
        print("\n处理完成，但没有生成任何结果。")


if __name__ == "__main__":
    main()
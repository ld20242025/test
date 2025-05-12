import paddle
import paddlehub as hub
from PIL import Image
import matplotlib.pyplot as plt

# 加载YOLOv3模型
yolo = hub.load('yolov3_resnet50_vd_ssld')

# 加载要检测的图片
image_path = 'test.png'  # 替换为你自己的图片路径
image = Image.open(image_path)

# 使用YOLO模型进行预测
result = yolo.object_detection(images=[image])

# 打印检测结果
print("检测到的物体：")
for idx, obj in enumerate(result[0]['data']):
    print(f"物体 {idx+1}: 类别: {obj['label']}, 置信度: {obj['confidence']:.2f}, 边界框: {obj['left']}, {obj['top']}, {obj['right']}, {obj['bottom']}")

# 绘制图片和检测结果
plt.imshow(image)
ax = plt.gca()

# 绘制检测到的物体的边界框
for obj in result[0]['data']:
    ax.add_patch(plt.Rectangle((obj['left'], obj['top']),
                               obj['right'] - obj['left'],
                               obj['bottom'] - obj['top'],
                               linewidth=2, edgecolor='r', facecolor='none'))

# 保存结果图像到文件
output_image_path = 'detected_image.jpg'  # 设置保存路径
plt.axis('off')  # 不显示坐标轴
plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)

print(f"检测结果已保存为: {output_image_path}")

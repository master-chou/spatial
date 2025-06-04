import json
from PIL import Image
import os
# 加载你的 JSON 文件
with open("output.json") as f:
    data = json.load(f)

image_folder = 'output_images'
# 转换格式
converted_data = []
for item in data:
    # 读取图像尺寸
    image_path = os.path.join(image_folder, item["image_1"])
    with Image.open(image_path) as img:
        width, height = img.size

    # 构建新的格式
    new_item = {
        "id": item["idx"],
        "image_info": {
            "file_path": item["image_1"],
            "height": height,
            "width": width,
        },
        "text_q": item["prompt"]+'\nAnswer the question using only the options (A) or (B).',
        "qa_info": "",  # 留空或填充默认值
        "conversations": [
            {"role": "user", "value": item["prompt"]+'\nAnswer the question using only the options (A) or (B).'},
            {"role": "assistant", "value": item["answer"]},
        ],
        "bbox": [],  # 如果没有 bbox 数据，留空
        "rle": [],   # 如果没有 rle 数据，留空
    }
    converted_data.append(new_item)

# 保存转换后的数据
with open("converted_output.json", "w") as f:
    json.dump(converted_data, f, indent=4)
import json
import re

def convert_to_natural_statement(sentence):
    """
    将 "Is the X Y? yes." 或 "Is the X Y? no." 形式的句子转换为陈述句：
    - "Is the car beneath the cat? yes." → "The car is beneath the cat."
    - "Is the car beneath the cat? no." → "The car is not beneath the cat."
    """
    match = re.match(r"Is the (\w+) (.+?)\? (yes|no)\.", sentence)
    if match:
        subject, condition, answer = match.groups()
        if answer.lower() == "yes":
            return f"The {subject} is {condition}."
        else:
            return f"The {subject} is not {condition}."
    return sentence  # 如果不符合格式，则返回原句

def process_json_file(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        if "sentence_A" in entry:
            entry["sentence_A"] = convert_to_natural_statement(entry["sentence_A"])
        if "sentence_B" in entry:
            entry["sentence_B"] = convert_to_natural_statement(entry["sentence_B"])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# 示例调用
input_json = "caption.json"  # 替换为你的 JSON 文件路径
output_json = "recaption.json"
process_json_file(input_json, output_json)

print(f"处理完成，转换后的 JSON 文件已保存至 {output_json}")

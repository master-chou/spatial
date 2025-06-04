import json
import re

def convert_prompt(prompt):
    match = re.match(r"^(.*?)\nSelect from the following choices\.\n\(A\) (.*?)\n\(B\) (.*?)$", prompt, re.DOTALL)
    if not match:
        return None, None
    
    question, option_a, option_b = match.groups()
    return f"{question} {option_a}.", f"{question} {option_b}."

def process_json_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        sentence_a, sentence_b = convert_prompt(item.get("prompt", ""))
        if sentence_a and sentence_b:
            item["sentence_A"] = sentence_a
            item["sentence_B"] = sentence_b
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# 示例调用
input_json_file = "output.json"  # 替换为你的 JSON 文件路径
output_json_file = "caption.json"
process_json_file(input_json_file, output_json_file)

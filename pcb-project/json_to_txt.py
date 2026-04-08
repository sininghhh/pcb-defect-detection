import json
import os

classes = ["missing_hole"]

def convert(json_path, txt_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    h = data['imageHeight']
    w = data['imageWidth']

    with open(txt_path, 'w', encoding='utf-8') as f:
        for shape in data['shapes']:
            label = shape['label']
            if label not in classes:
                continue

            cls_id = classes.index(label)

            points = shape['points']
            if len(points) != 2:
                continue

            (x1, y1), (x2, y2) = points

            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = abs(x2 - x1) / w
            height = abs(y2 - y1) / h

            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

folder = "D:/deeppcb/missinghole/raw_data"

for file in os.listdir(folder):
    if file.endswith(".json"):
        json_path = os.path.join(folder, file)
        txt_path = os.path.join(folder, file.replace(".json", ".txt"))
        convert(json_path, txt_path)

print("转换完成！")


"""
PCB 결함 데이터셋에 대한 EDA(탐색적 분석)
- 클래스별 데이터 수, 라벨 포맷, 이미지+바운딩박스 시각화, 추가 통계까지 모두 자동화
"""

import os
import xml.etree.ElementTree as ET
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np

BASE = "../../data/PCB_DATASET"
ANNOT = os.path.join(BASE, "Annotations")
IMG = os.path.join(BASE, "images")


def collect_data_info():
    class_dirs = sorted(os.listdir(ANNOT))
    class_counts = {}
    object_counts = Counter()
    bbox_areas = []

    for cls in class_dirs:
        ann_dir = os.path.join(ANNOT, cls)
        # img_dir = os.path.join(IMG, cls)
        anns = [f for f in os.listdir(ann_dir) if f.endswith(".xml")]
        class_counts[cls] = len(anns)
        for ann in anns:
            ann_path = os.path.join(ann_dir, ann)
            tree = ET.parse(ann_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                label = obj.find("name").text
                object_counts[label] += 1
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                area = (xmax - xmin) * (ymax - ymin)
                bbox_areas.append(area)
    return class_counts, object_counts, bbox_areas


def print_basic_stats(class_counts, object_counts):
    print("\n1. [클래스별(폴더) 이미지/라벨 파일 수]")
    for cls, cnt in class_counts.items():
        print(f" - {cls}: {cnt}")

    print("\n2. [라벨(클래스)별 오브젝트 수]")
    for label, cnt in object_counts.items():
        print(f" - {label}: {cnt}")


def print_label_format_info():
    print("""
3. [라벨 포맷 (Pascal VOC XML)]
- 각 annotation(.xml) 파일의 object > name(클래스명), bndbox(xmin, ymin, xmax, ymax)
- VOC 포맷이므로 다양한 파이썬 도구에서 바로 활용 가능
- 공식문서: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
- 실전 파싱 예시: https://towardsdatascience.com/coco-data-format-for-object-detection-a4c5eaf518c5
""")


def show_sample(class_name):
    ann_dir = os.path.join(ANNOT, class_name)
    img_dir = os.path.join(IMG, class_name)
    # 가장 첫 annotation 고름
    ann_file = sorted([f for f in os.listdir(ann_dir) if f.endswith(".xml")])[0]
    ann_path = os.path.join(ann_dir, ann_file)
    tree = ET.parse(ann_path)
    root = tree.getroot()
    imgname = root.find("filename").text
    boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        box = [
            int(bnd.find("xmin").text),
            int(bnd.find("ymin").text),
            int(bnd.find("xmax").text),
            int(bnd.find("ymax").text),
        ]
        boxes.append(box)
    # 이미지 표시
    img_path = os.path.join(img_dir, imgname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지 파일을 열 수 없습니다: {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for xmin, ymin, xmax, ymax in boxes:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
    plt.figure(figsize=(10, 6))
    plt.title(f"Sample: {class_name}, {imgname}")
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def show_bbox_area_hist(bbox_areas):
    plt.hist(np.sqrt(bbox_areas), bins=30)
    plt.title("bounding box area(root) histogram")
    plt.xlabel("sqrt(area)")
    plt.ylabel("count")
    plt.show()


def main():
    class_counts, object_counts, bbox_areas = collect_data_info()
    print_basic_stats(class_counts, object_counts)
    print_label_format_info()
    # 샘플 한 장 시각화(Missing_hole, 다른 클래스로 변경 가능)
    show_sample("Missing_hole")
    # 바운딩박스 면적 분포
    show_bbox_area_hist(bbox_areas)


if __name__ == "__main__":
    main()

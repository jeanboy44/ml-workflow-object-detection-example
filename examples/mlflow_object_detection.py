from ml_object_detector import load, load_image, predict


def main() -> None:
    model = load("models:/exp05_yolo/Production")
    image = load_image("experiments/sample_data/cat_01.jpg")
    result = predict(model, image, threshold=0.25)
    for detection in result.detections:
        print(detection)


if __name__ == "__main__":
    main()

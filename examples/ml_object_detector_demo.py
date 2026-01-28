from ml_object_detector import list_models, load, load_image, predict


def main() -> None:
    print("Available Models:")
    for model_info in list_models():
        print(model_info)
    model = load("study.object_detection.yolo26n_pretrained_onnx", model_version="3")
    image = load_image("data/sample_data/cat_01.jpg")
    result = predict(model, image, threshold=0.25)
    for detection in result.detections:
        print(detection)


if __name__ == "__main__":
    main()

from ultralytics import YOLO
import cv2


def detect_objects(image_path, output_path):
    # Загружаем предобученную модель YOLOv8 для сегментации.
    model = YOLO('model/yolov8n-seg.pt')

    # Выполняем предсказание на изображении.
    results = model(image_path)

    # Получаем аннотированное изображение из первого результата.
    annotated_image = results[0].plot()

    # Сохраняем результат.
    cv2.imwrite(output_path, annotated_image)
    print(f"Результат сохранён в {output_path}")


def main():
    image_path = "test_input/image1.jpg"  # Путь к входному изображению
    output_path = "output/output.jpg"  # Путь для сохранения результата
    detect_objects(image_path, output_path)


if __name__ == "__main__":
    main()

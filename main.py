from collections import Counter
from ultralytics import YOLO
import cv2


class ObjectDetector:
    def __init__(self):
        self.image_path = None
        self.output_path = None

    @staticmethod
    def detect_objects(image_path, output_path):
        # Загружаем пред-обученную модель YOLOv8 для сегментации
        # Модель из: https://github.com/matterport/Mask_RCNN
        model = YOLO('model/yolov8n-seg.pt')

        # Выполняем предсказание на изображении
        results = model(image_path)

        # Получаем аннотированное изображение из первого результата
        annotated_image = results[0].plot()

        # Сохраняем результат
        cv2.imwrite(output_path, annotated_image)
        print(f"Результат сохранён в {output_path}")

        # Подсчет объектов
        boxes = results[0].boxes
        if boxes is not None and len(boxes.cls) > 0:
            class_ids = boxes.cls.cpu().numpy()
            class_names = [model.names[int(cls_id)] for cls_id in class_ids]

            counts = Counter(class_names)

            print("Обнаруженные объекты:")
            for class_name, cnt in counts.items():
                print(f"{class_name}: {cnt}")
        else:
            print("Объекты не обнаружены")


def main():
    image_path = "test_input/image1.jpg"  # Путь к входному изображению
    output_path = "output/output1.jpg"  # Путь для сохранения результата
    object_detector = ObjectDetector()
    object_detector.detect_objects(image_path, output_path)


if __name__ == "__main__":
    main()

import os
import csv
import cv2
import logging
from ultralytics import YOLO
from collections import Counter


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )


class SetupModel:
    def __init__(self):
        # Модель из: https://github.com/matterport/Mask_RCNN
        self.model = YOLO('model/yolov8n-seg.pt')

    def get(self):
        return self.model

    def __repr__(self):
        return str(self.model)


class ObjectDetector:
    def __init__(self):
        pass

    @staticmethod
    def process_image(model, image_path, output_path, csv_path=None):
        # Выполняем предсказание на изображении
        logging.info(f"Обработка изображения: {image_path}")
        results = model(image_path)

        # Получаем аннотированное изображение из первого результата
        annotated_image = results[0].plot()

        # Сохраняем результат
        cv2.imwrite(output_path, annotated_image)
        logging.info(f"Сохранено: {output_path}")

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
            counts = {}

        base_name = os.path.basename(image_path)

        if csv_path:
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["image"] + list(model.names.values())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                row = {"image": base_name}
                # Инициализируем все классы нулем
                for class_id, class_name in model.names.items():
                    row[class_name] = counts.get(class_name, 0)
                writer.writerow(row)
            logging.info(f"Статистика для {base_name} сохранена в {csv_path}")

        return base_name, counts

    def process_directory(self, model, input_folder, output_folder, csv_path):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Открываем CSV-файл для записи статистики
        with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["image"] + list(model.names.values())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        # Проходим по всем файлам в папке
        for filename in os.listdir(input_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_file = os.path.join(input_folder, filename)
                output_file = os.path.join(output_folder, filename)
                self.process_image(model, image_file, output_file, csv_path)

        logging.info(f"Статистика сохранена в {csv_path}")


def main():
    setup_logging()

    # Для обработки одного изображения
    # image_path = "test_input/image1.jpg"  # Путь к входному изображению
    # output_path = "output/output1.jpg"  # Путь для сохранения результата

    # Для обработки директории
    folder_path = "test_input"
    output_path = "output"

    model = SetupModel().get()
    object_detector = ObjectDetector()

    # object_detector.process_image(model, image_path, output_path, 'result.csv')

    object_detector.process_directory(model, folder_path, output_path, 'result.csv')


if __name__ == "__main__":
    main()

# Instance Segmentation with CNN ai

Детекция и сегментация объектов на изображениях с использованием CNN моделей.

## Список проверенных моделей
- yolov8n
- detectron2

## Библиотеки

- [ultralytics](https://github.com/ultralytics/ultralytics)
- opencv-python

## Структура проекта

```
├── models/             # Директория с тестируемыми моделями  
├── notebooks/          # Директория с colab ноутбуками          
├── main.py             # Основной скрипт для выполнения детекции и сегментации
├── test_data/          # Директория с входными изображениями (пример) / Теперь используется coco датасет
├── output/             # Директория для сохранения аннотированных изображений / Теперь используется coco датасет
├── result.csv          # CSV-файл со статистикой (будет создан после выполнения скрипта) / Теперь используется coco датасет
├── coco.yaml           # Файл с важной онформацией для coco датасета
└── README.md           # Этот файл
```

## Установка
1. Скопируйте репозиторий
```sh
git clone https://github.com/pOsdas/ai-object-vision.git
cd [ваше имя]
```
2. Создайте и активируйте свою виртуальную среду 
    ### Windows:
    ```sh
    python -m venv venv
    source venv/Scripts/activate
    ```
    ### Linux/MacOS:
    ```sh
    python -m venv venv
    source venv/bin/activate
    ```
3. Установить зависимости:
```commandline
pip install -r req.txt
```

## License
Распространяется под лицензией MIT.\
Дополнительная информация в файле `LICENSE.txt`.

## Author
Автор: pOsdas


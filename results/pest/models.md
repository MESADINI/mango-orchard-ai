| Model | Params | Epochs | Val mAP50-95 | Test mAP50 | Test mAP50-95 | Test P / R | Test AP50-95 Fruit Fly | Test AP50-95 Mealy Bug | Test AP50-95 Scale Insect |
|---|---|---|---|---|---|---|---|---|---|
| yolov8n | 3.0 M | 50 | 0.533 | 0.743 | 0.519 | 0.796 / 0.765 | 0.539 | 0.118 | 0.899 |
| yolo11n | 2.6 M | 50 | 0.537 | 0.744 | 0.520 | 0.779 / 0.775 | 0.542 | 0.121 | 0.896 |

Single run per model (seed 42). Classes are confounded with their source (one non-mango source per class), so these numbers do not estimate performance on mango orchard images.

# Vehicle Detection using YOLOv7  

This repository contains an implementation of real-time **vehicle detection** using **YOLOv7**, fine-tuned on a **custom vehicle dataset** from Kaggle. The model detects and localizes vehicles in images and video streams, making it suitable for applications like **traffic monitoring, autonomous vehicles, and smart surveillance systems**.  

## 📌 Dataset  
The model is trained on the **[Vehicle Dataset for YOLO](https://www.kaggle.com/datasets/nadinpethiyagoda/vehicle-dataset-for-yolo)** from Kaggle. The dataset includes annotated images of vehicles for robust detection.  

## 🚀 Features  
- **Fine-tuned YOLOv7** for **optimized vehicle detection**  
- **Supports image & video inference**  
- **Real-time performance with high accuracy**  
- **Compatible with ONNX, TensorRT, and CoreML** for deployment  

## 📂 Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/MannShah0312/Vehicle-Detection-using-YoloV7.git
cd vehicle-detection-yolov7
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

## 🏋️ Training the Model  
To train YOLOv7 on the **vehicle dataset**, use:  
```bash
python train.py --workers 8 --device 0 --batch-size 16 --data data/vehicle.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights yolov7_training.pt --name yolov7-vehicle --hyp data/hyp.vehicle.yaml
```

## 🛠️ Inference  

### Run Inference on an Image  
```bash
python detect.py --weights weights/yolov7-vehicle.pt --conf 0.25 --img-size 640 --source sample.jpg

## 📜 Citation  
If you use this repository, consider citing:  
```
@inproceedings{wang2023yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

## 🛠️ Acknowledgments  
- **[Official YOLOv7 Repository](https://github.com/WongKinYiu/yolov7)**
- **[Kaggle Vehicle Dataset](https://www.kaggle.com/datasets/nadinpethiyagoda/vehicle-dataset-for-yolo)**  

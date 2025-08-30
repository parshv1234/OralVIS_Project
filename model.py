from ultralytics import YOLO

def main():
    model = YOLO('yolov8m.pt') 
    # model = YOLO('yolo11m.pt')
    results = model.train(
        data='data.yaml',
        imgsz=640,
        epochs=50,  
        batch=8,    
        device='mps',
        plots=True,
        workers=6,
        name='oralvis_yolov11m_run2',
        degrees=10,    
        translate=0.1, 
        scale=0.2, 
        fliplr=0.5,
        patience=15,
        visualize=True
    )
    # To resume model training if it stops in between uncomment below and comment the above model and results...
    # model = YOLO('runs/detect/oralvis_yolov11m_run1/weights/last.pt') 
    # results = model.train(resume=True, workers=4, patience=10)

if __name__ == '__main__':
    main()
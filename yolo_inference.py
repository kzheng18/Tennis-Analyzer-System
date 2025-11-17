from ultralytics import YOLO

model = YOLO('yolo12n.pt')

results = model.predict(source="input_video/input_video.mp4", save=True, )
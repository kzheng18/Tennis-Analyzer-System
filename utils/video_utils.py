import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path, input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    height, width = output_video_frames[0].shape[:2]
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise Exception("H.264 not available")
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f"Saving {len(output_video_frames)} frames at {fps} FPS...")
    for frame in output_video_frames:
        out.write(frame)
    
    out.release()
    print(f"✓ Video saved: {output_video_path}")
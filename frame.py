import cv2
import os

input_file = "video_file/NASA Scientific Visualization Studio ｜ Moon Essentials： Seasons (3).mp4"
output_folder = "frames"

cap = cv2.VideoCapture(input_file)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"fps: {fps}")

frame_count = 54

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if int(cap.get(cv2.CAP_PROP_POS_MSEC)) % 10000 > 0:  
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 10000.0  
        filename = os.path.join(output_folder, f"moon_{frame_count:04d}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename} at {timestamp:.2f} seconds")
        frame_count += 1

cap.release()
cv2.destroyAllWindows()

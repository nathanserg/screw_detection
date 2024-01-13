from ultralytics import YOLO
import cv2

model = YOLO("weights/ScrewDetector.pt")

frame = cv2.imread('evaluate/Photos/IMG_20231214_151226.jpg')

threshold = 0.3

results = model(frame)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

result = cv2.resize(frame, (1200,900))
cv2.imshow('result', result)
cv2.waitKey(0)  # 0 means wait indefinitely for a key press
cv2.destroyAllWindows()


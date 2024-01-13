from ultralytics import YOLO
import cv2
import os

model = YOLO("weights/yolo_detect_and_class.pt")

folder_path = os.path.abspath('evaluate/Photos')
files = [os.path.splitext(filename)[0] for filename in os.listdir(folder_path)]

# for file in files:
#     print(file)
#     frame = cv2.imread(f'evaluate/Photos/{file}.jpg')

#     threshold = 0.15

#     results = model(frame)[0]

#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#     result = cv2.resize(frame, (1200,900))
#     # cv2.imshow('result', result)
#     # cv2.waitKey(0)  # 0 means wait indefinitely for a key press
#     # cv2.destroyAllWindows()
#     cv2.imwrite(f'evaluate/YOLO_predictions_3/{file}.jpg', result)
# print("Done")

file = files[-1]

print(file)
frame = cv2.imread(f'evaluate/Photos/{file}.jpg')

threshold = 0.15

results = model(frame)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

result = cv2.resize(frame, (1200,900))
# cv2.imshow('result', result)
# cv2.waitKey(0)  # 0 means wait indefinitely for a key press
# cv2.destroyAllWindows()
cv2.imwrite(f'evaluate/YOLO_predictions_2/{file}.jpg', result)
print("Done")




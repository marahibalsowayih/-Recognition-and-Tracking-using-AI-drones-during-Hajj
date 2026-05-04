import cv2
import time
import argparse
import numpy as np
import onnxruntime as ort

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/yolo11n.onnx")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.40) # رفعنا عتبة الثقة
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    model_inputs = session.get_inputs()
    input_shape = model_inputs[0].shape
    input_width, input_height = input_shape[2], input_shape[3]

    # القائمة الرسمية لـ YOLOv8/v11 (تأكد من الترتيب)
    classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        img = cv2.resize(frame, (input_width, input_height))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)

        outputs = session.run(None, {model_inputs[0].name: img})
        output = np.squeeze(outputs[0]) # شكل المصفوفة: [84, 8400]

        boxes = []
        confidences = []
        class_ids = []

        # في YOLOv11، أول 4 صفوف هي الإحداثيات، والباقي هي الفئات
        rows = output.shape[1]
        for i in range(rows):
            classes_scores = output[4:, i]
            class_id = np.argmax(classes_scores)
            confidence = classes_scores[class_id]

            if confidence > args.conf:
                xc, yc, w, h = output[:4, i]
                x1 = int((xc - w/2) * frame.shape[1] / input_width)
                y1 = int((yc - h/2) * frame.shape[0] / input_height)
                width = int(w * frame.shape[1] / input_width)
                height = int(h * frame.shape[0] / input_height)
                
                boxes.append([x1, y1, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, args.conf, args.iou)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                # التأكد من عدم تجاوز فهرس المصفوفة
                name = classes[class_ids[i]] if class_ids[i] < len(classes) else "Unknown"
                label = f"{name}: {confidences[i]:.2f}"
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Hajj AI Drone - Final Fix", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
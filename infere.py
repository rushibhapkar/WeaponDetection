import cv2
import numpy as np
import smtplib
import time
from playsound import playsound

# Load YOLO
net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Email settings
SENDER_EMAIL = 'golanderushikesh2003@gmail.com'
SENDER_PASSWORD = 'xsmpukqribmnnrwz'  # App password
RECEIVER_EMAIL = 'rushikeshgolande01@gmail.com'

# Alarm cooldown to avoid spamming
last_alert_time = 0
cooldown = 10  # seconds

# Choose input: video file or webcam
def value():
    val = input("Enter file name or press Enter to start webcam: \n")
    return 0 if val == "" else val

cap = cv2.VideoCapture(value())

while True:
    ret, img = cap.read()
    if not ret:
        break

    height, width, _ = img.shape

    # Object detection
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0 and (time.time() - last_alert_time) > cooldown:
        print("⚠️ Weapon detected in frame")

        # Send Email
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            message = "Subject: Alert - Weapon Detected\n\nA weapon has been detected in your monitored area!"
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, message)
            server.quit()
            print("✅ Email sent successfully!")
        except Exception as e:
            print("❌ Error sending email:", e)

        # Play alarm sound
        try:
            playsound('alarm.wav')
        except Exception as e:
            print("❌ Error playing sound:", e)

        last_alert_time = time.time()

    # Draw boxes on screen
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    cv2.imshow("Weapon Detection", img)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()

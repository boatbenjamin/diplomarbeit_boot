import time
import cv2
from ultralytics import YOLO

# 1. Leichtes Modell laden (wird beim ersten Start automatisch heruntergeladen)
model = YOLO("yolov8n.pt")

# 2. Kamera-Stream öffnen (0 = Standard-Webcam/USB-Kamera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Fehler: Frame konnte nicht gelesen werden.")
        break

    # 3. YOLO-Inferenz durchführen (stream=True spart Arbeitsspeicher)
    results = model(frame, stream=True)

    for r in results:
        # Bounding Boxes, Labels und Confidence-Scores auf das Bild zeichnen
        annotated_frame = r.plot()

        # 4. FPS (Frames Per Second) berechnen
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        # FPS oben links einblenden
        cv2.putText(
            annotated_frame,
            f"FPS: {int(fps)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # 5. Live-Bild anzeigen
        cv2.imshow("YOLO Live-Erkennung", annotated_frame)

    # Beenden mit der Taste 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
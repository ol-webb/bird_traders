from datetime import datetime
import time
import cv2
from ultralytics import YOLO
from lumibot.strategies import Strategy
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
import json



class BirdTrader(Strategy):
    def initialize(self):
        self.camera = None  # Placeholder for the camera object
        self.last_trade_time = 0  # Cooldown timer
        self.cooldown_seconds = 10  # Time to wait before allowing another trade
        self.threshold = 0.4  # Confidence threshold for detection
        self.class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
        self.symbol = "BTC/USD"


    def before_starting_trading(self):
        """
        Called once at the start of the strategy.
        Open the camera and initialize the YOLO model.
        """
        print("Starting strategy...")
        self.camera = cv2.VideoCapture(0)  # Open the camera
        self.yolo_model = YOLO("yolo-Weights/yolov8n.pt")  # Load the YOLO model


    def detect_objects(self, frame):
        """
        Processes the frame using the YOLO model and returns detected objects
        of interest (e.g., birds) with their coordinates and other details.
        """
        results = self.yolo_model(frame, stream=True)
        objects_detected = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x_centre = (x1 + x2) // 2
                y_centre = (y1 + y2) // 2

                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                detected_class = self.class_names[cls]

                if confidence >= self.threshold:
                    objects_detected.append({
                        "class": detected_class,
                        "confidence": confidence,
                        "x": x_centre,
                        "y": y_centre,
                        "side_of_image": 1 if x_centre > 320 else 0,  # 1 for right, 0 for left
                    })

        return objects_detected



    def on_trading_iteration(self):
        """
        Called periodically during the strategy's execution.
        Fetch object detection input, and execute trading logic.
        """
        success, frame = self.camera.read()
        if not success:
            print("Failed to read from camera.")
            return

        detected_objects = self.detect_objects(frame)
        current_time = time.time()

        for idx, obj in enumerate(detected_objects):
            print(f"Detected {obj['class']} at ({obj['x']}, {obj['y']}), confidence: {obj['confidence']:.2f}")
            print(f"Object {idx + 1} is in the {'right' if obj['side_of_image'] == 1 else 'left'} of the image.")

            # Buy if on the right (1), sell if on the left (0), with cooldown
            if obj["class"] == "cell phone" and current_time - self.last_trade_time > self.cooldown_seconds:
                if obj["side_of_image"] == 1:
                    print(f"cell phone detected on the right! Placing a BUY order.")
                    order = self.create_order("BTC/USD", 0.00001, "buy")
                else:
                    print(f"cell phone detected on the left! Placing a SELL order.")
                    order = self.create_order("BTC/USD", 0.00001, "sell")

                self.submit_order(order)
                self.last_trade_time = current_time  # Update cooldown



    def on_finish(self):
        """
        Called once when the strategy finishes.
        Clean up resources like the camera.
        """
        print("Stopping strategy...")
        if self.camera is not None:
            self.camera.release()  # Release the camera



    def before_market_opens(self):
        if "USD" in self.symbol:
            print("Crypto market is always open. Skipping market hours check.")
            return
        super().before_market_opens()




if __name__ == "__main__":
    with open("alpaca_keys.json", "r") as file:
        data = json.load(file)

    alpaca_key = data['alpaca_key']
    alpaca_secret = data['alpaca_secret']

    ALPACA_CONFIG = {
        "API_KEY": alpaca_key,
        "API_SECRET": alpaca_secret,
        "PAPER": True,
    }
    # Create the broker instance
    broker = Alpaca(ALPACA_CONFIG)

    # Initialize the strategy
    strategy = BirdTrader(broker=broker)

    # Create and configure the trader
    trader = Trader()
    trader.add_strategy(strategy)

    trader.run_all()
from ultralytics import YOLO
from lumibot.strategies import Strategy
from lumibot.brokers import Alpaca
from lumibot.traders import Trader
import cv2
import json


class BirdTrader(Strategy):
    def initialize(self):
        """
        Initialization for the strategy.
        """
        self.camera = None  # Placeholder for the camera
        self.yolo_model = None  # Placeholder for YOLO model
        self.trade_made = False  # Flag to indicate if a trade has been made
        self.target_class = "cell phone"  # The object to detect
        self.threshold = 0.4  # Confidence threshold for detection
        self.symbol = "BTC/USD"  # Symbol for trading (crypto example)

    def before_starting_trading(self):
        """
        Called once before trading starts.
        Initializes the camera and YOLO model.
        """
        print("Initializing camera and YOLO model...")
        self.camera = cv2.VideoCapture(1)  # Open the camera
        self.yolo_model = YOLO("yolo-Weights/yolov8n.pt")  # Load the YOLO model

    def detect_objects(self, image):
        """
        Processes a frame using YOLO and returns detected objects.
        Only returns objects matching the target class above the confidence threshold.
        """
        results = self.yolo_model(image, stream=True, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract bounding box and detection details
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x_centre = (x1 + x2) // 2

                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                detected_class = self.yolo_model.names[cls]

                if confidence >= self.threshold and detected_class == self.target_class:
                    # Draw bounding box and details on the image
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(image, f"{detected_class} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    return {
                        "class": detected_class,
                        "confidence": confidence,
                        "x_centre": x_centre,
                        "image_width": image.shape[1],
                    }
        return None

    def on_trading_iteration(self):
        """
        Processes the camera feed continuously and trades only once when the target is detected.
        """
        if self.trade_made:
            print("Trade already made. Stopping further trades.")
            self.stop_trading()
            return

        while True:
            success, image = self.camera.read()  # Capture a frame from the camera
            if not success:
                print("Failed to read from camera.")
                break

            # Detect objects in the frame
            detected_object = self.detect_objects(image)

            # Show the live camera feed
            cv2.imshow("Camera Feed", image)

            # Check if the target object was detected
            if detected_object:
                print(f"Detected {detected_object['class']} with confidence {detected_object['confidence']:.2f}")

                # Make a trade based on the object's position
                if detected_object["x_centre"] > detected_object["image_width"] // 2:
                    print("Object on the right: Placing a BUY order.")
                    order = self.create_order(self.symbol, 0.001, "buy")
                else:
                    print("Object on the left: Placing a SELL order.")
                    order = self.create_order(self.symbol, 0.001, "sell")

                self.submit_order(order)
                self.trade_made = True  # Set the flag to prevent further trades
                break

            # Quit the camera feed by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def on_finish(self):
        """
        Called when the strategy stops.
        Releases the camera resource.
        """
        print("Stopping strategy and releasing camera...")
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()  # Close any open windows


if __name__ == "__main__":
    # Alpaca configuration
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

    # Run the trader in real-time
    trader.run_all()

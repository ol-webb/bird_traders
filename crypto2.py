from ultralytics import YOLO
from lumibot.brokers import Alpaca
from lumibot.entities import Asset
from lumibot.strategies import Strategy
from lumibot.traders import Trader
import cv2
import json
import time

class BirdTrader(Strategy):
    def initialize(self):
        """
        Initialization for the strategy.
        """
        self.symbol = "BTC/USD"  # Symbol for trading
        self.order_qty = 0.004  # Quantity for orders
        self.target_class = "bird"  # The object to detect
        self.threshold = 0.1  # Confidence threshold for detection
        self.trade_made = False  # Flag to indicate if a trade has been made
        self.camera = None  # Placeholder for the camera
        self.yolo_model = None  # Placeholder for YOLO model
        self.cooldown_time = 10  # Cooldown time in seconds
        self.last_trade_time = 0  # Time of the last trade
        self.set_market("24/7")
        self.camera = cv2.VideoCapture(1)  # Open the camera
        self.yolo_model = YOLO("yolo-Weights/yolov8n.pt")  # Load the YOLO model

    def before_starting_trading(self):
        """
        Called once before trading starts.
        Initializes the camera and YOLO model.
        """
        print("Initializing camera and YOLO model...")
        self.camera = cv2.VideoCapture(1)  # Open the camera
        self.yolo_model = YOLO("yolo-Weights/yolov8n.pt")  # Load the YOLO model
        print("done")

    def detect_objects(self, image):
        """
        Processes a frame using YOLO and returns detected objects.
        Draws bounding boxes and central points for detected objects.
        """
        results = self.yolo_model(image, stream=True, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract bounding box and detection details
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                x_centre = (x1 + x2) // 2
                y_centre = (y1 + y2) // 2
                image_width = image.shape[1]
                image_height = image.shape[0]
                norm_x = x_centre / image_width
                norm_y = y_centre / image_height

                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                detected_class = self.yolo_model.names[cls]

                if confidence >= self.threshold and detected_class == self.target_class:
                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(image, f"{detected_class} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Draw central point and coordinates
                    cv2.circle(image, (x_centre, y_centre), radius=5, color=(0, 0, 255), thickness=-1)
                    coord_text = f"({norm_x:.2f}, {norm_y:.2f})"
                    cv2.putText(image, coord_text, (x_centre + 10, y_centre),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    return {
                        "class": detected_class,
                        "confidence": confidence,
                        "norm_x": norm_x,
                    }
        return None

    def on_trading_iteration(self):
        """
        Processes the camera feed continuously and submits a buy or sell order based on the target's position.
        Includes a cooldown timer between trades.
        """
        while True:
            success, image = self.camera.read()  # Capture a frame from the camera
            if not success:
                print("Failed to read from camera.")
                break

            # Calculate remaining cooldown time
            current_time = time.time()
            time_since_last_trade = current_time - self.last_trade_time
            remaining_cooldown = max(0, self.cooldown_time - time_since_last_trade)

            # Display cooldown timer on the feed
            cv2.putText(image, f"Cooldown: {remaining_cooldown:.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Detect objects in the frame
            detected_object = self.detect_objects(image)

            # Show the live camera feed
            cv2.imshow("Camera Feed", image)

            # Check if the target object was detected and cooldown is over
            if detected_object and remaining_cooldown == 0:
                print(f"Detected {detected_object['class']} with confidence {detected_object['confidence']:.2f}")

                # Determine order side based on object's position
                order_side = "buy" if detected_object["norm_x"] <= 0.5 else "sell"
                order_type = "Buy" if order_side == "buy" else "Sell"
                print(f"{order_type} order selected based on object position.")

                # Create and submit the order
                asset = Asset(symbol="BTC", asset_type=Asset.AssetType.CRYPTO)
                quote = Asset(symbol="USD", asset_type=Asset.AssetType.CRYPTO)

                order = self.create_order(
                    asset=asset,
                    quantity=self.order_qty,
                    side=order_side,
                    time_in_force="gtc",
                    quote=quote
                )

                self.submit_order(order)
                print(f"{order_type} Order Submitted!")

                # Update last trade time
                self.last_trade_time = current_time

            # Quit the camera feed by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting trading loop...")
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

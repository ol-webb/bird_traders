from lumibot.brokers import Alpaca
from lumibot.entities import Asset
from lumibot.strategies import Strategy
from lumibot.traders import Trader
import json


class SimpleCryptoTrader(Strategy):
    def initialize(self):
        self.symbol = "BTC/USD"  # Crypto pair to trade
        self.order_qty = 0.0001  # Quantity of crypto to trade
        self.set_market('24/7')

    def before_market_opens(self):
        print("Crypto market is always open. Skipping market hours check.")
        return

    def on_trading_iteration(self):
        print(f"Placing a BUY order for {self.symbol}...")
        asset = Asset(symbol="BTC", asset_type=Asset.AssetType.CRYPTO)
        quote = Asset(symbol="USD", asset_type=Asset.AssetType.CRYPTO)

        order = self.create_order(
            asset=asset,
            quantity=self.order_qty,
            side="buy",
            time_in_force="gtc",
            quote=quote
        )
        self.submit_order(order)
        print("Order submitted successfully!")



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
    strategy = SimpleCryptoTrader(broker=broker)

    # Create and configure the trader
    trader = Trader()
    trader.add_strategy(strategy)

    # Run the trader
    trader.run_all()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import heapq
from scipy.linalg import cholesky
import tabulate
from sklearn.mixture import GaussianMixture

def randomize_rho(base_rho=0.35, variation=0.15):
    """Generate random correlation within ±variation% of base_rho"""
    return np.clip(
        base_rho + np.random.uniform(-variation, variation) * base_rho,
        0.01,  # Ensure positive definite
        0.99    # Keep below 1
    )

class Order:
    def __init__(self, order_id, price, quantity, side, symbol=None, timestamp=None):
        self.order_id = order_id
        self.price = price
        self.quantity = quantity
        self.side = side  # 'bid' (buy) or 'ask' (sell)
        self.symbol = symbol  # New: track which security this order is for
        self.timestamp = timestamp or time.time()
    
    def __repr__(self):
        return f"Order(ID={self.order_id}, Symbol={self.symbol}, Price={self.price}, Qty={self.quantity}, Side={self.side})"

class OrderBook:
    def __init__(self, symbol):
        self.symbol = symbol  # Track which security this book is for
        self.bids = []  # Max-heap (use negative prices)
        self.asks = []  # Min-heap
        self.order_map = {}  # order_id -> Order
        self.trade_history = []
    
    def add_order(self, order):
        if order.side == 'bid':
            heapq.heappush(self.bids, (-order.price, order.timestamp, order))
        elif order.side == 'ask':
            heapq.heappush(self.asks, (order.price, order.timestamp, order))
        self.order_map[order.order_id] = order
        self._match_orders()
    
    def cancel_order(self, order_id):
        if order_id in self.order_map:
            order = self.order_map[order_id]
            if order.side == 'bid':
                self.bids = [o for o in self.bids if o[2].order_id != order_id]
                heapq.heapify(self.bids)
            elif order.side == 'ask':
                self.asks = [o for o in self.asks if o[2].order_id != order_id]
                heapq.heapify(self.asks)
            del self.order_map[order_id]
    
    def _match_orders(self):
        while self.bids and self.asks and (-self.bids[0][0] >= self.asks[0][0]):
            best_bid = self.bids[0][2]
            best_ask = self.asks[0][2]
            
            trade_qty = min(best_bid.quantity, best_ask.quantity)
            trade_price = best_ask.price  # Execution at ask price
            print(f"Trade {self.symbol} @ {trade_price:.2f}: {trade_qty} shares")
            
            # Record trade
            self.trade_history.append({
                'timestamp': time.time(),
                'price': trade_price,
                'quantity': trade_qty,
                'symbol': self.symbol
            })
            
            # Update quantities
            best_bid.quantity -= trade_qty
            best_ask.quantity -= trade_qty
            
            # Remove filled orders
            if best_bid.quantity == 0:
                heapq.heappop(self.bids)
                del self.order_map[best_bid.order_id]
            if best_ask.quantity == 0:
                heapq.heappop(self.asks)
                del self.order_map[best_ask.order_id]
    
    def get_depth(self, side, levels=5):
        if side == 'bid':
            return sorted([(-p, t, o) for p, t, o in self.bids], reverse=True)[:levels]
        else:
            return sorted([(p, t, o) for p, t, o in self.asks])[:levels]
    
    def get_midprice(self):
        best_bid = -self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        return (best_bid + best_ask)/2 if (best_bid and best_ask) else None

    def display(self, levels=5):
        print(f"\n------ {self.symbol} Order Book ------")
        print("Bids (Top {}):".format(levels))
        for bid in self.get_depth('bid', levels):
            print(f"  {bid[2].quantity:.2f} @ {bid[0]:.2f}")
        
        print("\nAsks (Top {}):".format(levels))
        for ask in self.get_depth('ask', levels):
            print(f"  {ask[2].quantity:.2f} @ {ask[0]:.2f}")
        print("-----------------------")

    def get_orderbook_state(self, levels=5):
        bids = sorted([(-p, t, o) for p, t, o in self.bids], reverse=True)[:levels]
        asks = sorted([(p, t, o) for p, t, o in self.asks])[:levels]

        # Safe calculations
        best_bid = -bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None

        return {
            'symbol': self.symbol,
            'timestamp': time.time(),
            'bids': [{'price': p, 'qty': o.quantity} for p, _, o in bids],
            'asks': [{'price': p, 'qty': o.quantity} for p, _, o in asks],
            'midprice': (best_bid + best_ask)/2 if (best_bid and best_ask) else None,
            'spread': (best_ask - best_bid) if (best_bid and best_ask) else None
        }



class EquityGBM:
    def __init__(self, symbols, initial_prices, mus, sigmas, rho_matrix, dt=1/252):
        self.symbols = symbols
        self.prices = dict(zip(symbols, initial_prices))
        self.mus = dict(zip(symbols, mus))
        self.sigmas = dict(zip(symbols, sigmas))
        self.rho_matrix = rho_matrix
        self.dt = dt
        self._setup_correlated_gbm()
    
    def _setup_correlated_gbm(self):
        """Initialize Cholesky decomposition"""
        self.L = cholesky(self.rho_matrix, lower=True)
    
    def update_correlation(self, new_rho_matrix):
        self.rho_matrix = new_rho_matrix
        self._setup_correlated_gbm()
    
    def step(self):
        z = np.random.normal(0, 1, len(self.symbols))
        epsilons = self.L @ z

        for i, symbol in enumerate(self.symbols):
            drift = (self.mus[symbol] - 0.5*self.sigmas[symbol]**2)*self.dt
            diffusion = self.sigmas[symbol]*np.sqrt(self.dt)*epsilons[i]
            self.prices[symbol] *= np.exp(drift + diffusion)
        return self.prices  # This line was missing


class MarketMaker:
    def __init__(self, symbols, spread=0.002, base_qty=10):  # Reduced from 0.01 to 0.002 (0.2%)
        self.symbols = symbols
        self.spread = spread
        self.base_qty = base_qty
        self.order_id_counter = 0

    def _calculate_spread(self, symbol):
        return self.spread  # Can enhance with volatility-based calculation later
    
    def generate_orders(self, current_prices):
        orders = []
        for symbol in self.symbols:
            mid = current_prices[symbol]
            spread = self._calculate_spread(symbol)
            
            # Generate tighter order levels
            for i in range(5):
                # Reduced spread multiplier from (i+1) to (0.2*(i+1))
                bid_price = mid * (1 - spread*(0.1*(i+1)))
                ask_price = mid * (1 + spread*(0.1*(i+1)))
                
                # Keep quantities reasonable
                qty = int(self.base_qty * (1 - i*0.15))  # Slightly more gradual quantity reduction
                
                orders.extend([
                    Order(f"MM_{symbol}_bid_{self.order_id_counter}", 
                         bid_price, qty, 'bid', symbol),
                    Order(f"MM_{symbol}_ask_{self.order_id_counter}", 
                         ask_price, qty, 'ask', symbol)
                ])
                self.order_id_counter += 2
        return orders



def run_extended_simulation(time_steps=50, display_every=5):
    # Initialize parameters
    symbols = ['S1', 'S2', 'S3']
    initial_prices = [100.0, 150.0, 80.0]
    mus = [0.12, 0.17, 0.09]
    sigmas = [0.20, 0.15, 0.25]
    rho_matrix = np.array([ 
        [1.00, 0.35, 0.35],
        [0.35, 1.00, 0.35],
        [0.35, 0.35, 1.00]
    ])
    
    # Create components
    gbm = EquityGBM(symbols, initial_prices, mus, sigmas, rho_matrix)
    order_books = {symbol: OrderBook(symbol) for symbol in symbols}
    market_maker = MarketMaker(symbols)
    
    # Data collection structures
    simulation_data = {
        'timesteps': [],
        'orderbook_states': [],
        'correlation_matrices': [],
        'price_history': {sym: [] for sym in symbols}
    }
    
    # Initial setup
    prices = gbm.step()
    print("Initial Prices:", {k: f"{v:.2f}" for k,v in prices.items()})
    
    # Initial market making
    for order in market_maker.generate_orders(prices):
        order_books[order.symbol].add_order(order)
    
    # Main simulation loop
    for step in range(time_steps):
        current_time = time.time()
        print(f"\n=== Time Step {step + 1}/{time_steps} ===")
        
        # Update correlation matrix periodically
        if step % 5 == 0:
            new_rho = np.array([
                [1.00, randomize_rho(), randomize_rho()],
                [randomize_rho(), 1.00, randomize_rho()],
                [randomize_rho(), randomize_rho(), 1.00]
            ])
            gbm.update_correlation(new_rho)
            print("Updated Correlations:\n", gbm.rho_matrix)
            simulation_data['correlation_matrices'].append({
                'timestep': step,
                'matrix': gbm.rho_matrix.copy()
            })
        
        # Update prices and record
        prices = gbm.step()
        for sym, price in prices.items():
            simulation_data['price_history'][sym].append(price)
        print("\nCurrent Prices:", {k: f"{v:.2f}" for k,v in prices.items()})
        
        # Refresh market maker orders
        mm_orders = market_maker.generate_orders(prices)
        for symbol in symbols:
            # Clear old market maker orders
            book = order_books[symbol]
            for order_id in list(book.order_map.keys()):
                if order_id.startswith("MM_"):
                    book.cancel_order(order_id)
            
            # Add new market maker orders
            for order in mm_orders:
                if order.symbol == symbol:
                    book.add_order(order)
        
        # Add random orders
        if step % 2 == 0:
            for symbol in symbols:
                side = np.random.choice(['bid', 'ask'])
                price = prices[symbol] * (1 + np.random.uniform(-0.02, 0.02))
                order_id = f"RND_{step}_{symbol}_{np.random.randint(1000)}"
                order_books[symbol].add_order(Order(
                    order_id, price, np.random.randint(1, 20), side, symbol
                ))
                print(f"Added random {side} order for {symbol} @ {price:.2f}")
        
        # Capture orderbook state (every step)
        step_data = {
            'timestep': step,
            'timestamp': current_time,
            'orderbooks': {},
            'features': {}
        }
        
        for symbol in symbols:
            book = order_books[symbol]
            # Store raw orderbook state
            step_data['orderbooks'][symbol] = book.get_orderbook_state()
            
            step_data['features'][symbol] = {
                'imbalance': (sum(b['qty'] for b in step_data['orderbooks'][symbol]['bids']) - 
                             sum(a['qty'] for a in step_data['orderbooks'][symbol]['asks'])) / 
                            (sum(b['qty'] for b in step_data['orderbooks'][symbol]['bids']) + 
                             sum(a['qty'] for a in step_data['orderbooks'][symbol]['asks']) + 1e-6),
                'rel_spread': (step_data['orderbooks'][symbol]['spread'] / 
                              step_data['orderbooks'][symbol]['midprice']) 
                              if (step_data['orderbooks'][symbol]['spread'] is not None and 
                                  step_data['orderbooks'][symbol]['midprice'] is not None and
                                  step_data['orderbooks'][symbol]['midprice'] != 0) 
                              else None,
                'microprice': (sum(b['qty']*b['price'] for b in step_data['orderbooks'][symbol]['bids'][:5]) +
                              sum(a['qty']*a['price'] for a in step_data['orderbooks'][symbol]['asks'][:5])) /
                             (sum(b['qty'] for b in step_data['orderbooks'][symbol]['bids'][:5]) +
                              sum(a['qty'] for a in step_data['orderbooks'][symbol]['asks'][:5]) + 1e-6)
            }
        
        simulation_data['timesteps'].append(step_data)
        
        # Display order books periodically
        if step % display_every == 0 or step == time_steps - 1:
            for symbol in symbols:
                order_books[symbol].display()
        
        # Pause for visualization/analysis
        time.sleep(0.10)
    
    # Post-simulation processing
    def create_dataframe(simulation_data):
        records = []
        for step_data in simulation_data['timesteps']:
            for symbol in symbols:
                records.append({
                    'timestep': step_data['timestep'],
                    'timestamp': step_data['timestamp'],
                    'symbol': symbol,
                    **step_data['features'][symbol],
                    'price': simulation_data['price_history'][symbol][step_data['timestep']]
                })
        return pd.DataFrame(records)
    
    simulation_data['df'] = create_dataframe(simulation_data)
    
    return simulation_data

if __name__ == "__main__":
    # Run either the original or extended simulation
    # run_simulation()  # Original version
    run_extended_simulation(time_steps=50, display_every=5)  # New version



def get_orderbook_state(self, levels=5):
    bids = sorted([(-p, t, o) for p, t, o in self.bids], reverse=True)[:levels]
    asks = sorted([(p, t, o) for p, t, o in self.asks])[:levels]
    
    return {
        'symbol': self.symbol,
        'timestamp': time.time(),
        'bids': [{'price': p, 'qty': o.quantity} for p, _, o in bids],
        'asks': [{'price': p, 'qty': o.quantity} for p, _, o in asks],
        'midprice': self.get_midprice(),
        'spread': asks[0][0] - (-bids[0][0]) if bids and asks else None
    }

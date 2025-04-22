import json
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from itertools import permutations
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from orderbook import run_extended_simulation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#Plot D and p-values over time.
#Overlay with price series and volatility (e.g., rolling std dev of returns).
#Null hypothesis: Bivariate normal distribution with mean/covariance estimated from the entire training set (or a "calm" baseline period).
#If D is significant, classify the window as "non-Gaussian" (potential regime shift)s

class OrderBookAnalyzer:
    def __init__(self, window_size=100, step_size=50):
        self.raw_data = None
        self.features = pd.DataFrame()
        self.gmm = None
        self.window_size = window_size
        self.step_size = step_size
        self.results = []

    def run_simulation(self, time_steps=1000):
        self.raw_data = run_extended_simulation(time_steps=time_steps)
        records = []
        
        for step_data in self.raw_data['timesteps']:
            for symbol, ob_state in step_data['orderbooks'].items():
                records.append({
                    'timestep': step_data['timestep'],
                    'timestamp': step_data['timestamp'],
                    'symbol': symbol,
                    'price': self.raw_data['price_history'][symbol][step_data['timestep']],
                    **self._create_features(ob_state)
                })
        
        self.features = pd.DataFrame(records).sort_values('timestep')
        return self.features

    def _create_features(self, ob_state):
        # Safely calculate bid/ask volumes
        bid_vol = sum(b['qty'] for b in ob_state.get('bids', []))
        ask_vol = sum(a['qty'] for a in ob_state.get('asks', []))

        # Safely get spread and midprice
        spread = ob_state.get('spread')
        midprice = ob_state.get('midprice')

        # Calculate relative spread with protection
        try:
            rel_spread = spread / midprice if (spread is not None and midprice is not None and midprice != 0) else None
        except (TypeError, ZeroDivisionError):
            rel_spread = None

        return {
            'imbalance': (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6),
            'rel_spread': rel_spread
        }

    def _rosenblatt_transform(self, X, order):
        transformed = np.zeros_like(X)
        n_samples, n_features = X.shape
        
        transformed[:, 0] = norm.cdf(X[:, order[0]])
        
        for i in range(1, n_features):
            kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X[:, order[:i]])
            log_prob = kde.score_samples(X[:, order[:i]])
            transformed[:, i] = norm.cdf(X[:, order[i]] - np.exp(log_prob))
            
        return transformed

    def _bivariate_ks_statistic(self, X, order):
        transformed = self._rosenblatt_transform(X, order)
        n = len(X)
        ecdf = np.zeros(n)
        
        for i in range(n):
            ecdf[i] = np.mean(
                (transformed[:,0] <= transformed[i,0]) & 
                (transformed[:,1] <= transformed[i,1])
            )
        
        uniform_prod = transformed[:,0] * transformed[:,1]
        d_n = np.max(np.abs(ecdf - uniform_prod))
        return d_n

    def bivariate_ks_test(self, X, n_permutations=100, alpha=0.05):
        """
        Returns KS stat, p-value, and rejection flag.
        """
        # Standardize features
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        feature_orders = list(permutations(range(X.shape[1])))
        p_factorial = len(feature_orders)
        
        D_n_star = max(
            self._bivariate_ks_statistic(X, order)
            for order in feature_orders
        )
        
        null_stats = []
        for _ in range(n_permutations):
            null_samples = np.random.uniform(size=X.shape)
            null_D = max(
                self._bivariate_ks_statistic(null_samples, order)
                for order in feature_orders
            )
            null_stats.append(null_D)
        
        critical_value = np.percentile(null_stats, 100*(1 - alpha/p_factorial))
        p_value = np.mean(np.array(null_stats) >= D_n_star) * p_factorial
        
        return D_n_star, p_value, D_n_star > critical_value

    def analyze_regimes(self):
        """Sliding Window"""
        n_windows = (len(self.features) - self.window_size) // self.step_size
        for i in tqdm(range(n_windows)):
            start = i * self.step_size
            end = start + self.window_size
            window = self.features.iloc[start:end]
            window_data = window[['imbalance', 'rel_spread']].dropna().values
            
            if len(window_data) < 10:
                continue
            
            # Fit GMM to window
            gmm = GaussianMixture(n_components=2, covariance_type='full')
            gmm.fit(window_data)
            labels = gmm.predict(window_data)
            
            # Run your original KS test on the window
            ks_stat, p_value, reject = self.bivariate_ks_test(window_data)
            
            # Save results with timestamps and prices
            self.results.append({
                'start_timestep': window['timestep'].min(),
                'end_timestep': window['timestep'].max(),
                'ks_stat': ks_stat,
                'p_value': p_value,
                'reject_null': reject,
                'mean_price': window['price'].mean(),
                'price_volatility': window['price'].std(),
                'cluster_0_mean': gmm.means_[0].tolist(),
                'cluster_1_mean': gmm.means_[1].tolist(),
                'n_cluster_0': sum(labels == 0),
                'n_cluster_1': sum(labels == 1),
                'cluster_0_imb_mean': gmm.means_[0][0],  # Mean imbalance for cluster 0
                'cluster_0_spread_mean': gmm.means_[0][1],  # Mean spread for cluster 0
                'cluster_1_imb_mean': gmm.means_[1][0],
                'cluster_1_spread_mean': gmm.means_[1][1],
                'cluster_assignment': labels[-1]  # Use last point's cluster for visualization
            })
        
        return pd.DataFrame(self.results)

    def plot_results(self, results_df):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        ax[0].plot(results_df['mean_price'], label='Price')
        ax[1].plot(results_df['price_volatility'], label='Volatility')
        ax[2].plot(results_df['ks_stat'], label='KS Statistic')
        ax[2].axhline(y=0.15, color='r', linestyle='--', label='Rejection Threshold')
        plt.tight_layout()
        plt.savefig('regime_analysis.png')

    def _process_simulation_data(self, simulation_data):
        records = []
        for step_data in simulation_data['timesteps']:
            for symbol, ob_state in step_data['orderbooks'].items():
                records.append({
                    'timestep': step_data['timestep'],
                    'timestamp': step_data['timestamp'],
                    'symbol': symbol,
                    'price': simulation_data['price_history'][symbol][step_data['timestep']],
                    **self._create_features(ob_state)
                })
        return pd.DataFrame(records).sort_values('timestep')


def visualize_results(results_df, price_history):
    """
    - Price series
    - Rolling volatility
    - KS statistics
    - Regime annotations
    """
    symbols = list(price_history.keys())
    
    # Create Plotly interactive figure
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Price Series", "Rolling Volatility", 
                       "KS Test Statistic", "Regime Classification")
    )
    
    # 1. Price Series
    for sym in symbols:
        fig.add_trace(
            go.Scatter(
                x=results_df['end_timestep'],
                y=price_history[sym],
                name=f"{sym} Price",
                line=dict(width=1)
            ),
            row=1, col=1  # Correct placement: in add_trace()
        )
    
    # 2. Rolling Volatility (20-period std dev of returns)
    for sym in symbols:
        prices = pd.Series(price_history[sym])
        returns = prices.pct_change()
        volatility = returns.rolling(20).std()
        
        fig.add_trace(
            go.Scatter(
                x=results_df['end_timestep'],
                y=volatility,
                name=f"{sym} Volatility",
                line=dict(width=1)
            ),
            row=2, col=1  # Correct placement
        )
    
    # 3. KS Statistics
    fig.add_trace(
        go.Scatter(
            x=results_df['end_timestep'],
            y=results_df['ks_stat'],
            name="KS Statistic",
            mode='lines+markers',
            marker=dict(
                color=np.where(results_df['reject_null'], 'red', 'green'),
                size=5
            ),
            line=dict(color='gray', width=1)
        ),
        row=3, col=1  # Correct placement
    )
    
    # 4. Regime Classification (simplified)
    fig.add_trace(
        go.Scatter(
            x=results_df['end_timestep'],
            y=results_df['cluster_0_imb_mean'],  # Using actual column name
            name="Regime Classification",
            mode='markers',
            marker=dict(
                size=8,
                color=results_df['cluster_assignment'],
                colorscale='Viridis'
            )
        ),
        row=4, col=1  # Correct placement
    )
    
    fig.update_layout(
        height=1200,
        title_text="Multi-Asset Regime Detection",
        hovermode="x unified"
    )
    fig.show()




if __name__ == "__main__":
    analyzer = OrderBookAnalyzer(window_size=100, step_size=50)
    
    # 1. Run simulation
    print("Running simulation...")
    simulation_data = run_extended_simulation(time_steps=1000)
    
    # 2. Process data
    print("Processing features...")
    analyzer.features = analyzer._process_simulation_data(simulation_data)
    
    # 3. Analyze regimes
    print("Detecting regimes...")
    results_df = analyzer.analyze_regimes()
    
    # 4. Save and visualize
    results_df.to_csv('regime_results.csv', index=False)
    print("Generating visualizations...")
    visualize_results(results_df, simulation_data['price_history'])
    
    print("Analysis complete! Results saved to:")
    print("- regime_results.csv")
    print("- regime_detection.png")




## JSON output
#import json
#with open('orderbook_states.json', 'w') as f:
#    json.dump(analyzer.raw_data, f)
#
## Parquet for efficient storage
#features.to_parquet('orderbook_features.parquet')



#analyzer = OrderBookAnalyzer()
#features = analyzer.run_simulation(time_steps=500)  # More data for better KS
#gmm = analyzer.fit_gmm(n_components=3)
#
## Rosenblatt and KS tests
#ks_results = perform_ks_tests(analyzer)
#
#print("KS Test Results:")
#print(ks_results)
#
## Visualize transformed data
#import matplotlib.pyplot as plt
#for k in range(analyzer.gmm.n_components):
#    transformed = analyzer.transform_to_rosenblatt(k)
#    plt.scatter(transformed[:,0], transformed[:,1], alpha=0.5, label=f'Cluster {k}')
#plt.title('Rosenblatt-Transformed Data')
#plt.legend()
#plt.show()

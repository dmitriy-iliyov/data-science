import pandas as pd

from src import crypto_analyzer as ca


currency = 'bitcoin'
# df = parser.coin_parsing(currency, 365)
df = pd.read_csv('files/' + currency + '.csv')
sample = df['price']

analyzer = ca.CryptoAnalyzer(sample, currency)
analyzer.remove_anomalies('sliding-window', 2)
analyzer.approximate(9)

# analyzer.remove_anomalies('medium', 3)
# analyzer.approximate(9)

analyzer.filter('alpha-beta', 50)
analyzer.approximate(9)

analyzer.extrapolate()

analyzer.model(365, True, True, True)

analyzer.remove_anomalies('sliding-window', 2)
analyzer.approximate(9)

analyzer.filter('alpha-beta', 50)
analyzer.approximate(9)

import pandas as pd

from src import crypto_analyzer as ca
from lab_2.src.tools import parser


currency = 'bitcoin'
df = parser.coin_parsing(currency, 365, 'daily')
if df.empty:
    df = pd.read_csv('files/' + currency + '_daily.csv')
sample = df['price']

analyzer = ca.CryptoAnalyzer(sample, currency)

analyzer.clean('sliding-window', 2)
analyzer.approximate(5)
analyzer.extrapolate()

analyzer.filter('alpha-beta', 50)
analyzer.extrapolate()

analyzer.model(365, True, True, True)

analyzer.clean('sliding-window', 2)
analyzer.approximate(5)
analyzer.extrapolate()

analyzer.filter('alpha-beta', 50)
analyzer.extrapolate()

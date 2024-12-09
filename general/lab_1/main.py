import pandas as pd

import parser
import statistic_learning as sl


currency = 'bitcoin'
# df = parser.coin_parsing(currency, 365, 'daily')
df = pd.read_csv('files/' + currency + '.csv')
sample = df['price']
analyzer = sl.CryptoAnalyzer(sample, currency)
# analyzer.lsm_approximation()
# analyzer.lsm_extrapolation(365)
# analyzer.model(365, True, True)

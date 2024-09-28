import pandas as pd

import parser
import ploter
import statistic_learning as sl


currency = 'ethereum'
# df = parser.coin_parsing(currency, 365)
df = pd.read_csv('files/' + currency + '.csv')
sample = df['price']
model = sl.CryptoAnalyzer(sample, currency)
model.lsm_approximation()
model.lsm_extrapolation(1000)
model.model(1000, True, True)

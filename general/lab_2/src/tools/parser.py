import datetime
import time

import numpy as np
import requests
import pandas as pd


def coin_price_parsing(currency, days=365):
    test_url = 'https://api.coingecko.com/api/v3/coins/' + currency
    test_response = requests.get(test_url)
    if test_response.status_code == 200:
        print('Test request successful')
        url = test_url + '/market_chart'
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }

        response = requests.get(url, params=params)
        coin_year_prices = response.json()
        dates = []
        prices = []
        for daily_price in coin_year_prices['prices']:
            dates.append(datetime.datetime.fromtimestamp(daily_price[0] / 1000))
            prices.append(daily_price[1])
        coin_year_prices_df = pd.DataFrame({'date': dates, 'price': prices})
        coin_year_prices_df.to_csv('files/' + currency + '.csv', index=True)
        print('Coin parsing successful')
        return coin_year_prices_df
    else:
        print('Test request failed, status src ' + str(test_response.status_code))
        print(test_response.text)
        return None


def coin_data_parsing(currency, days=365):
    currency = currency.lower()
    test_url = 'https://api.coingecko.com/api/v3/coins/' + currency
    test_response = requests.get(test_url)

    if test_response.status_code == 200:
        print('Test request successful')

        url = f'https://api.coingecko.com/api/v3/coins/{currency}/market_chart'
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }

        response = requests.get(url, params=params)
        market_data = response.json()
        dates, prices = [], []
        for daily_price in market_data['prices']:
            dates.append(datetime.datetime.fromtimestamp(daily_price[0] / 1000))
            prices.append(daily_price[1])

        market_caps, volumes = [], []
        for daily_cap in market_data['market_caps']:
            market_caps.append(daily_cap[1])
        for daily_vol in market_data['total_volumes']:
            volumes.append(daily_vol[1])
        c_market_cap = market_caps[-1]
        std_market_cap = np.std(market_caps)
        volume = sum(volumes)

        coin_info_url = f'https://api.coingecko.com/api/v3/coins/{currency}'
        coin_info_response = requests.get(coin_info_url)
        if coin_info_response.status_code == 200:
            coin_info = coin_info_response.json()
            circulating_supply = coin_info['market_data']['circulating_supply']
            total_supply = coin_info['market_data']['total_supply']
        else:
            circulating_supply = 0
            total_supply = 0

        growth = prices[-1] - prices[0]
        volatility = np.std(prices)

        coin_data_df = pd.DataFrame([{
            'growth': growth,
            'volatility': volatility,
            'c_market_cap': c_market_cap,
            'std_market_cap': std_market_cap,
            'volume': volume,
            'circulating_supply': circulating_supply,
            'total_supply': total_supply
        }])

        coin_data_df.to_csv(f'files/{currency}_market_data.csv', index=False)
        print('Coin parsing successful')
        return coin_data_df
    elif test_response.status_code == 429:
        print(f'Test request failed, status code: {test_response.status_code}')
        time.sleep(60)
        return coin_data_parsing(currency, days)
    else:
        print(f'Test request failed, status code: {test_response.status_code}')
        return None

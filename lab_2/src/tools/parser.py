import datetime
import requests
import pandas as pd


def coin_parsing(currency, days=365):
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

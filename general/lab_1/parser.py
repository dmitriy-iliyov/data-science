import datetime
import requests
import pandas as pd

from lab_6.tools import filer


def coin_parsing(currency, days=90, interval='hourly'):

    filepath = ("/Users/sayner/github_repos/data-science/general/lab_1/files/" + str(currency).lower() + "_"
                + interval + ".csv")

    test_url = 'https://api.coingecko.com/api/v3/coins/' + currency
    test_response = requests.get(test_url)

    if test_response.status_code == 200:

        print('Test request successful')
        url = test_url + '/market_chart'

        params = {
            'vs_currency': 'usd',
            'days': days
        }

        if interval != 'hourly':
            params['interval'] = interval

        response = requests.get(url, params=params)

        if response.status_code == 429:
            return pd.read_csv(filepath)

        coin_year_prices = response.json()
        dates = []
        prices = []

        for daily_price in coin_year_prices['prices']:
            dates.append(datetime.datetime.fromtimestamp(daily_price[0] / 1000))
            prices.append(daily_price[1])
        coin_year_prices_df = pd.DataFrame({'date': dates, 'price': prices})

        filer.write_to_csv(filepath, coin_year_prices_df)
        print('Coin parsing successful')

        return pd.read_csv(filepath)
    else:
        print('Test request failed, status code ' + str(test_response.status_code))
        print(test_response.text)
        return False

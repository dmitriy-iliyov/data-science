import requests


def parse():
    url = "https://api.yelp.com/v3/businesses/business_id_or_alias/reviews?limit=20&sort_by=yelp_sort"

    headers = {"accept": "application/json"}

    response = requests.get(url, headers=headers)

    print(response.text)


parse()
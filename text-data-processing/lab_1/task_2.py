import re


def do(data):
    data = re.sub(r'([^\w\s])([^\w\s]{1,100})', lambda x: x.group(1), data)
    data = re.sub(r'\s+', ' ', data)
    data = re.sub(r'([A-Z])([a-zA-Z]{1,100})', lambda x: x.group(1) + x.group(2).lower(), data)
    print(data)
    return data

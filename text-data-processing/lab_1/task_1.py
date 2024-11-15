
def do(text):
    print(text)
    len(text)
    print("count of \"e\" letters - " + str(text.count("e")))
    print("index of \"s\" letter - " + str(text.find("s")))
    try:
        text.index("was")
    except:
        print("ValueError: substring not found")
    print(text.upper())
    print(text.lower())
    print(text.title())
    print(text.capitalize())
    print(".".join(text))
    print(text.replace("1895", "2024"))
    print(text.split("I"))

from task_1 import Task1
from task_2 import Task2


# task 1
t1 = Task1()
result = t1.do("The annual awards ceremony was a star-studded event. This year, the Nobel Prize in Literature was awarded "
            "to a renowned author whose work has transcended cultural boundaries. The Booker Prize, one of the most "
            "prestigious literary awards, was given to a debut novelist who captivated the judges with their gripping "
            "narrative. In the world of film, the Academy Award for Best Picture went to a groundbreaking movie that "
            "tackled important social issues. The Best Actor award at the Academy Awards was received by a veteran "
            "performer known for their powerful performances. Meanwhile, the Golden Globe for Best Director was awarded "
            "to a visionary filmmaker who brought an epic story to life. On the music front, the Grammy Award for Album "
            "of the Year went to a popular artist whose album dominated the charts. The Best New Artist Grammy was "
            "awarded to a breakout star who has taken the music industry by storm. At the same time, the Pulitzer Prize "
            "for Music was awarded to a composer who has redefined contemporary classical music. In the television "
            "industry, the Emmy Award for Outstanding Drama Series was given to a show that has captivated audiences "
            "worldwide. The Emmy for Best Actress in a Drama Series went to a talented actress who delivered a stunning "
            "performance. Additionally, the Tony Award for Best Musical was awarded to a Broadway show that has received"
            " rave reviews from critics and audiences alike. Finally, the Fields Medal in Mathematics was awarded to a "
            "young mathematician whose work has made significant contributions to the field. The Turing Award, often "
            "regarded as the 'Nobel Prize of Computing', was awarded to a computer scientist whose research has had a "
            "profound impact on the industry.")

for item in result:
    print(item)

# task 2
t2 = Task2()
test_data = [
    "Can you recommend any good action movies?",
    "I'm in the mood for a horror movie tonight.",
    "Could you tell me the showtimes for the latest releases?",
    "I'm looking for a romantic comedy to watch with my partner.",
    "What's the rating for the new sci-fi movie?"
]
result = t2.do(test_data)
for item in result:
    print(f"Utterance: {item[0]}")
    print("Predicted Intentions:", item[1])
    print("\n")

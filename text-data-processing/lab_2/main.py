import re

import task_1 as t1
import task_2 as t2


with open('files/text5.txt', 'r') as file:
    content = re.sub(r'\s+', ' ', file.read())

t1.do(content)
t2.do()

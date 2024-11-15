import task_1 as t1
import task_2 as t2


with open('files/my_file.txt', 'r') as file:
    content = file.read()
    print(content)
    text = content[24:44]

t1.do(text)
processing_data = t2.do(content)
with open('files/new_my_file.txt', 'w') as file:
    file.write(processing_data)

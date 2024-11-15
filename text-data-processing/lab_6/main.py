import pandas as pd

from task_1 import Task1
from task_2 import Task2


# pandas settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# task_1
t1 = Task1()
t1.do_task()
# task_2
t2 = Task2()
t2.do_task()

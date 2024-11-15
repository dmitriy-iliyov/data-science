import pandas as pd
from nltk.corpus import gutenberg

import task_1 as t1
import task_2 as t2


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# task 1
t1.do_task_1()

# task 2
moby_dick = gutenberg.raw('melville-moby_dick.txt')
result = t2.do_task_2(moby_dick)
print(result)




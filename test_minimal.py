import pandas as pd
from rusket import fpgrowth

df = pd.DataFrame({"A": [True, False], "B": [False, True]})
res = fpgrowth(df, min_support=0.1)
print(res)

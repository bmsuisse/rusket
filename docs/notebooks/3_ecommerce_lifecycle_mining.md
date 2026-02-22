# Sequential Pattern Mining (PrefixSpan)

In standard Market Basket Analysis, we look at the items inside a *single checkout*. However, if we want to model **lifecycle purchasing** or **churn behavior**, we need an algorithm that natively understands *time*.

In this cookbook, we will mine **Sequential Patterns** using `rusket`'s blazing fast PrefixSpan implementation over an e-commerce clickstream log.


```python
import time

import pandas as pd

from rusket import prefixspan, sequences_from_event_log
```

## 1. The E-Commerce Event Log
We start with a classic log of distinct user events over time. This could be page views, checkout events, or support tickets.


```python
events = pd.DataFrame(
    {
        "user_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4],
        "timestamp": [
            "2024-01-01 10:00",
            "2024-01-05 10:05",
            "2024-01-10 10:10",
            "2024-01-02 11:00",
            "2024-01-07 11:05",
            "2024-01-03 09:00",
            "2024-01-04 09:05",
            "2024-01-09 09:10",
            "2024-01-01 12:00",
            "2024-01-08 12:00",
            "2024-01-15 12:00",
        ],
        "event_name": [
            "signup",
            "view_product",
            "add_to_cart",
            "signup",
            "view_product",
            "signup",
            "view_product",
            "checkout",
            "view_product",
            "checkout",
            "churn",
        ],
    }
)

# Ensure correct temporal ordering
events["timestamp"] = pd.to_datetime(events["timestamp"])
events.sort_values(["user_id", "timestamp"], inplace=True)
events.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>timestamp</th>
      <th>event_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2024-01-01 10:00:00</td>
      <td>signup</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2024-01-05 10:05:00</td>
      <td>view_product</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2024-01-10 10:10:00</td>
      <td>add_to_cart</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2024-01-02 11:00:00</td>
      <td>signup</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>2024-01-07 11:05:00</td>
      <td>view_product</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Compiling the Sequential Database
Rusket requires data grouped into discrete sequential arrays of integers per user. We provide a `sequences_from_event_log` helper to automatically convert your Pandas DataFrame into this required zero-copy format.


```python
sequences, label_mapping = sequences_from_event_log(
    events, user_col="user_id", time_col="timestamp", item_col="event_name"
)

print(f"Compiled {len(sequences)} distinct user sequences.")
print(f"Internal Mapping Table: {label_mapping}")
```

    ```text
    Compiled 4 distinct user sequences.
    Internal Mapping Table: {0: 'signup', 1: 'view_product', 2: 'add_to_cart', 3: 'checkout', 4: 'churn'}
    ```


## 3. Mining Sequential Patterns
Now we pass our compiled sequences into the `prefixspan` model. We will ask for patterns that happen to at least 2 independent users.


```python
# Mine patterns
t0 = time.time()
patterns_df = prefixspan(sequences, min_support=2)
print(f"Found {len(patterns_df)} sequential patterns in {time.time() - t0:.4f}s!")

# Restore the human-readable labels from our internal `label_mapping`
patterns_df["event_path"] = patterns_df["sequence"].apply(lambda seq: " → ".join([label_mapping[idx] for idx in seq]))

# Display the most frequent sequences
patterns_df.sort_values("support", ascending=False)[["support", "event_path"]]
```

    Found 5 sequential patterns in 0.0013s!





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>event_path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>view_product</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>signup</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>signup → view_product</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>view_product → checkout</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>checkout</td>
    </tr>
  </tbody>
</table>
</div>



Using these sequential outputs, businesses can automatically map out the 'Happy Path' to `checkout` vs the 'Failure Path' leading to `churn`.

#!/usr/bin/env python

# # Market Basket Analysis Cookbook
#
# Welcome to the `rusket` cookbook! This guide provides comprehensive examples on how to perform market basket analysis and generate recommendation rules efficiently.
# We will use Plotly for visualizations to get insights into our frequent itemsets and association rules.
#

# In[1]:


import numpy as np
import pandas as pd
import plotly.express as px
import polars as pl
import pyarrow.compute as pc

from rusket import association_rules, fpgrowth

# ## 1. Synthetic Dataset Generation
#
# Let's start by generating a synthetic retail dataset. We'll simulate a supermarket where customers buy various categories of items.
#

# In[2]:


np.random.seed(42)

items = [
    "Milk",
    "Bread",
    "Butter",
    "Eggs",
    "Cheese",
    "Yogurt",
    "Coffee",
    "Tea",
    "Sugar",
    "Apples",
    "Bananas",
    "Oranges",
    "Chicken",
    "Beef",
    "Fish",
    "Rice",
    "Pasta",
    "Tomato Sauce",
    "Onions",
    "Garlic",
]

n_transactions = 10_000
n_items = len(items)

# Simulate different purchase frequencies (power-law distribution)
probabilities = np.power(np.arange(1, n_items + 1, dtype=float), -0.7)
probabilities /= probabilities.max()
probabilities = np.clip(probabilities * 0.3, 0.01, 0.8)

data = np.random.rand(n_transactions, n_items) < probabilities
df = pd.DataFrame(data, columns=items)

print(f"Dataset shape: {df.shape}")
df.head()


# ## 2. Frequent Pattern Mining
#
# We'll extract frequent itemsets using the blazing-fast `fpgrowth` algorithm from `rusket`.
# We set `min_support=0.05`, meaning an itemset must appear in at least 5% of all transactions.
#

# In[3]:


# Extract frequent itemsets
fi = fpgrowth(df, min_support=0.05, use_colnames=True)

print(f"Found {len(fi)} frequent itemsets.")
fi.sort_values(by="support", ascending=False).head(10)


# ### Visualizing Frequent Itemsets
#
# Let's plot the top 20 most frequent itemsets to understand what items are bought together most often.
#

# In[4]:


# Get top 20 itemsets
top_fi = fi.sort_values(by="support", ascending=False).head(20).copy()
# Format itemsets as strings
top_fi["itemsets_str"] = top_fi["itemsets"].apply(lambda x: " + ".join(list(x)))

fig = px.bar(
    top_fi,
    x="support",
    y="itemsets_str",
    orientation="h",
    title="Top 20 Frequent Itemsets by Support",
    labels={"support": "Support", "itemsets_str": "Itemset"},
    color="support",
    color_continuous_scale="Viridis",
)
fig.update_layout(yaxis={"categoryorder": "total ascending"})
fig.show()


# ## 3. Generating Association Rules
#
# Now that we have our frequent itemsets, we can generate association rules.
# We'll use the fundamental `confidence` metric, setting a threshold to filter out weak rules.
#

# In[5]:


# Generate association rules
rules = association_rules(fi, num_itemsets=len(df), min_threshold=0.3)

print(f"Generated {len(rules)} association rules.")
rules.sort_values(by="lift", ascending=False).head()


# ### Filtering Rules
#
# Often, we'll want to filter rules based on multiple metrics. For example, rules with high confidence *and* high lift. High lift (> 1) indicates that the items are positively correlated.
#

# In[6]:


# Filter for strong rules
strong_rules = rules[(rules["confidence"] > 0.4) & (rules["lift"] > 1.2)]
strong_rules = strong_rules.sort_values(by="lift", ascending=False)
strong_rules.head(10)


# ### Visualizing Association Rules
#
# A scatter plot of Support vs. Confidence is great for identifying the most valuable rules. We'll use color to represent `lift`.
#

# In[7]:


fig = px.scatter(
    rules,
    x="support",
    y="confidence",
    color="lift",
    hover_data=["antecedents", "consequents"],
    title="Association Rules: Support vs Confidence",
    color_continuous_scale="Plasma",
)
fig.show()


# ## 4. Seamless Polars Integration
#
# `rusket` works natively with `polars` without requiring expensive conversions.
#

# In[8]:


df_pl = pl.from_pandas(df)
fi_pl = fpgrowth(df_pl, min_support=0.05, use_colnames=True)

# Generate rules directly from Polars dataframe
rules_pl = association_rules(fi_pl, num_itemsets=df_pl.height, min_threshold=0.3)
rules_pl.head(5)


# ## 5. Working with PyArrow Outputs
#
# To achieve blazing-fast performance, `rusket` returns itemsets as zero-copy **PyArrow `ListArray`** structures backed by Pandas.
# This eliminates Python object overhead and allows you to process millions of rules with minimal memory.
#
# ### Querying PyArrow Itemsets
#
# Because the `itemsets` column uses `pd.ArrowDtype(pa.list_(pa.string()))`, standard Python `set` equality operations won't work perfectly out of the box.
# You should use PyArrow compute functions or cast them to Python sets when filtering row-by-row.
#

# In[9]:


# Extract itemsets using PyArrow compute to find transactions containing a specific item
# For example, let's find all itemsets that contain 'Milk'
contains_milk = pc.list_element(fi["itemsets"].array, 0) == "Milk"

# Alternatively, if you need to do complex Python-native filtering, you can convert to sets:
# (Note: this materializes Python objects, so only do this on filtered sub-sets!)
top_10 = fi.head(10).copy()
top_10["python_sets"] = top_10["itemsets"].apply(set)

print("Zero-Copy PyArrow Array Dtype:", fi["itemsets"].dtype)
top_10.head()


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


import pyarrow.compute as pc

# Extract itemsets using PyArrow compute to find transactions containing a specific item
# For example, let's find all itemsets that contain 'Milk'
contains_milk = pc.list_element(fi['itemsets'].array, 0) == "Milk"

# Alternatively, if you need to do complex Python-native filtering, you can convert to sets:
# (Note: this materializes Python objects, so only do this on filtered sub-sets!)
top_10 = fi.head(10).copy()
top_10['python_sets'] = top_10['itemsets'].apply(set)

print("Zero-Copy PyArrow Array Dtype:", fi['itemsets'].dtype)
top_10.head()


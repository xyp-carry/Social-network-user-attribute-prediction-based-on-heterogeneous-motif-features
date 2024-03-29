import pandas as pd
import networkx as nx
import pickle
import random

with open('user_dict.pkl', 'rb') as f:
    a = pickle.load(f)

print(a[50471224])
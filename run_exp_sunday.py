import numpy as np
import pandas as pd
from docplex.mp.model import Model
import glob, re
from collections import Counter
import time

#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
from vrp_eval import *
from mat_functs import *
from testFile import *

dayofweek = 'DIMANCHE'
weighing = 'exp'

results_df, actual, res1, res2, mat1, mat2 = solve('all',7, weighing, dayofweek)
fname = 'FiveHours_'+dayofweek+'_'+weighing+'.csv'
results_df.to_csv(fname, mode='a', header=True, index=True)
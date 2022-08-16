#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:

def solve(data_range = 'all', n_test = 7, weighing = 'unif', dayofweek = 'DIMANCHE'):

    # initialize df of results 
    results_df = pd.DataFrame()
    results_df = pd.DataFrame(columns = ['Date','Target','Solution', 'Status', 'SD','%SD','AD','%AD', 'Gap', 'Time'])

#     days = ['DIMANCHE', 'LUNDI', 'MARDI', 'MERCREDI', 'JEUDI', 'VENDREDI', 'SAMEDI']
    days = [dayofweek]

    for day in days:

            # create list of files (same day) for test and target data extraction
            if data_range == 'all':
                file_list = [file for file in sorted(glob.glob("*.csv")) if day in file][:-5]
            elif data_range == '18':
                file_list = [file for file in sorted(glob.glob("*.csv")) if file.startswith('18') and day in file]
            elif data_range == '19':
                file_list = [file for file in sorted(glob.glob("*.csv")) if file.startswith('19') and day in file]	

            for x in range(len(file_list)-n_test, len(file_list)):
                if weighing == 'fxd':
                    train = file_list[:-n_test]
                else:
                    train = file_list[:x]
                test = file_list[x]

#                 2MM
                proba_matrix2, s_list2, idx_list2 = create3d_mat(train, test, weighing)
#                 1MM
                proba_matrix1, s_list1, idx_list1 = create_pmatrix(train, test, weighing)

                res_plan1, stat1, actual_for_comp1, gap1, sol_time1 = solve_vrp(train, test, s_list1, proba_matrix1, idx_list1, 'first', day)
                print('sd: ', eval_sd(actual_for_comp1,res_plan1)[1], 'ad: ', eval_ad(actual_for_comp1,res_plan1)[1])
                
                res_plan, stat, actual_for_comp, gap, sol_time = solve_vrp(train, test, s_list2, proba_matrix2, idx_list2, 'second', day)
                results_df.loc[len(results_df)] = [test.split(" ")[0], actual_for_comp, res_plan, stat,
                                               eval_sd(actual_for_comp,res_plan)[1], eval_sd(actual_for_comp,res_plan)[2],
                                               eval_ad(actual_for_comp,res_plan)[1], eval_ad(actual_for_comp,res_plan)[2], gap, sol_time]
            
    return results_df, actual_for_comp, res_plan1, res_plan, proba_matrix1, proba_matrix2

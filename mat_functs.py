import numpy as np
import pandas as pd
from docplex.mp.model import Model
import glob
from collections import Counter
import time

def convert(full_route_list,fin_all_stops):
    # converts list of stop names to list of stop numbers
    # stop numbers from indices of stops in fin_all_stops list
    
    list_to_num = []
    for route in full_route_list:
        list_to_num.append([fin_all_stops.index(stop) for stop in route])
    
    return list_to_num

def get_route(file):
    # returns full route of file
    # stop names; function used in transition matrix construction
    
    df = pd.read_csv(file)
    data = df[df['active']==1][['order','name']]
    
    num_of_routes = len(data[data['order']==1]) + len(data[data['name'].str.contains('SUMY')])
    
    # initialize; fix number of routes
    route_list = []
    for i in range(num_of_routes):
        route_list.append(['SUMY'])

    route_cnt = 0
    for i in range(len(data)):
        if 'SUMY' not in data.iloc[i]['name']:
            if data.iloc[i]['order'] != 1:
                route_list[route_cnt-1].append(data.iloc[i]['name'])
            else:
                route_cnt += 1
                route_list[route_cnt-1].append(data.iloc[i]['name'])
        else:
            route_cnt += 1

    # append 'SUMY' to end of each route
    for i in range(num_of_routes):
        route_list[i].insert(0,'SUMY')
        route_list[i].append('SUMY')
        route_list[i].append('SUMY')

    return route_list

# https://stackoverflow.com/questions/47432632/flatten-multi-dimensional-array-in-python-3
def flatten(something):
    if isinstance(something, (list, tuple, set, range)):
        for sub in something:
            yield from flatten(sub)
    else:
        yield something
        
# def mylog(x, least):
#     if x == 0:
#         return np.log(least)
#     else:
#         return np.log(x)
        
def create3d_mat(train, test, weighing = 'unif'):

        # gather all stops visited in train + test
        all_stops_set = set()
        for instance in train + [test]:
            all_stops_set.update(set(flatten(get_route(instance))))
        all_stops_list = list(all_stops_set)

        all_stops_list.remove('SUMY')
        all_stops_list.insert(0, 'SUMY')

        # initialize matrix of size len(all_stops_set)
        mat_dim = len(all_stops_set)
        proba_matrix = np.zeros((mat_dim, mat_dim, mat_dim))

        # collect historical data
        data = [] # list of route plans
        for instance in train:
            data.append(get_route(instance))

        # construct 3D matrix
        for (hist_ind,plan) in enumerate(data):

            set_tt = set(flatten(get_route(test)))
            set_tr = set(flatten(plan))
            
            if weighing == 'unif':
                weight = 1
            elif weighing == 'fxd':
                weight = 1
            elif weighing == 'wtd':
                weight = (hist_ind+1)/len(data)
            elif weighing == 'wtd2':
                weight = ((hist_ind+1)/len(data))**2 
            elif weighing == 'jac':
                weight = len(set_tt.intersection(set_tr))/len(set(set_tt).union(set_tr))
            elif weighing == 'jac2':
                weight = (len(set_tt.intersection(set_tr))/len(set(set_tt).union(set_tr)))**2
            elif weighing == 'exp':
                weight = 0.7*(1-0.7)**(len(data)-(hist_ind+1))
            
            for route in plan:
                for i in range(1,len(route)-1):
                    idx_im1 = all_stops_list.index(route[i-1])
                    idx_i   = all_stops_list.index(route[i])
                    idx_ip1 = all_stops_list.index(route[i+1])
                    proba_matrix[idx_im1, idx_i, idx_ip1] += weight
        
        # reduce matrix to include only relevant stops
        test_stops_list = list(set(flatten(get_route(test))))
        
        test_stops_list.remove('SUMY')
        test_stops_list.insert(0, 'SUMY')
                
        test_idx_list = [all_stops_list.index(i) for i in test_stops_list]

        init_matrix = proba_matrix[np.ix_(test_idx_list, test_idx_list, test_idx_list)]

#         print(np.min(init_matrix[np.nonzero(init_matrix)]))
#         print(init_matrix) 

#         # 1. Normalize 
        init_matrix = init_matrix + 0.0000001
        init_matrix = init_matrix/init_matrix.sum(axis=-1,keepdims=True)
        # OR
#         init_matrix = init_matrix + 0.0000001
#         init_matrix = init_matrix/init_matrix.sum()
        # OR
#         for i in range(len(init_matrix)):
#             sub = init_matrix[i]
#             init_matrix[i] = sub/sub.sum()
        
        # 2. Eliminate nans        
        init_matrix = np.nan_to_num(init_matrix)
#         init_matrix = init_matrix + 1
        
        # 3. Take negative log        
        init_matrix = -np.log(init_matrix)
#         init_matrix[init_matrix >= 100] = 0
#         init_matrix[init_matrix == 0] = np.max(init_matrix) + 0.000001
#         print(np.max(init_matrix))
#         print(np.min(init_matrix))
        # OR
#         init_matrix = -init_matrix     

        return [init_matrix, all_stops_list, test_idx_list]
    
def create_pmatrix(tr_set, tt_file, weighing):
# creates probability matrix from the training list

    # gather all stops visited in train + test
    all_stops_set = set()
    for instance in tr_set + [tt_file]:
        all_stops_set.update(set(flatten(get_route(instance))))
    all_stops_list = list(all_stops_set)
    
    all_stops_list.remove('SUMY')
    all_stops_list.insert(0, 'SUMY')

    # initialize matrix of size len(all_stops_list)
    mat_dim = len(all_stops_list)
    proba_matrix = np.zeros((mat_dim, mat_dim))
    
    # construct adjacency matrix for each train instance
    for (i, instance) in enumerate(tr_set):
        
        set_tt = set(flatten(get_route(tt_file)))
        set_tr = set(flatten(get_route(instance)))
        
        if weighing == 'unif':
            weight = 1
        elif weighing == 'fxd':
            weight = 1
        elif weighing == 'wtd':
            weight = (i+1)/len(tr_set)
        elif weighing == 'wtd2':
            weight = ((i+1)/len(tr_set))**2
        elif weighing == 'wtd3':
            weight = ((i+1)/len(tr_set))**3
        elif weighing == 'wtd4':
            weight = ((i+1)/len(tr_set))**4
        elif weighing == 'wtd5':
            weight = ((i+1)/len(tr_set))**5
        elif weighing == 'wtd10':
            weight = ((i+1)/len(tr_set))**10
        elif weighing == 'wtd15':
            weight = ((i+1)/len(tr_set))**15
        elif weighing == 'wtd20':
            weight = ((i+1)/len(tr_set))**20
        elif weighing == 'jac':
            weight = len(set_tt.intersection(set_tr))/len(set(set_tt).union(set_tr))
        elif weighing == 'jac2':
            weight = (len(set_tt.intersection(set_tr))/len(set(set_tt).union(set_tr)))**2
        elif weighing == 'jac3':
            weight = (len(set_tt.intersection(set_tr))/len(set(set_tt).union(set_tr)))**3
        elif weighing == 'jac4':
            weight = (len(set_tt.intersection(set_tr))/len(set(set_tt).union(set_tr)))**4
        elif weighing == 'jac5':
            weight = (len(set_tt.intersection(set_tr))/len(set(set_tt).union(set_tr)))**5
        elif weighing == 'jac10':
            weight = (len(set_tt.intersection(set_tr))/len(set(set_tt).union(set_tr)))**10
        elif weighing == 'jac15':
            weight = (len(set_tt.intersection(set_tr))/len(set(set_tt).union(set_tr)))**15
        elif weighing == 'jac20':
            weight = (len(set_tt.intersection(set_tr))/len(set(set_tt).union(set_tr)))**20
        elif weighing == 'exp':
            weight = 0.7*(1-0.7)**(len(tr_set)-(i+1))
            
        routing = get_route(instance)
        for route in routing:
            for i in range(len(route)-1):
                proba_matrix[all_stops_list.index(route[i])][all_stops_list.index(route[i+1])] += weight

    # reduce matrix to include only relevant stops
    test_stops_list = list(set(flatten(get_route(tt_file))))

    test_stops_list.remove('SUMY')
    test_stops_list.insert(0, 'SUMY')

    test_idx_list = [all_stops_list.index(i) for i in test_stops_list]

    init_matrix = proba_matrix[np.ix_(test_idx_list, test_idx_list)]

    # 1. Laplace smoothing    
    init_matrix = init_matrix + 1

    # 2. Normalize; convert to probabilities
    init_matrix = init_matrix/init_matrix.sum(axis=1)[:,None]
    # OR
#     init_matrix = init_matrix/init_matrix.sum()
    
    # 3. Eliminate zeros
    init_matrix = np.nan_to_num(init_matrix)
    
    # 4.Take negative log
    init_matrix = -np.log(init_matrix)
#     init_matrix[init_matrix >= 100] = 0
#     init_matrix[init_matrix == 0] = np.max(init_matrix)
    # OR
#     init_matrix = -init_matrix

    return [init_matrix, all_stops_list, test_idx_list]

def solve_vrp(train, test, stop_list, proba_matrix, idx_list, order, day):
        
        print(test)
        all_stops_list = stop_list

        actual = convert(get_route(test),all_stops_list)

        n = len(set(flatten(actual)))-1 # len(stop_list)-1
        
        N = [i for i in range(1, n+1)]
        rt_count = len(actual)

#        Q = n
        
        if day == 'MARDI':
            Q = 14
        elif day == 'SAMEDI':
            Q = 25 # 20?
        else:
            Q = 13

        # add demands
        cap = []
        
        for i in range(len(stop_list)):
            if any(['DATS' in stop_list[i], 'Dats' in stop_list[i]]):
                cap.append(1)
            elif any(['TFCO LH' in stop_list[i], 'TFCO WOO' in stop_list[i], 'TFCO IXL' in stop_list[i]]):
                cap.append(4)
            elif 'ST CATH' in stop_list[i]:
                if day == 'DIMANCHE' or day == 'LUNDI':
                    cap.append(12)
                else:
                    cap.append(10)
            elif any(['TONGRES' in stop_list[i], 'HANKAR' in stop_list[i], 'TFCO OVErijse' in stop_list[i], 'TFCO VDK' in stop_list[i]],):
                cap.append(3)
            else:
                cap.append(2)

        V = [0] + N
#         q = dict(zip(V, cap))
        q = {i : 1 for i in N} # dictionary of demands

        # create set of arcs
        A = [(i,j) for i in V for j in V if i!=j]
     
        # solve using CPLEX
        mdl = Model('CVRP')
        
#         # gap tolerance
#         mdl.parameters.mip.tolerances.mipgap=0.02;
#         mdl.parameters.emphasis.mip=3;
#         mdl.parameters.mip.strategy.search=1;

        x = mdl.binary_var_dict(A, name='x')
        u = mdl.continuous_var_dict(N, ub=Q, name='u') 
        
# 1MM ------------------------------------------------

        if order == 'first':
            mdl.minimize(mdl.sum((proba_matrix[i,j])*x[i,j] for i,j in A))
    
# ----------------------------------------------------
        
# 2MM ------------------------------------------------
        
        elif order == 'second':
            # objective function
            depot_list = []
            square_list = []
            for i in V:
                if i == 0: # depot
                    for j in V:
                        if j != i:
    #                         print("adding depot cost:",i,j, proba_matrix[0,0,j])
                            depot_list.append( proba_matrix[0,0,j]*x[0,j] ) # np.log(proba_matrix[0,0,j])*x[0,j]
        
                            for k in V:
                                if k == i or k == j:
                                    continue
                                square_list.append( proba_matrix[0,j,k]*x[0,j]*x[j,k] )
                            
                else: # not depot
                    for j in V:
                        if j == 0 or j == i:
                            continue # depot and loop not allowed
                        for k in V:
                            if k == i or k == j:
                                continue # depot allowed, loops not
#                             if k == 0:
#                                 depot_list.append( proba_matrix[i,j,0]*x[i,j]*x[j,0] ) # np.log(proba_matrix[i,j,0])*x[j,0] 
    #                         print("adding sq cost:",i,j,k, proba_matrix[i,j,k])
                            square_list.append( proba_matrix[i,j,k]*x[i,j]*x[j,k] ) # np.log(proba_matrix[i,j,k])*x[i,j]*x[j,k] 
    #                         square_list.append( proba_matrix[i,j,k]*(x[i,j]*x[i,j]+x[j,k]*x[j,k]) )
            mdl.minimize(mdl.sum(depot_list + square_list)) # depot_list + square_list

# OLD ------------------------------------------------

#             mdl.minimize(mdl.sum(proba_matrix[i,j,k]*x[i,j]*x[j,k] for i,j in A for j,k in A if i!=j and j!=k and k!=i))

        # constraints
        mdl.add_constraints(mdl.sum(x[i,j] for j in V if j!=i)==1 for i in N)
        mdl.add_constraints(mdl.sum(x[i,j] for i in V if i!=j)==1 for j in N)
        mdl.add_indicator_constraints(mdl.indicator_constraint(x[i,j], u[i]+q[j]==u[j]) for i,j in A if i!=0 and j!=0)
        mdl.add_constraints(u[i]>=q[i] for i in N)

        # fix number of routes
        mdl.add_constraints(mdl.sum(x[0,j] for j in N)<=rt_count for j in N)
        mdl.add_constraints(mdl.sum(x[0,j] for j in N)>=rt_count for j in N)

        # set time limit
#         mdl.parameters.timelimit = 3600*12 # 5 hours (86400 - 1 day)
        mdl.parameters.timelimit = 20 # 5 hours (86400 - 1 day)
        
        start = time.time()

        # set (log_output=True) to display solution
        solution = mdl.solve()
#         solution = mdl.solve(log_output=True)
#         print("gap:",mdl.MIP.get_mip_relative_gap())
#         print("gap:",mdl.get_solve_details().mip_relative_gap)
        gap = mdl.get_solve_details().mip_relative_gap
        
        end = time.time()

        sol_time = end-start
        
        print("solution time:",sol_time)

        print('mdl.objective_value =', mdl.objective_value)
        
        active_arcs = [a for a in A if x[a].solution_value>0.9]
        start_arcs = [a for a in active_arcs if a[0]==0]

        res_plan = []
        for arc in start_arcs:    
            res_route = [0,arc[1]]
            while res_route[-1]!=0:
                for a in active_arcs:
                    if res_route[-1]==a[0]:
                        res_route.append(a[1])
            res_plan.append(res_route)

        # convert stops in actual to correspond to stops in test
        
        d = dict(zip(idx_list, V))
#         print(idx_list)
        
        new_actual = []
        
        for route in actual:
            new_route = []
            for s in route:
                new_route.append(d[s])
            new_actual.append(new_route)
        
        actual_for_comp = [route[1:-1] for route in new_actual]
        return [res_plan, str(solution.solve_status).split(".")[1].split("_")[0], actual_for_comp, gap, sol_time]
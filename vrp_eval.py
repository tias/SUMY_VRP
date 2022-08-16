import numpy as np

# arc difference
def eval_ad(P, A):
    P_set = set()
    for sublist in P:
        for i in range(len(sublist)-1):
            P_set.add( (sublist[i],sublist[i+1]) )
        
    A_set = set()
    for sublist in A:
        for i in range(len(sublist)-1):
            A_set.add( (sublist[i],sublist[i+1]) )
    
    result = set(A_set).difference((set(P_set)))
    
    # assert(len(A_set) == len(P_set))
    # diffset, diffcount, diffrelative
    return result, len(result), len(result)/len(A_set)


# locate minimum
# save location in a list (idx_list)
# before next iteration, increase all values in row and column where minimum is located by a large number
def get_best_route_mapping(P, A):
    # create initial matrix of symmetric differences
    np_matrix = np.zeros((len(P),len(A)))
    for x in range(len(P)):
        for y in range(len(A)):
            np_matrix[x][y] = len(set(P[x]).symmetric_difference(set(A[y])))

    idx_list = []
    while len(idx_list) < len(A):
        # find smallest (x,y) in matrix
        (idx_r, idx_c) = np.where(np_matrix == np.nanmin(np_matrix))
        (r, c) = (idx_r[0], idx_c[0]) # avoid duplicates
        idx_list.append( (r, c) )

        # blank out row/column selected
        np_matrix[r,:] = np.NaN
        np_matrix[:,c] = np.NaN
        #print(np_matrix)
        #print(len(idx_list), len(Act))
    
    return idx_list

# get all unique stops
def allstops(R):
    result = set()
    for route in R:
        result.update(route)
    return result

# stop difference
def eval_sd(P, A):
    # get paired mapping
    idx_list = get_best_route_mapping(P, A)
    
    diff = set()
    for (idx_P, idx_A) in idx_list:
        diff.update(set(P[idx_P]).symmetric_difference(set(A[idx_A])))
    
    nr_stops = len(allstops(A))
    # diffset, diffcount, diffrelative
    return diff, len(diff), len(diff)/nr_stops
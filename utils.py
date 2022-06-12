def gen_adjacency_matrix_unweighted(n=10, p=0.5):
    '''generate a binary symmetric matrix'''
    mat = np.random.rand(n, n)

    for i in range(n):
        for j in range(0, i + 1):
            if mat[i,j] <= p:
                mat[i, j] = 1
            else:
                mat[i, j] = 0
            
            mat[j,i] = mat[i,j] # symmetric
        mat[i, i] = 0
    return mat

def gen_adjacency_matrix_weighted(n=10, p=0.5):
    '''generate a weighted symmetric matrix'''
    mat = np.random.rand(n, n)

    for i in range(n):
        for j in range(0, i + 1):
            if mat[i,j] > p:
                mat[i, j] = 0
            mat[j,i] = mat[i,j] # symmetric
        mat[i, i] = 0 
    return mat
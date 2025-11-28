# Funcții pentru calculul distanțelor
import numpy as np


# Calculează distanța între doi vectori folosind o normă specificată
# Args: vector1 - Primul vector, vector2 - Al doilea vector, norma - Tipul normei ('Manhattan', 'Euclidian', 'Infinit', 'Cosinus')
# Returns: Distanța calculată
def CALC_DISTANTA_NORMA(vector1, vector2, norma):
    if norma == 'Manhattan':
        return np.linalg.norm(vector1 - vector2, ord=1)
    elif norma == 'Euclidian':
        return np.linalg.norm(vector1 - vector2, ord=2)
    elif norma == 'Infinit':
        return np.linalg.norm(vector1 - vector2, ord=np.inf)
    elif norma == 'Cosinus':
        v1_norm = vector1 / (np.linalg.norm(vector1) + 1e-10)
        v2_norm = vector2 / (np.linalg.norm(vector2) + 1e-10)
        return 1 - np.dot(v1_norm, v2_norm)
    else:
        raise ValueError(f"Norma necunoscuta: {norma}")


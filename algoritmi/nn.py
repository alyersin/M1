# Algoritmul Nearest Neighbor (NN)
import numpy as np
from utils.distante import CALC_DISTANTA_NORMA


# Algoritmul Nearest Neighbor - găsește cea mai apropiată poză
# Args: A - Matricea de antrenare (nrPixeli x nrPoze), poza_cautata - Poza de căutat (vectorizată)
#       norma - Norma de distanță, etichete_antrenare - Etichetele pozelor (opțional)
# Returns: pozitia, eticheta_gasita
def ALG_NN(A, poza_cautata, norma, etichete_antrenare=None):
    total_imagini_antrenament = A.shape[1]
    distante = np.zeros(total_imagini_antrenament)
    
    for i in range(total_imagini_antrenament):
        distante[i] = CALC_DISTANTA_NORMA(A[:, i], poza_cautata, norma)
    
    pozitia = np.argmin(distante)
    
    eticheta_gasita = None
    if etichete_antrenare is not None:
        eticheta_gasita = etichete_antrenare[pozitia]
    
    return pozitia, eticheta_gasita


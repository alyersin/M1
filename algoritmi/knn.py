# Algoritmul K-Nearest Neighbors (KNN)
import numpy as np
import statistics
from utils.distante import CALC_DISTANTA_NORMA


# Algoritmul K-Nearest Neighbors - găsește cea mai comună etichetă din k vecini
# Args: A - Matricea de antrenare (nrPixeli x nrPoze), poza_cautata - Poza de căutat (vectorizată)
#       k - Numărul de vecini, norma - Norma de distanță, etichete_antrenare - Etichetele (opțional), poze_per_persoana - (opțional)
# Returns: pozitia, eticheta_gasita
def ALG_KNN(A, poza_cautata, k, norma, etichete_antrenare=None, poze_per_persoana=8):
    total_imagini_antrenament = A.shape[1]
    distante = np.zeros(total_imagini_antrenament)
    
    for i in range(total_imagini_antrenament):
        distante[i] = CALC_DISTANTA_NORMA(A[:, i], poza_cautata, norma)
    
    indici_vecini_apropiati = np.argsort(distante)[:k]
    
    if etichete_antrenare is not None:
        etichete_vecini = etichete_antrenare[indici_vecini_apropiati]
        eticheta_gasita = statistics.mode(etichete_vecini)
        
        pozitii_persoana = [i for i, et in enumerate(etichete_antrenare) if et == eticheta_gasita]
        pozitia = pozitii_persoana[0]
    else:
        pozitia = indici_vecini_apropiati[0]
        eticheta_gasita = None
    
    return pozitia, eticheta_gasita


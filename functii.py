import numpy as np
import cv2
import os
import time
import statistics
import matplotlib.pyplot as plt


def LOAD_IMGS(folder_path='att_faces', nr_persoane=40, poze_per_persoana=10):
    poze = []
    etichete = []
    
    for persoana in range(1, nr_persoane + 1):
        folder = os.path.join(folder_path, f's{persoana}')
        if not os.path.exists(folder):
            continue
        
        for numar_poza in range(1, poze_per_persoana + 1):
            cale = os.path.join(folder, f'{numar_poza}.pgm')
            if os.path.exists(cale):
                poza = cv2.imread(cale, 0)
                if poza is not None and poza.shape == (112, 92):
                    poze.append(poza)
                    etichete.append(persoana - 1)
    
    return {
        'poze': poze,
        'etichete': np.array(etichete),
        'nr_persoane': len(set(etichete))
    }


def MATRICE_ANTRENARE(baza_date, poze_antrenare=8):
    poze = baza_date['poze']
    etichete = baza_date['etichete']
    nr_persoane = baza_date['nr_persoane']
    
    dimensiune_vector = 112 * 92
    
    imagini_antrenare = []
    etichete_antrenare = []
    imagini_test = []
    etichete_test = []
    
    for persoana in range(nr_persoane):
        index = [i for i, et in enumerate(etichete) if et == persoana]
        index.sort()
        
        for idx in index[:poze_antrenare]:
            imagini_antrenare.append(poze[idx])
            etichete_antrenare.append(persoana)
        
        for idx in index[poze_antrenare:]:
            imagini_test.append(poze[idx])
            etichete_test.append(persoana)
    
    A = np.zeros((dimensiune_vector, len(imagini_antrenare)))
    for i, poza in enumerate(imagini_antrenare):
        A[:, i] = poza.reshape(dimensiune_vector)
    
    A_test = np.zeros((dimensiune_vector, len(imagini_test)))
    for i, poza in enumerate(imagini_test):
        A_test[:, i] = poza.reshape(dimensiune_vector)
    
    return A, np.array(etichete_antrenare), A_test, np.array(etichete_test)


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



# Funcții pentru încărcarea și prelucrarea datelor
import numpy as np
import cv2
import os


# Încarcă imaginile din folderul specificat
# Args: folder_path - Calea către folderul cu poze, nr_persoane - Numărul de persoane, poze_per_persoana - Numărul de poze per persoană
# Returns: Dict cu 'poze', 'etichete', 'nr_persoane'
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


# Creează matricele de antrenare și test din baza de date
# Args: baza_date - Dict returnat de LOAD_IMGS, poze_antrenare - Numărul de poze de antrenare per persoană
# Returns: A, etichete_antrenare, A_test, etichete_test
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


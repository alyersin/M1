# Răspunsuri la Întrebările Profesoarei

## 1. Ce este reprezentant de clasă?

**Răspuns:** Un reprezentant de clasă este o poză care reprezintă întreaga clasă (persoană). În loc să comparăm cu toate pozele unei persoane, comparăm doar cu reprezentantul.

**În cod:** Funcția `MATRICE_REPREZENTANTI()` din `algoritmi/eigenfaces.py` (linia 72)

**Două variante:**
- **a) Media pozelor:** Media tuturor pozelor de antrenare ale persoanei
- **b) Poză aleatorie:** O poză aleatorie a persoanei

**Dimensiune:** Matricea RC are dimensiunea (nrPixeli × nrPersoane) = (10304 × 40)

---

## 2. Cum este inițializarea matricei?

**Răspuns:** Matricea A este inițializată în funcția `MATRICE_ANTRENARE()` din `utils/date.py`.

**Pași:**
1. Se creează o matrice zero: `A = np.zeros((dimensiune_vector, len(imagini_antrenare)))`
2. Dimensiune: (10304 pixeli × 320 poze de antrenare)
3. Fiecare poză este vectorizată (112 × 92 → 10304 pixeli) și pusă pe o coloană
4. `A[:, i] = poza.reshape(dimensiune_vector)`

**Cod:** `utils/date.py`, linia 61-63

---

## 3. Ce e diferit la SVD față de PCA?

**Răspuns:**

| Aspect | SVD | PCA |
|-------|-----|-----|
| **Metodă** | Singular Value Decomposition | Principal Component Analysis |
| **Matrice calculată** | Nu calculează C sau L direct | Calculează C (neoptimizată) sau L (optimizată) |
| **Proces** | Aplică SVD direct pe A.T | Calculează vectorii proprii ai C sau L |
| **Eficiență** | Mai eficientă computațional | Mai lentă (mai ales varianta neoptimizată) |
| **Rezultat** | U, s, Vt → HQPB = Vt.T | Vectorii proprii ai C/L → HQPB |

**În cod:** 
- SVD: `algoritmi/eigenfaces.py`, linia 23-38
- PCA: linia 39-59 (neoptimizată) sau 60-85 (optimizată)

---

## 4. Este optimizat sau neoptimizat?

**Răspuns:** Aplicația suportă **AMBELE variante**:

### Varianta Neoptimizată (PCA cu matricea C):
- **Matricea C:** `C = A @ A.T` 
- **Dimensiune:** (10304 × 10304) - **FOARTE MARE!**
- **Problema:** Calculul vectorilor proprii este foarte lent
- **Cod:** `algoritmi/eigenfaces.py`, linia 39-59, când `metoda='PCA'`

### Varianta Optimizată (PCA cu matricea L):
- **Matricea L:** `L = A.T @ A`
- **Dimensiune:** (320 × 320) - **MULT MAI MICĂ!**
- **Avantaj:** Calculul vectorilor proprii este mult mai rapid
- **Cod:** `algoritmi/eigenfaces.py`, linia 60-85, când `metoda='PCA_optimizata'`

**De ce e optimizată?**
- L are 320×320 = 102,400 elemente
- C are 10304×10304 = 106,172,416 elemente
- **L este de ~1000 de ori mai mică!**

---

## 5. Eigenfaces cu reprezentanți de clasă

**Răspuns:** DA, este implementat!

**Funcție:** `PREPROCESARE_EIGENFACES_REPREZENTANTI()` din `algoritmi/eigenfaces.py` (linia 99)

**Două variante pentru HQPB:**
- **a) 'clasic':** HQPB calculat din toate pozele (A), dar proiectăm doar reprezentanții (RC)
- **b) 'direct':** HQPB calculat direct din reprezentanți (RC)

**Avantaj:** În loc să calculăm 320 de distanțe, calculăm doar 40 (câte un reprezentant per persoană)

---

## 6. Este PCA sau SVD ce prezentați acolo?

**Răspuns:** **AMBELE!** Aplicația suportă:
- **SVD** (varianta implicită, folosită în interfață)
- **PCA neoptimizată** (cu matricea C)
- **PCA optimizată** (cu matricea L)

**În interfață:** Momentan folosește SVD (implicit), dar codul suportă toate variantele.

---

## 7. Eigenfaces clasic - optimizat sau neoptimizat?

**Răspuns:** **AMBELE variante sunt implementate!**

### Varianta Neoptimizată:
- **Matricea C:** `C = A @ A.T` (10304 × 10304)
- **Vectorii proprii:** Calculați direct din C
- **Cod:** `algoritmi/eigenfaces.py`, linia 39-59

### Varianta Optimizată:
- **Matricea L:** `L = A.T @ A` (320 × 320)
- **Vectorii proprii:** Calculați din L, apoi înmulțiți cu A
- **Cod:** `algoritmi/eigenfaces.py`, linia 60-85

**Diferența:** Varianta optimizată calculează vectorii proprii ai unei matrice mult mai mici (L), apoi îi transformă în vectorii proprii ai C prin înmulțire cu A.

---

## 8. Ce e matricea L?

**Răspuns:** Matricea L este matricea de covarianță **optimizată**.

**Formula:** `L = A.T @ A`

**Dimensiuni:**
- A: (10304 × 320)
- A.T: (320 × 10304)
- **L: (320 × 320)** ← Mult mai mică decât C!

**De ce e optimizată:**
- În loc să calculăm vectorii proprii ai C (10304×10304), calculăm vectorii proprii ai L (320×320)
- Apoi transformăm: `HQPB = A @ v_L` (unde v_L sunt vectorii proprii ai L)

**Cod:** `algoritmi/eigenfaces.py`, linia 60-85

---

## 9. Ce e matricea C?

**Răspuns:** Matricea C este matricea de covarianță **neoptimizată**.

**Formula:** `C = A @ A.T`

**Dimensiuni:**
- A: (10304 × 320)
- A.T: (320 × 10304)
- **C: (10304 × 10304)** ← Foarte mare!

**Problema:** Calculul vectorilor proprii ai unei matrice 10304×10304 este foarte lent și consumă multă memorie.

**Cod:** `algoritmi/eigenfaces.py`, linia 39-59

---

## 10. De ce e optimizarea calculului L?

**Răspuns:** Optimizarea constă în calcularea vectorilor proprii ai unei matrice **mult mai mici**.

### Comparație:

| Matrice | Dimensiune | Număr elemente | Complexitate |
|---------|------------|----------------|--------------|
| **C** (neoptimizată) | 10304 × 10304 | 106,172,416 | O(n³) unde n=10304 |
| **L** (optimizată) | 320 × 320 | 102,400 | O(n³) unde n=320 |

**Avantaje:**
1. **Memorie:** L ocupă ~1000 de ori mai puțină memorie
2. **Viteză:** Calculul vectorilor proprii este mult mai rapid (n³ vs n³, dar n mult mai mic)
3. **Echivalență:** Rezultatul este același, doar că calculul este mai eficient

**Matematic:** Vectorii proprii ai C se obțin din vectorii proprii ai L prin: `v_C = A @ v_L`

**Cod:** `algoritmi/eigenfaces.py`, linia 60-85

---

## 📋 Rezumat - Ce ai în aplicație:

✅ **Reprezentant de clasă** - DA (`MATRICE_REPREZENTANTI`)  
✅ **Inițializarea matricei** - DA (`MATRICE_ANTRENARE`)  
✅ **SVD** - DA (implementat)  
✅ **PCA neoptimizată (C)** - DA (implementat)  
✅ **PCA optimizată (L)** - DA (implementat acum)  
✅ **Eigenfaces clasic** - DA (toate variantele)  
✅ **Eigenfaces cu reprezentanți** - DA (implementat)  

---

## 🔧 Cum să testezi toate variantele:

În cod, poți schimba metoda în `main.py`:

```python
# Pentru SVD (implicit)
PREPROCESARE_EIGENFACES_CLASIC(A, k, metoda='SVD')

# Pentru PCA neoptimizată (C)
PREPROCESARE_EIGENFACES_CLASIC(A, k, metoda='PCA')

# Pentru PCA optimizată (L)
PREPROCESARE_EIGENFACES_CLASIC(A, k, metoda='PCA_optimizata')
```


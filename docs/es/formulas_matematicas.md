# Métricas de Equidad: Fundamentos Matemáticos

## Notación

| Símbolo | Descripción |
|---------|-------------|
| $Y$ | Etiqueta verdadera (ground truth) |
| $\hat{Y}$ | Etiqueta predicha |
| $A$ | Atributo protegido (ej: sexo) |
| $a$ | Valor del grupo privilegiado |
| $\bar{a}$ | Valor del grupo no privilegiado |
| $P(\cdot)$ | Probabilidad |
| $TP, FP, TN, FN$ | Componentes de la matriz de confusión |

---

## 1. Diferencia de Paridad Estadística (SPD)

### Definición

La Diferencia de Paridad Estadística mide la diferencia en tasas de predicción positiva entre grupos no privilegiados y privilegiados.

$$\text{SPD} = P(\hat{Y} = 1 | A = \bar{a}) - P(\hat{Y} = 1 | A = a)$$

### Interpretación

| Valor | Interpretación |
|-------|----------------|
| SPD = 0 | Paridad perfecta |
| SPD < 0 | Grupo no privilegiado en desventaja |
| SPD > 0 | Grupo privilegiado en desventaja |
| \|SPD\| < 0.1 | Generalmente aceptable |

### Cálculo

$$\text{SPD} = \frac{\sum_{i: A_i = \bar{a}} \hat{Y}_i}{|\{i: A_i = \bar{a}\}|} - \frac{\sum_{i: A_i = a} \hat{Y}_i}{|\{i: A_i = a\}|}$$

### Ejemplo

```
Grupo        Total    Predicción Positiva    Tasa
─────────────────────────────────────────────────
Hombre (a)   138      103                   74.6%
Mujer (ā)     62       41                   66.1%

SPD = 0.661 - 0.746 = -0.085
```

---

## 2. Impacto Dispar (DI)

### Definición

El Impacto Dispar es la razón de tasas de predicción positiva entre grupos.

$$\text{DI} = \frac{P(\hat{Y} = 1 | A = \bar{a})}{P(\hat{Y} = 1 | A = a)}$$

### La Regla del 80%

La EEOC (Equal Employment Opportunity Commission) estableció que:

$$\text{DI} < 0.8 \implies \text{Evidencia de impacto adverso}$$

### Interpretación

| Valor | Interpretación |
|-------|----------------|
| DI = 1.0 | Paridad perfecta |
| DI ∈ [0.8, 1.25] | Generalmente aceptable |
| DI < 0.8 | Impacto adverso contra no privilegiados |
| DI > 1.25 | Impacto adverso contra privilegiados |

### Cálculo

$$\text{DI} = \frac{\frac{n_{\bar{a},+}}{n_{\bar{a}}}}{\frac{n_{a,+}}{n_a}}$$

Donde:
- $n_{\bar{a},+}$ = Predicciones positivas para grupo no privilegiado
- $n_{\bar{a}}$ = Total en grupo no privilegiado
- $n_{a,+}$ = Predicciones positivas para grupo privilegiado
- $n_a$ = Total en grupo privilegiado

### Ejemplo

```
DI = 0.661 / 0.746 = 0.886

Como 0.886 > 0.8, la regla del 80% se satisface.
```

---

## 3. Diferencia de Igualdad de Oportunidades (EOD)

### Definición

La EOD mide la diferencia en Tasas de Verdaderos Positivos (TPR) entre grupos, enfocándose solo en individuos calificados (Y=1).

$$\text{EOD} = P(\hat{Y} = 1 | Y = 1, A = \bar{a}) - P(\hat{Y} = 1 | Y = 1, A = a)$$

Equivalentemente:

$$\text{EOD} = \text{TPR}_{\bar{a}} - \text{TPR}_a$$

### Tasa de Verdaderos Positivos

$$\text{TPR}_g = \frac{TP_g}{TP_g + FN_g} = \frac{TP_g}{P_g}$$

Donde $P_g$ es el número de positivos reales en el grupo $g$.

### Interpretación

| Valor | Interpretación |
|-------|----------------|
| EOD = 0 | Igualdad de oportunidades |
| EOD < 0 | Individuos calificados no privilegiados menos propensos a ser aprobados |
| EOD > 0 | Individuos calificados no privilegiados más propensos a ser aprobados |

### Ejemplo

```
Grupo        Verdaderos Pos    Positivos Reales    TPR
────────────────────────────────────────────────────────
Hombre (a)        89               100            89.0%
Mujer (ā)         34                44            77.3%

EOD = 0.773 - 0.890 = -0.117
```

---

## 4. Diferencia de Odds Promedio (AOD)

### Definición

La AOD combina diferencias de TPR y FPR, midiendo la disparidad general en tasas de error.

$$\text{AOD} = \frac{1}{2}\left[(\text{FPR}_{\bar{a}} - \text{FPR}_a) + (\text{TPR}_{\bar{a}} - \text{TPR}_a)\right]$$

### Tasa de Falsos Positivos

$$\text{FPR}_g = \frac{FP_g}{FP_g + TN_g} = \frac{FP_g}{N_g}$$

Donde $N_g$ es el número de negativos reales en el grupo $g$.

### Interpretación

| Valor | Interpretación |
|-------|----------------|
| AOD = 0 | Igualdad de odds |
| AOD ≠ 0 | Diferencia sistemática en tasas de error |

### Relación con Igualdad de Odds

La restricción de Igualdad de Odds requiere:

$$\text{TPR}_{\bar{a}} = \text{TPR}_a \quad \text{Y} \quad \text{FPR}_{\bar{a}} = \text{FPR}_a$$

Lo que implica AOD = 0.

### Ejemplo

```
Grupo        TPR      FPR
─────────────────────────
Hombre (a)  89.0%    35.0%
Mujer (ā)   77.3%    27.8%

AOD = 0.5 × [(0.278 - 0.350) + (0.773 - 0.890)]
    = 0.5 × [-0.072 + (-0.117)]
    = 0.5 × (-0.189)
    = -0.0945
```

---

## 5. Algoritmo de Reponderación

### Objetivo

Transformar pesos muestrales para lograr paridad estadística en datos de entrenamiento.

### Fórmula de Pesos

Para cada muestra $(x_i, y_i)$ con atributo protegido $a_i$:

$$w_i = \frac{P(Y = y_i) \cdot P(A = a_i)}{P(Y = y_i, A = a_i)}$$

### Derivación

Queremos:

$$P_w(\hat{Y} = 1 | A = a) = P_w(\hat{Y} = 1 | A = \bar{a})$$

Usando el teorema de Bayes y el supuesto de independencia bajo reponderación:

$$w(a, y) = \frac{P(Y = y) \cdot P(A = a)}{P(Y = y, A = a)} = \frac{n \cdot n_y \cdot n_a}{n^2 \cdot n_{y,a}} = \frac{n_y \cdot n_a}{n \cdot n_{y,a}}$$

### Ejemplo de Cálculo de Pesos

```
Total de muestras: n = 800
Crédito bueno:     n_bueno = 560
Crédito malo:      n_malo = 240
Hombre:            n_hombre = 552
Mujer:             n_mujer = 248

Conteos observados:
- Hombre, Bueno:   n_h,b = 400
- Hombre, Malo:    n_h,m = 152
- Mujer, Bueno:    n_m,b = 160
- Mujer, Malo:     n_m,m = 88

Pesos:
w(hombre, bueno) = (560 × 552) / (800 × 400) = 0.966
w(hombre, malo)  = (240 × 552) / (800 × 152) = 1.089
w(mujer, bueno)  = (560 × 248) / (800 × 160) = 1.085
w(mujer, malo)   = (240 × 248) / (800 × 88)  = 0.845
```

---

## 6. Post-procesamiento de Igualdad de Odds

### Problema de Optimización

Dadas predicciones del clasificador base, encontrar predicciones ajustadas que satisfagan:

**Objetivo:**
$$\min \sum_i \mathcal{L}(\tilde{Y}_i, \hat{Y}_i)$$

**Sujeto a:**
$$P(\tilde{Y} = 1 | Y = y, A = a) = P(\tilde{Y} = 1 | Y = y, A = \bar{a}) \quad \forall y \in \{0, 1\}$$

### Formulación de Programación Lineal

Variables de decisión:
- $p_{a,0}$: Probabilidad de voltear 0→1 para grupo privilegiado
- $p_{a,1}$: Probabilidad de voltear 1→0 para grupo privilegiado
- $p_{\bar{a},0}$: Probabilidad de voltear 0→1 para grupo no privilegiado
- $p_{\bar{a},1}$: Probabilidad de voltear 1→0 para grupo no privilegiado

El TPR ajustado para el grupo $g$ se convierte en:

$$\widetilde{\text{TPR}}_g = \text{TPR}_g \cdot (1 - p_{g,1}) + (1 - \text{TPR}_g) \cdot p_{g,0}$$

Similarmente para FPR:

$$\widetilde{\text{FPR}}_g = \text{FPR}_g \cdot (1 - p_{g,1}) + (1 - \text{FPR}_g) \cdot p_{g,0}$$

### Restricciones

TPR igualado:
$$\widetilde{\text{TPR}}_a = \widetilde{\text{TPR}}_{\bar{a}}$$

FPR igualado:
$$\widetilde{\text{FPR}}_a = \widetilde{\text{FPR}}_{\bar{a}}$$

---

## 7. Métricas de Rendimiento

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

### Balanced Accuracy

$$\text{Balanced Accuracy} = \frac{1}{2}\left(\frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right) = \frac{\text{TPR} + \text{TNR}}{2}$$

### Área Bajo la Curva ROC (AUC)

$$\text{AUC} = P(\hat{Y}_{i^+} > \hat{Y}_{i^-})$$

Donde $i^+$ es una muestra positiva aleatoria e $i^-$ es una muestra negativa aleatoria.

### Puntaje F1

$$F_1 = 2 \cdot \frac{\text{Precisión} \cdot \text{Recall}}{\text{Precisión} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

---

## Tabla Resumen

| Métrica | Fórmula | Ideal | Rango |
|---------|---------|-------|-------|
| SPD | $P(\hat{Y}=1\|A=\bar{a}) - P(\hat{Y}=1\|A=a)$ | 0 | [-1, 1] |
| DI | $\frac{P(\hat{Y}=1\|A=\bar{a})}{P(\hat{Y}=1\|A=a)}$ | 1 | [0, ∞) |
| EOD | $\text{TPR}_{\bar{a}} - \text{TPR}_a$ | 0 | [-1, 1] |
| AOD | $\frac{1}{2}[(\text{FPR}_{\bar{a}}-\text{FPR}_a)+(\text{TPR}_{\bar{a}}-\text{TPR}_a)]$ | 0 | [-1, 1] |

---

## Referencias

1. Hardt, M., Price, E., & Srebro, N. (2016). *Equality of opportunity in supervised learning*. NeurIPS.
2. Kamiran, F., & Calders, T. (2012). *Data preprocessing techniques for classification without discrimination*. Knowledge and Information Systems.
3. Feldman, M., et al. (2015). *Certifying and removing disparate impact*. KDD.
4. Chouldechova, A. (2017). *Fair prediction with disparate impact: A study of bias in recidivism prediction instruments*. Big Data.

---

*Para implementación, ver Apéndice Técnico*

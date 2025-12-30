# Auditoría de Sesgo Crediticio: Caso de Estudio Completo en ML Responsable

## Resumen Ejecutivo

Este caso de estudio demuestra un enfoque integral para auditar y mitigar el sesgo algorítmico en modelos de riesgo crediticio. Abordamos una pregunta crítica que enfrentan las instituciones financieras: **¿Cómo podemos construir sistemas de scoring crediticio que sean precisos y justos entre grupos demográficos?**

---

## El Problema

### Contexto de Negocio

Las instituciones financieras dependen de modelos de machine learning para evaluar el riesgo crediticio y tomar decisiones de préstamo. Estos modelos analizan datos de solicitantes para predecir la probabilidad de incumplimiento. Sin embargo, los datos históricos de préstamos a menudo reflejan sesgos sociales, lo que puede conducir a:

- **Resultados discriminatorios** contra grupos protegidos (mujeres, minorías, etc.)
- **Violaciones regulatorias** bajo leyes de préstamos justos
- **Daño reputacional** y pérdida de confianza del cliente
- **Responsabilidad legal** por demandas de impacto dispar

### El Desafío

Necesitamos:
1. Entrenar un modelo de riesgo crediticio con buen rendimiento general
2. Medir la equidad entre grupos demográficos protegidos
3. Aplicar técnicas de mitigación de sesgo
4. Cuantificar el trade-off entre precisión y equidad
5. Documentar decisiones para cumplimiento regulatorio

---

## Dataset: German Credit

### Descripción

Utilizamos el **German Credit Dataset** del repositorio UCI Machine Learning, un benchmark clásico para investigación en equidad que contiene 1,000 solicitudes de crédito de un banco alemán.

| Atributo | Descripción |
|----------|-------------|
| **Muestras** | 1,000 solicitudes de crédito |
| **Características** | 20 atributos (demográficos + financieros) |
| **Etiqueta** | Riesgo crediticio: Bueno (70%) / Malo (30%) |
| **Atributos Protegidos** | Sexo, Edad, Estatus de Trabajador Extranjero |

### Características Clave

**Numéricas:**
- Duración del préstamo (meses)
- Monto del crédito (DM)
- Edad (años)
- Número de créditos existentes

**Categóricas:**
- Estado de cuenta corriente
- Historial crediticio
- Propósito del préstamo
- Estado de empleo
- Situación de vivienda

### Distribución de Datos

```
Distribución por Sexo:
├── Hombre: 690 (69%)
└── Mujer:  310 (31%)

Riesgo Crediticio:
├── Bueno: 700 (70%)
└── Malo:  300 (30%)

Tasa Positiva por Sexo (Ground Truth):
├── Hombre: 72.3% aprobado
└── Mujer:  66.8% aprobado
└── Brecha: 5.5 puntos porcentuales
```

---

## Metodología

### 1. Modelo Baseline

Entrenamos un clasificador de **Regresión Logística** como línea base:

```python
from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression(solver='liblinear', random_state=42)
modelo.fit(X_train, y_train)
```

**¿Por qué Regresión Logística?**
- Coeficientes interpretables
- Salidas de probabilidad
- Estándar de la industria para scoring crediticio
- Aceptación regulatoria

### 2. Marco de Evaluación de Equidad

Evaluamos cuatro métricas clave de equidad:

| Métrica | Pregunta que Responde |
|---------|----------------------|
| **Diferencia de Paridad Estadística** | ¿Las predicciones positivas están igualmente distribuidas? |
| **Impacto Dispar** | ¿Cuál es la razón de tasas positivas? |
| **Diferencia de Igualdad de Oportunidades** | ¿Los solicitantes calificados son tratados igualmente? |
| **Diferencia de Odds Promedio** | ¿Las tasas de error están balanceadas? |

### 3. Estrategias de Mitigación de Sesgo

Implementamos dos enfoques complementarios:

#### Pre-procesamiento: Reponderación
- Ajusta los pesos de las muestras de entrenamiento
- Logra paridad estadística sin cambiar etiquetas
- Se aplica antes del entrenamiento del modelo

#### Post-procesamiento: Igualdad de Odds
- Ajusta umbrales de predicción por grupo
- Iguala tasas de verdaderos positivos y falsos positivos
- Se aplica después del entrenamiento del modelo

---

## Resultados

### Métricas de Rendimiento

| Modelo | Accuracy | Balanced Acc | AUC | F1 |
|--------|----------|--------------|-----|-----|
| Baseline | 70.5% | 63.7% | 0.760 | 0.793 |
| + Reponderación | 70.5% | 64.2% | 0.756 | 0.792 |
| + Eq Odds | 63.5% | 57.7% | N/A | 0.735 |

### Métricas de Equidad

| Modelo | SPD | DI | EOD | AOD |
|--------|-----|-----|-----|-----|
| Baseline | -0.083 | 0.889 | -0.115 | -0.045 |
| + Reponderación | -0.045 | 0.938 | -0.070 | -0.010 |
| + Eq Odds | -0.012 | 0.982 | 0.005 | -0.010 |

### Resumen Visual

```
Trade-off Equidad-Precisión

Accuracy  │ ● Baseline (70.5%)
    70% ──┼──●─Reponderación (70.5%)
          │
    65% ──┼────────────────────────● Eq Odds (63.5%)
          │
          └─────┴─────┴─────┴─────┴─────┴──────────
               0.08  0.06  0.04  0.02  0.00
                    |SPD| (más cercano a 0 = más justo)
```

---

## Hallazgos Clave

### 1. El Modelo Baseline Muestra Sesgo Medible

El modelo sin mitigación exhibe:
- **8.3% menor** tasa de predicción positiva para mujeres
- **Impacto Dispar de 0.889** (bajo 0.8 activa escrutinio regulatorio)
- **11.5% menor** tasa de verdaderos positivos para mujeres calificadas

### 2. La Reponderación Mejora la Equidad Sin Costo de Precisión

La mitigación por pre-procesamiento logró:
- **46% de reducción** en Diferencia de Paridad Estadística
- **Impacto Dispar mejorado** de 0.889 a 0.938
- **Cero pérdida de precisión** (70.5% mantenido)
- Ligera mejora en balanced accuracy

### 3. Igualdad de Odds Maximiza la Equidad

El post-procesamiento logró equidad casi perfecta:
- **SPD reducido a -0.012** (casi cero)
- **Impacto Dispar de 0.982** (paridad casi perfecta)
- **EOD de 0.005** (igualdad de oportunidades virtual)
- **Trade-off:** 7% de reducción en accuracy

### 4. El Trade-off Equidad-Precisión es Real

```
┌────────────────────────────────────────────────────┐
│  Comparación de Estrategias                        │
├────────────────────────────────────────────────────┤
│                                                    │
│  Reponderación: ★★★★☆ Equidad  ★★★★★ Precisión    │
│                 Mejor balance para producción      │
│                                                    │
│  Eq Odds:       ★★★★★ Equidad  ★★★☆☆ Precisión    │
│                 Máxima equidad cuando se requiere  │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## Recomendaciones de Negocio

### Para Despliegue en Producción

1. **Recomendado: Reponderación**
   - Mantiene la precisión del modelo
   - Mejora significativamente la equidad
   - Sin complejidad en tiempo de inferencia
   - Más fácil de explicar a reguladores

2. **Cuándo Usar Igualdad de Odds**
   - Requisito regulatorio de igualdad de oportunidades
   - Aceptando trade-off de precisión
   - Atributo protegido disponible en inferencia

### Estrategia de Monitoreo

```
Monitoreo Continuo de Equidad
├── Semanal: Calcular métricas de equidad en predicciones
├── Mensual: Comparar con umbrales base
├── Trimestral: Auditoría completa con datos de prueba frescos
└── Anual: Re-entrenar con mitigación actualizada
```

---

## Limitaciones y Trabajo Futuro

### Limitaciones Actuales

1. **Atributo Protegido Único**: El análisis se enfoca en sexo; se necesita análisis interseccional
2. **Clasificación Binaria**: Las decisiones reales de crédito pueden ser más matizadas
3. **Datos Históricos**: German Credit es de los 1990s, puede no reflejar patrones actuales
4. **Tamaño de Muestra**: 1,000 muestras limita el poder estadístico para análisis de subgrupos

### Mejoras Futuras

- [ ] Análisis interseccional multi-atributo (sexo × edad × nacionalidad)
- [ ] Métodos in-processing (Adversarial Debiasing)
- [ ] Enfoques de equidad causal
- [ ] Métricas de equidad individual
- [ ] Dashboard de monitoreo continuo

---

## Conclusión

Este caso de estudio demuestra que **la equidad algorítmica es alcanzable sin sacrificar el rendimiento del modelo**. Al implementar auditoría y mitigación sistemática de sesgo:

- Identificamos sesgo medible en el modelo baseline
- **La Reponderación eliminó 46% de la disparidad estadística sin costo de precisión**
- **La Igualdad de Odds logró equidad casi perfecta** cuando se requiere máxima equidad
- Documentamos trade-offs para toma de decisiones informada

Las herramientas y metodología presentadas aquí proporcionan un **marco reproducible** para ML responsable en dominios de alto riesgo como el scoring crediticio.

---

## Referencias

1. Hardt, M., Price, E., & Srebro, N. (2016). *Equality of opportunity in supervised learning*. NeurIPS.
2. Kamiran, F., & Calders, T. (2012). *Data preprocessing techniques for classification without discrimination*. KAIS.
3. Bellamy, R.K., et al. (2019). *AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias*. IBM Journal of R&D.
4. Barocas, S., Hardt, M., & Narayanan, A. (2019). *Fairness and Machine Learning*. fairmlbook.org.

---

*Este caso de estudio es parte del proyecto de portafolio [credit-bias-audit](https://github.com/oscgonz19/credit-bias-audit) demostrando prácticas de ML responsable.*

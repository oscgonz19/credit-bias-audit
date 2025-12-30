# Cómo Funciona el Pipeline de Auditoría de Sesgo

## Visión General del Pipeline

El pipeline de auditoría procesa datos crediticios a través de cuatro etapas principales para evaluar y mitigar el sesgo en predicciones de riesgo crediticio.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE AUDITORÍA DE SESGO                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ETAPA 1          ETAPA 2          ETAPA 3          ETAPA 4               │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐           │
│  │ PREP    │ ───▶ │ENTRENA- │ ───▶ │  EVAL   │ ───▶ │REPORTES │           │
│  │ DATOS   │      │ MIENTO  │      │ EQUIDAD │      │         │           │
│  └─────────┘      └─────────┘      └─────────┘      └─────────┘           │
│       │                │                │                │                 │
│       ▼                ▼                ▼                ▼                 │
│   Cargar CSV       Baseline        Calcular         Generar              │
│   Preprocesar      Reponderado     Métricas         CSV + MD             │
│   Dividir          Eq Odds         Comparar         Reportes             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Etapa 1: Preparación de Datos

### 1.1 Carga de Datos

```python
# Descargar del Repositorio UCI
python scripts/download_data.py --dataset german

# Cargar en pandas
df = pd.read_csv('data/german_credit.csv')
```

**Entrada:** Archivo CSV crudo (1,000 solicitudes de crédito)
**Salida:** DataFrame de Pandas

### 1.2 Ingeniería de Características

```
Datos Crudos                          Características Procesadas
─────────────────────────────────────────────────────────────────
checking_status: "no checking"    →   checking_no_checking: 1
                                      checking_<0: 0
                                      checking_0<=X<200: 0

sexo: "male"                      →   sexo: 1

categoria_edad: "aged"            →   edad_cat: 1

riesgo-crediticio: "good"         →   riesgo-crediticio: 1
```

### 1.3 División Train/Test

```
┌─────────────────────────────────────────────────┐
│           Dataset Original (1000)               │
├─────────────────────────────────────────────────┤
│                                                 │
│   ┌─────────────────────┐  ┌────────────────┐  │
│   │  Conjunto Train     │  │ Conjunto Test  │  │
│   │      (800)          │  │    (200)       │  │
│   │                     │  │                │  │
│   │  Bueno: 560 (70%)   │  │ Bueno: 140(70%)│  │
│   │  Malo:  240 (30%)   │  │ Malo:   60(30%)│  │
│   └─────────────────────┘  └────────────────┘  │
│                                                 │
└─────────────────────────────────────────────────┘
                  División Estratificada
```

---

## Etapa 2: Entrenamiento del Modelo

### 2.1 Modelo Baseline

```python
# Regresión logística estándar
modelo = LogisticRegression(solver='liblinear', random_state=42)
modelo.fit(X_train, y_train)
```

**Sin mitigación aplicada** - Esto establece la línea base para comparación.

### 2.2 Reponderación (Mitigación Pre-procesamiento)

```
ALGORITMO DE REPONDERACIÓN
────────────────────────────────────────────────────

Paso 1: Calcular frecuencias esperadas
┌──────────────────────────────────────────────────┐
│  Para cada combinación (grupo, etiqueta):        │
│  Esperado = P(grupo) × P(etiqueta) × N           │
└──────────────────────────────────────────────────┘

Paso 2: Calcular pesos
┌──────────────────────────────────────────────────┐
│  peso = esperado / observado                     │
│                                                  │
│  Ejemplo:                                        │
│  Hombre + Bueno: peso = 490/483 = 1.015         │
│  Mujer + Bueno:  peso = 210/217 = 0.968         │
└──────────────────────────────────────────────────┘

Paso 3: Entrenar con pesos
┌──────────────────────────────────────────────────┐
│  modelo.fit(X_train, y_train,                    │
│             sample_weight=pesos)                 │
└──────────────────────────────────────────────────┘
```

**Representación visual del efecto de los pesos:**

```
Antes de Reponderación         Después de Reponderación
(Distribución original)        (Pesos ajustados)

Hombre ████████████████████    Hombre ████████████████████
       70% tasa bueno                 ~68% tasa efectiva

Mujer  █████████████            Mujer  █████████████████
       66% tasa bueno                 ~68% tasa efectiva
                                      ↑
                               Los pesos ajustan la influencia
```

### 2.3 Igualdad de Odds (Mitigación Post-procesamiento)

```
ALGORITMO DE IGUALDAD DE ODDS
────────────────────────────────────────────────────

Paso 1: Obtener predicciones baseline
┌──────────────────────────────────────────────────┐
│  y_pred_baseline = modelo.predict(X_test)        │
└──────────────────────────────────────────────────┘

Paso 2: Calcular tasas de error por grupo
┌──────────────────────────────────────────────────┐
│  TPR_hombre = TP_hombre / P_hombre               │
│  TPR_mujer = TP_mujer / P_mujer                  │
│  FPR_hombre = FP_hombre / N_hombre               │
│  FPR_mujer = FP_mujer / N_mujer                  │
└──────────────────────────────────────────────────┘

Paso 3: Resolver optimización (programa lineal)
┌──────────────────────────────────────────────────┐
│  Encontrar probabilidades de ajuste que:         │
│  - Igualen TPR entre grupos                      │
│  - Igualen FPR entre grupos                      │
│  - Minimicen pérdida de accuracy                 │
└──────────────────────────────────────────────────┘

Paso 4: Aplicar ajustes probabilísticos
┌──────────────────────────────────────────────────┐
│  Para cada predicción:                           │
│  - Con probabilidad p: voltear 0→1 o 1→0        │
│  - La probabilidad depende de la membresía al    │
│    grupo                                         │
└──────────────────────────────────────────────────┘
```

---

## Etapa 3: Evaluación de Equidad

### 3.1 Flujo de Cálculo de Métricas

```
                 Predicciones + Ground Truth
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌───────────┐       ┌───────────┐       ┌───────────┐
    │Rendimiento│       │  Equidad  │       │  Desglose │
    │           │       │           │       │ por Grupo │
    └───────────┘       └───────────┘       └───────────┘
          │                   │                   │
          ▼                   ▼                   ▼
    • Accuracy          • SPD               • TPR por grupo
    • Balanced Acc      • DI                • FPR por grupo
    • AUC               • EOD               • Accuracy por grupo
    • Precision         • AOD
    • Recall
    • F1
```

### 3.2 Cálculo de Métricas de Equidad

```python
# Para cada variante del modelo (baseline, reponderado, eq_odds):

valores_protegidos = X_test['sex'].values

metricas = {
    'rendimiento': calcular_metricas_rendimiento(y_test, y_pred, y_proba),
    'equidad': calcular_metricas_equidad(y_test, y_pred, valores_protegidos),
}
```

### 3.3 Tabla Comparativa de Métricas

```
┌─────────────────────────────────────────────────────────────────────┐
│                   COMPARACIÓN DE MÉTRICAS                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Modelo       Accuracy    SPD      DI      EOD      AOD            │
│  ─────────────────────────────────────────────────────────         │
│  Baseline     70.5%      -0.083   0.889   -0.115   -0.045          │
│  Reponderado  70.5%      -0.045   0.938   -0.070   -0.010          │
│  Eq Odds      63.5%      -0.012   0.982    0.005   -0.010          │
│                                                                     │
│  ─────────────────────────────────────────────────────────         │
│  Mejor para:  Repond.    Eq Odds  Eq Odds Eq Odds  Ambos           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Etapa 4: Generación de Reportes

### 4.1 Salida CSV

```csv
modelo,mitigacion,categoria,metrica,valor
logreg,none,rendimiento,accuracy,0.705
logreg,none,equidad,statistical_parity_difference,-0.083
logreg,reweighing,rendimiento,accuracy,0.705
logreg,reweighing,equidad,statistical_parity_difference,-0.045
logreg,eq_odds,rendimiento,accuracy,0.635
logreg,eq_odds,equidad,statistical_parity_difference,-0.012
```

### 4.2 Estructura del Reporte Markdown

```
reports/report.md
├── Encabezado y Timestamp
├── Configuración de Auditoría
├── Resumen
├── Tabla Comparativa de Resultados
├── Análisis Detallado
│   ├── Baseline
│   ├── Reponderación
│   └── Igualdad de Odds
├── Análisis de Trade-offs
├── Limitaciones
└── Recomendaciones
```

---

## Ejemplo de Flujo End-to-End

```python
# 1. PREPARACIÓN DE DATOS
datos = cargar_y_preparar_datos(
    dataset='german',
    ruta_datos='data/german_credit.csv',
    atributo_protegido='sex',
    random_state=42
)

# 2. MODELO BASELINE
modelo = entrenar_modelo_baseline(datos['X_train'], datos['y_train'])
y_pred, y_proba = obtener_predicciones(modelo, datos['X_test'])

# 3. EVALUACIÓN DE EQUIDAD
metricas_baseline = calcular_todas_metricas(
    datos['y_test'], y_pred, y_proba,
    datos['X_test']['sex'].values
)

# 4. MITIGACIÓN POR REPONDERACIÓN
pesos = aplicar_reponderacion(
    datos['X_train'], datos['y_train'], 'sex',
    datos['grupos_privilegiados'], datos['grupos_no_privilegiados']
)
modelo_rp = entrenar_modelo_baseline(
    datos['X_train'], datos['y_train'],
    sample_weight=pesos
)

# 5. MITIGACIÓN POR IGUALDAD DE ODDS
y_pred_eq = aplicar_eq_odds_postprocesamiento(
    datos['X_test'], datos['y_test'], y_pred, 'sex',
    datos['grupos_privilegiados'], datos['grupos_no_privilegiados']
)

# 6. GENERAR REPORTES
generar_csv_metricas(resultados, 'reports/metrics.csv')
generar_reporte_markdown(resultados, config, 'reports/report.md')
```

---

## Configuración del Pipeline

### Argumentos de Línea de Comandos

```bash
python scripts/run_audit.py \
    --dataset german \           # Dataset: german o adult
    --protected-attr sex \       # Atributo protegido
    --model logreg \             # Tipo de modelo
    --mitigation all \           # none, reweighing, eq_odds, all
    --seed 42 \                  # Semilla aleatoria
    --sample-size 1000 \         # Opcional: limitar tamaño de datos
    --out-dir reports/           # Directorio de salida
```

### Matriz de Configuración

| Parámetro | Opciones | Default | Efecto |
|-----------|----------|---------|--------|
| dataset | german, adult | german | Fuente de datos |
| protected-attr | sex, age, race | sex | Objetivo de análisis de equidad |
| model | logreg, logreg_cv, rf | logreg | Tipo de clasificador |
| mitigation | none, reweighing, eq_odds, all | all | Estrategias a aplicar |
| seed | int | 42 | Reproducibilidad |

---

## Extensibilidad del Pipeline

### Agregar Nuevos Datasets

```python
# En src/data.py
def cargar_nuevo_dataset(ruta_datos):
    df = pd.read_csv(ruta_datos)
    # Preprocesamiento específico del dataset
    return df

# Agregar a PROTECTED_ATTR_CONFIG
PROTECTED_ATTR_CONFIG['nuevo_dataset'] = {
    'atributo_protegido': {
        'privilegiado': 1,
        'no_privilegiado': 0,
        'map': {'grupo_a': 1, 'grupo_b': 0}
    }
}
```

### Agregar Nuevas Estrategias de Mitigación

```python
# En src/mitigation.py
class NuevaMitigacion:
    def __init__(self, grupos_privilegiados, grupos_no_privilegiados):
        # Inicializar
        pass

    def fit(self, dataset):
        # Aprender parámetros
        pass

    def transform(self, dataset):
        # Aplicar mitigación
        pass

# Agregar a función factory
def obtener_estrategia_mitigacion(nombre, ...):
    if nombre == 'nueva_estrategia':
        return NuevaMitigacion(...)
```

---

*Ver Apéndice Técnico para detalles de implementación.*

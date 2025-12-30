# Apéndice Técnico

## Arquitectura del Sistema

### Visión General de Módulos

```
src/
├── __init__.py          # Inicialización del paquete
├── data.py              # Carga y preprocesamiento de datos
├── models.py            # Wrappers de modelos ML
├── fairness_metrics.py  # Cálculo de métricas de equidad
├── mitigation.py        # Algoritmos de mitigación de sesgo
└── reporting.py         # Utilidades de generación de reportes
```

### Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           run_audit.py                                  │
│                      (Punto de Entrada CLI)                             │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│     data.py      │    │    models.py     │    │   reporting.py   │
├──────────────────┤    ├──────────────────┤    ├──────────────────┤
│ cargar_datos()   │    │ ModeloCredito    │    │ generar_csv()    │
│ preprocesar()    │    │ entrenar()       │    │ generar_md()     │
│ dividir_datos()  │    │ predecir()       │    │ formatear()      │
└──────────────────┘    └──────────────────┘    └──────────────────┘
          │                       │
          ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│fairness_metrics.py│    │  mitigation.py   │
├──────────────────┤    ├──────────────────┤
│ calcular_spd()   │    │ Reponderación    │
│ calcular_di()    │    │ EqOdds           │
│ calcular_eod()   │    │ aplicar_repond() │
│ calcular_aod()   │    │ aplicar_eq_odds()│
└──────────────────┘    └──────────────────┘
```

---

## Pipeline de Datos

### Flujo de Preprocesamiento

```
Datos Crudos                Datos Preprocesados
┌─────────────────┐         ┌─────────────────┐
│ sexo: "male"    │   ──▶   │ sexo: 1         │
│ edad: "aged"    │   ──▶   │ edad_cat: 1     │
│ riesgo: "good"  │   ──▶   │ riesgo: 1       │
│ propósito:"car" │   ──▶   │ proposito_car:1 │
└─────────────────┘         └─────────────────┘
```

### Ingeniería de Características

| Característica Original | Transformación | Salida |
|------------------------|----------------|--------|
| Categórica (sexo, edad) | Mapeo binario | 0/1 |
| Categórica (propósito) | One-hot encoding | Múltiples columnas |
| Numérica (monto) | Sin cambios | Valores originales |
| Etiqueta (riesgo) | Mapeo binario | 0=malo, 1=bueno |

### División Train/Test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # División 80/20
    random_state=42,      # Reproducibilidad
    stratify=y            # Mantener balance de clases
)
```

---

## Implementación del Modelo

### Clase CreditRiskModel

```python
class CreditRiskModel:
    """Wrapper para modelos de clasificación de riesgo crediticio."""

    MODELOS_SOPORTADOS = ["logreg", "logreg_cv", "rf"]

    def __init__(self, tipo_modelo: str, random_state: int = 42):
        self.tipo_modelo = tipo_modelo
        self.random_state = random_state
        self.modelo = self._crear_modelo()

    def fit(self, X, y, sample_weight=None):
        """Entrenar modelo con pesos opcionales."""
        if sample_weight is not None:
            self.modelo.fit(X, y, sample_weight=sample_weight)
        else:
            self.modelo.fit(X, y)
        return self

    def predict_proba(self, X):
        """Retornar predicciones de probabilidad."""
        return self.modelo.predict_proba(X)
```

### Configuraciones de Modelos

| Modelo | Parámetros | Caso de Uso |
|--------|------------|-------------|
| `logreg` | solver='liblinear', max_iter=1000 | Baseline por defecto |
| `logreg_cv` | cv=5, solver='liblinear' | Con validación cruzada |
| `rf` | n_estimators=100, n_jobs=-1 | Relaciones no lineales |

---

## Implementación de Métricas de Equidad

### Diferencia de Paridad Estadística

```python
def calcular_spd(y_pred, atributo_protegido, valor_privilegiado=1):
    """
    SPD = P(Ŷ=1|no_privilegiado) - P(Ŷ=1|privilegiado)

    Rango: [-1, 1]
    Ideal: 0
    """
    mask_priv = atributo_protegido == valor_privilegiado
    mask_no_priv = ~mask_priv

    p_priv = np.mean(y_pred[mask_priv])
    p_no_priv = np.mean(y_pred[mask_no_priv])

    return p_no_priv - p_priv
```

### Impacto Dispar

```python
def calcular_impacto_dispar(y_pred, atributo_protegido, valor_privilegiado=1):
    """
    DI = P(Ŷ=1|no_privilegiado) / P(Ŷ=1|privilegiado)

    Rango: [0, ∞)
    Ideal: 1.0
    Regla del 80%: DI < 0.8 indica impacto adverso
    """
    mask_priv = atributo_protegido == valor_privilegiado
    mask_no_priv = ~mask_priv

    p_priv = np.mean(y_pred[mask_priv])
    p_no_priv = np.mean(y_pred[mask_no_priv])

    return p_no_priv / p_priv if p_priv > 0 else np.inf
```

### Diferencia de Igualdad de Oportunidades

```python
def calcular_eod(y_true, y_pred, atributo_protegido, valor_privilegiado=1):
    """
    EOD = TPR_no_privilegiado - TPR_privilegiado

    Solo considera clase positiva (Y=1)
    Rango: [-1, 1]
    Ideal: 0
    """
    priv_pos = (y_true == 1) & (atributo_protegido == valor_privilegiado)
    no_priv_pos = (y_true == 1) & (atributo_protegido != valor_privilegiado)

    tpr_priv = np.mean(y_pred[priv_pos])
    tpr_no_priv = np.mean(y_pred[no_priv_pos])

    return tpr_no_priv - tpr_priv
```

### Diferencia de Odds Promedio

```python
def calcular_aod(y_true, y_pred, atributo_protegido, valor_privilegiado=1):
    """
    AOD = 0.5 * [(FPR_np - FPR_p) + (TPR_np - TPR_p)]

    Combina ambos tipos de error
    Rango: [-1, 1]
    Ideal: 0
    """
    # Calcular diferencia de TPR
    tpr_diff = calcular_eod(y_true, y_pred, atributo_protegido, valor_privilegiado)

    # Calcular diferencia de FPR
    priv_neg = (y_true == 0) & (atributo_protegido == valor_privilegiado)
    no_priv_neg = (y_true == 0) & (atributo_protegido != valor_privilegiado)

    fpr_priv = np.mean(y_pred[priv_neg])
    fpr_no_priv = np.mean(y_pred[no_priv_neg])
    fpr_diff = fpr_no_priv - fpr_priv

    return 0.5 * (fpr_diff + tpr_diff)
```

---

## Algoritmos de Mitigación

### Reponderación (Pre-procesamiento)

Asigna pesos a las muestras de entrenamiento para lograr paridad estadística.

**Fórmula de pesos:**
```
w(X,Y) = P(Y) × P(G) / P(Y,G)
```

Donde:
- G: Grupo protegido
- Y: Etiqueta

**Ejemplo de Cálculo de Pesos:**

```
Grupo     Etiqueta  Conteo  Esperado  Peso
────────────────────────────────────────────
Hombre    Bueno     483     490       1.015
Hombre    Malo      207     200       0.966
Mujer     Bueno     217     210       0.968
Mujer     Malo      93      100       1.075
```

### Igualdad de Oportunidades (Post-procesamiento)

Ajusta las predicciones para igualar TPR y FPR entre grupos.

**Resuelve programa lineal:**
```
minimizar: costo
sujeto a: TPR_np = TPR_p, FPR_np = FPR_p
```

```python
class EqOddsMitigation:
    """
    Ajusta predicciones para igualar TPR y FPR entre grupos.
    """

    def __init__(self, grupos_privilegiados, grupos_no_privilegiados, seed=42):
        self.postprocesador = EqOddsPostprocessing(
            privileged_groups=grupos_privilegiados,
            unprivileged_groups=grupos_no_privilegiados,
            seed=seed
        )

    def fit(self, dataset_verdadero, dataset_pred):
        """Aprender parámetros de ajuste."""
        self.postprocesador.fit(dataset_verdadero, dataset_pred)
        return self

    def predict(self, dataset_pred):
        """Aplicar ajuste de igualdad de odds."""
        return self.postprocesador.predict(dataset_pred)
```

---

## Integración con AIF360

### Conversión de Dataset

```python
def crear_dataset_aif360(
    X: pd.DataFrame,
    y: pd.Series,
    atributo_protegido: str,
    nombre_etiqueta: str = "label"
) -> BinaryLabelDataset:
    """Convertir pandas a formato AIF360."""
    df = X.copy()
    df[nombre_etiqueta] = y.values

    return BinaryLabelDataset(
        df=df,
        label_names=[nombre_etiqueta],
        protected_attribute_names=[atributo_protegido],
        favorable_label=1,
        unfavorable_label=0
    )
```

### Definiciones de Grupos

```python
# German Credit - Sexo
grupos_privilegiados = [{'sex': 1}]      # Hombre
grupos_no_privilegiados = [{'sex': 0}]   # Mujer

# German Credit - Edad
grupos_privilegiados = [{'age_cat': 1}]  # Mayor (>=25)
grupos_no_privilegiados = [{'age_cat': 0}] # Joven (<25)
```

---

## Estrategia de Testing

### Categorías de Tests

| Categoría | Archivos | Descripción |
|-----------|----------|-------------|
| Unitarios | `test_smoke.py` | Tests de funciones individuales |
| Integración | `test_smoke.py` | Tests de pipeline completo |
| Smoke | CI workflow | Tests de validación rápida |

### Fixtures de Test

```python
@pytest.fixture
def datos_german_muestra():
    """Crear datos mínimos de muestra para testing."""
    np.random.seed(42)
    n_muestras = 100

    return pd.DataFrame({
        "sex": np.random.choice(["male", "female"], n_muestras),
        "age_cat": np.random.choice(["aged", "young"], n_muestras),
        "credit-risk": np.random.choice(["good", "bad"], n_muestras),
        # ... otras características
    })
```

---

## Pipeline CI/CD

### Workflow de GitHub Actions

```yaml
name: CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"

      - name: Instalar dependencias
        run: pip install -r requirements-dev.txt

      - name: Descargar datos de muestra
        run: python scripts/download_data.py --create-sample

      - name: Ejecutar tests
        run: pytest tests/ -v

      - name: Ejecutar test de integración
        run: python scripts/run_audit.py --sample-size 500
```

---

## Manejo de Errores

### Problemas Comunes y Soluciones

| Error | Causa | Solución |
|-------|-------|----------|
| `ValueError: NaN in linprog` | Grupo vacío en datos de test | Asegurar representación de todos los grupos |
| `KeyError: protected_attr` | Columna faltante | Verificar preprocesamiento |
| `ZeroDivisionError` | Sin predicciones positivas | Verificar umbral |

### Programación Defensiva

```python
def calcular_impacto_dispar(...):
    if p_priv == 0:
        return np.inf if p_no_priv > 0 else 1.0
    return p_no_priv / p_priv
```

---

## Consideraciones de Rendimiento

### Uso de Memoria

| Tamaño Dataset | Memoria (aprox) |
|----------------|-----------------|
| 1,000 filas | ~50 MB |
| 10,000 filas | ~200 MB |
| 100,000 filas | ~1.5 GB |

### Tiempo de Ejecución

| Operación | Tiempo (1000 filas) |
|-----------|---------------------|
| Carga de datos | <1s |
| Entrenamiento | ~2s |
| Cálculo de métricas | <1s |
| Generación de reporte | <1s |
| **Total** | **~5s** |

---

*Ver Fórmulas Matemáticas para derivaciones detalladas.*

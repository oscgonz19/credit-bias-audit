# Credit Bias Audit | Auditoría de Sesgo Crediticio

[![CI](https://github.com/oscgonz19/credit-bias-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/oscgonz19/credit-bias-audit/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **EN** | [ES](#español)

---

## Documentation | Documentación

### Complete Documentation Suite

| Document | Description | Audience | EN | ES |
|----------|-------------|----------|----|----|
| **Case Study** | Complete portfolio case study | General | [Link](docs/en/case_study.md) | [Link](docs/es/caso_estudio.md) |
| **Executive Summary** | High-level overview with architecture | Recruiters / Managers | [Link](docs/en/executive_summary.md) | [Link](docs/es/resumen_ejecutivo.md) |
| **Technical Appendix** | Detailed technical documentation | Tech Leads / Engineers | [Link](docs/en/technical_appendix.md) | [Link](docs/es/apendice_tecnico.md) |
| **Pipeline Explained** | How the prediction pipeline works | Data Scientists | [Link](docs/en/pipeline_explained.md) | [Link](docs/es/pipeline_explicado.md) |
| **Mathematical Formulas** | Fairness metrics derivations | Statisticians / Quants | [Link](docs/en/mathematical_formulas.md) | [Link](docs/es/formulas_matematicas.md) |
| **Model Card** | Model documentation and intended use | All | [Link](docs/en/model_card.md) | [Link](docs/es/ficha_modelo.md) |
| **Data Sheet** | Dataset documentation and biases | All | [Link](docs/en/data_sheet.md) | [Link](docs/es/hoja_datos.md) |

### Quick Navigation | Navegación Rápida

```
📚 Start Here Based on Your Role / Comienza Según Tu Rol:

👔 Recruiters/Managers   → docs/en/executive_summary.md | docs/es/resumen_ejecutivo.md
👨‍💻 Engineers            → docs/en/technical_appendix.md | docs/es/apendice_tecnico.md
📊 Data Scientists       → docs/en/pipeline_explained.md | docs/es/pipeline_explicado.md
🔬 Researchers           → docs/en/mathematical_formulas.md | docs/es/formulas_matematicas.md
📋 Everyone              → docs/en/case_study.md | docs/es/caso_estudio.md
```

---

## English

A reproducible pipeline for auditing fairness in credit risk models. This project demonstrates responsible ML practices by:

- Training baseline credit risk classification models
- Computing fairness metrics (Statistical Parity, Disparate Impact, Equal Opportunity)
- Applying bias mitigation techniques (Reweighing, Equalized Odds)
- Generating reproducible reports showing fairness-performance trade-offs

### Key Results

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AUDIT RESULTS SUMMARY                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Mitigation      Accuracy    SPD       DI      EOD      Status     │
│  ─────────────────────────────────────────────────────────────────  │
│  Baseline        70.5%      -0.083    0.889   -0.115   ⚠ Bias      │
│  Reweighing      70.5%      -0.045    0.938   -0.070   ✓ Improved  │
│  Eq Odds         63.5%      -0.012    0.982    0.005   ✓ Fair      │
│                                                                     │
│  Key Insight: 46% fairness improvement with ZERO accuracy loss     │
│               using Reweighing pre-processing                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Quick Start

```bash
# 1. Clone and setup environment
git clone https://github.com/oscgonz19/credit-bias-audit.git
cd credit-bias-audit
make setup
conda activate credit-bias-audit

# 2. Download data
make data

# 3. Run full audit
make run
```

### Project Structure

```
credit-bias-audit/
├── src/
│   ├── data.py              # Data loading and preprocessing
│   ├── models.py            # Model wrappers (LogReg, RF)
│   ├── fairness_metrics.py  # SPD, DI, EOD, AOD metrics
│   ├── mitigation.py        # Reweighing, EqOdds post-processing
│   └── reporting.py         # CSV and Markdown report generation
├── scripts/
│   ├── download_data.py     # Download German Credit dataset
│   └── run_audit.py         # CLI for end-to-end audit
├── docs/                    # Complete documentation (EN/ES)
├── tests/
│   └── test_smoke.py        # Unit and integration tests
├── reports/                 # Generated reports (gitignored)
├── environment.yml          # Conda environment
├── Makefile                 # Common commands
└── .github/workflows/ci.yml # CI pipeline
```

### Usage

#### Command Line Interface

```bash
# Run audit with defaults (German Credit, sex attribute, all mitigations)
python scripts/run_audit.py

# Specify options
python scripts/run_audit.py \
    --dataset german \
    --protected-attr sex \
    --model logreg \
    --mitigation all \
    --seed 42 \
    --out-dir reports/

# Audit age attribute
python scripts/run_audit.py --protected-attr age --out-dir reports/age_audit
```

#### CLI Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--dataset` | german, adult | german | Dataset to audit |
| `--protected-attr` | sex, age, race, foreign_worker | sex | Protected attribute |
| `--model` | logreg, logreg_cv, rf | logreg | Model type |
| `--mitigation` | none, reweighing, eq_odds, all | all | Mitigation strategy |
| `--seed` | int | 42 | Random seed |
| `--out-dir` | path | reports/ | Output directory |
| `--sample-size` | int | None | Limit data size (for testing) |

#### Makefile Targets

```bash
make setup      # Create conda environment
make data       # Download German Credit dataset
make run        # Run audit with defaults
make test       # Run pytest smoke tests
make test-ci    # Run full CI test suite
make lint       # Check code formatting
make format     # Auto-format code
make clean      # Remove generated files
```

### Outputs

#### metrics.csv

CSV file with all computed metrics:

```csv
model,mitigation,category,metric,value
logreg,none,performance,accuracy,0.705
logreg,none,fairness,statistical_parity_difference,-0.083
logreg,reweighing,performance,accuracy,0.705
logreg,reweighing,fairness,statistical_parity_difference,-0.045
```

#### report.md

Markdown report containing:
- Audit configuration
- Results comparison table
- Detailed analysis per mitigation
- Trade-off analysis
- Limitations and recommendations

### Fairness Metrics

| Metric | Formula | Ideal | Interpretation |
|--------|---------|-------|----------------|
| **Statistical Parity Difference (SPD)** | P(Ŷ=1\|unpriv) - P(Ŷ=1\|priv) | 0 | Difference in positive prediction rates |
| **Disparate Impact (DI)** | P(Ŷ=1\|unpriv) / P(Ŷ=1\|priv) | 1.0 | Ratio of positive rates (< 0.8 = adverse impact) |
| **Equal Opportunity Difference (EOD)** | TPR_unpriv - TPR_priv | 0 | Difference in true positive rates |
| **Average Odds Difference (AOD)** | 0.5 × (ΔFPR + ΔTPR) | 0 | Average difference in error rates |

### Mitigation Strategies

#### 1. Reweighing (Pre-processing) - Recommended

Assigns weights to training samples to achieve statistical parity without changing labels.

```
✓ No accuracy loss
✓ Simple to implement
✓ No inference-time complexity
✓ Easy to explain to regulators
```

#### 2. Equalized Odds (Post-processing)

Adjusts predictions to equalize TPR and FPR across groups.

```
✓ Maximum fairness improvement
✗ Requires protected attribute at inference
✗ May reduce accuracy significantly
```

### Testing

```bash
# Run all tests
make test

# Run specific test
pytest tests/test_smoke.py::TestFairnessMetrics -v

# Quick integration test
make test-quick
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CREDIT BIAS AUDIT SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│   │   DATA       │    │   MODEL      │    │   FAIRNESS   │             │
│   │   LAYER      │───▶│   LAYER      │───▶│   LAYER      │             │
│   └──────────────┘    └──────────────┘    └──────────────┘             │
│         │                   │                    │                      │
│         ▼                   ▼                    ▼                      │
│   • Load Data         • Train Model        • Compute Metrics           │
│   • Preprocess        • Predict            • Compare Groups            │
│   • Split             • Evaluate           • Assess Fairness           │
│                                                  │                      │
│                           ┌──────────────────────┴───────┐              │
│                           │      MITIGATION LAYER        │              │
│                           ├──────────────────────────────┤              │
│                           │  Reweighing    Equalized     │              │
│                           │  (Pre-proc)    Odds (Post)   │              │
│                           └──────────────────────────────┘              │
│                                          │                              │
│                           ┌──────────────┴───────────────┐              │
│                           │      REPORTING LAYER         │              │
│                           │  CSV + Markdown + Analysis   │              │
│                           └──────────────────────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Requirements

- Python 3.10+
- conda (recommended) or pip

### Key Dependencies

- numpy, pandas, scikit-learn
- aif360 (AI Fairness 360)
- aequitas
- pytest (for testing)

---

## Limitations

1. **Single Protected Attribute:** Analysis is univariate; intersectional effects not captured
2. **Binary Classification:** Designed for binary credit risk (good/bad)
3. **Historical Data:** German Credit dataset reflects 1990s lending practices
4. **Post-processing Constraints:** Equalized Odds requires protected attributes at inference

---

## References

- Bellamy, R. K., et al. (2019). AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning.
- Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination.

---

## License

MIT License

---

---

# Español

Un pipeline reproducible para auditar equidad en modelos de riesgo crediticio. Este proyecto demuestra prácticas de ML responsable mediante:

- Entrenamiento de modelos baseline de clasificación de riesgo crediticio
- Cálculo de métricas de equidad (Paridad Estadística, Impacto Dispar, Igualdad de Oportunidades)
- Aplicación de técnicas de mitigación de sesgo (Reponderación, Igualdad de Odds)
- Generación de reportes reproducibles mostrando trade-offs equidad-rendimiento

### Resultados Clave

```
┌─────────────────────────────────────────────────────────────────────┐
│                   RESUMEN DE RESULTADOS DE AUDITORÍA                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Mitigación      Accuracy    SPD       DI      EOD      Estado     │
│  ─────────────────────────────────────────────────────────────────  │
│  Baseline        70.5%      -0.083    0.889   -0.115   ⚠ Sesgo     │
│  Reponderación   70.5%      -0.045    0.938   -0.070   ✓ Mejorado  │
│  Eq Odds         63.5%      -0.012    0.982    0.005   ✓ Justo     │
│                                                                     │
│  Hallazgo: 46% mejora en equidad con CERO pérdida de accuracy      │
│            usando Reponderación (pre-procesamiento)                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Inicio Rápido

```bash
# 1. Clonar y configurar entorno
git clone https://github.com/oscgonz19/credit-bias-audit.git
cd credit-bias-audit
make setup
conda activate credit-bias-audit

# 2. Descargar datos
make data

# 3. Ejecutar auditoría completa
make run
```

### Estructura del Proyecto

```
credit-bias-audit/
├── src/
│   ├── data.py              # Carga y preprocesamiento de datos
│   ├── models.py            # Wrappers de modelos (LogReg, RF)
│   ├── fairness_metrics.py  # Métricas SPD, DI, EOD, AOD
│   ├── mitigation.py        # Reponderación, EqOdds post-procesamiento
│   └── reporting.py         # Generación de reportes CSV y Markdown
├── scripts/
│   ├── download_data.py     # Descarga dataset German Credit
│   └── run_audit.py         # CLI para auditoría end-to-end
├── docs/                    # Documentación completa (EN/ES)
├── tests/
│   └── test_smoke.py        # Tests unitarios y de integración
├── reports/                 # Reportes generados (gitignored)
├── environment.yml          # Entorno Conda
├── Makefile                 # Comandos comunes
└── .github/workflows/ci.yml # Pipeline CI
```

### Uso

#### Interfaz de Línea de Comandos

```bash
# Ejecutar auditoría con valores por defecto
python scripts/run_audit.py

# Especificar opciones
python scripts/run_audit.py \
    --dataset german \
    --protected-attr sex \
    --model logreg \
    --mitigation all \
    --seed 42 \
    --out-dir reports/

# Auditar atributo edad
python scripts/run_audit.py --protected-attr age --out-dir reports/age_audit
```

#### Argumentos CLI

| Argumento | Opciones | Default | Descripción |
|-----------|----------|---------|-------------|
| `--dataset` | german, adult | german | Dataset a auditar |
| `--protected-attr` | sex, age, race, foreign_worker | sex | Atributo protegido |
| `--model` | logreg, logreg_cv, rf | logreg | Tipo de modelo |
| `--mitigation` | none, reweighing, eq_odds, all | all | Estrategia de mitigación |
| `--seed` | int | 42 | Semilla aleatoria |
| `--out-dir` | path | reports/ | Directorio de salida |
| `--sample-size` | int | None | Limitar tamaño de datos (para testing) |

#### Comandos Makefile

```bash
make setup      # Crear entorno conda
make data       # Descargar dataset German Credit
make run        # Ejecutar auditoría con defaults
make test       # Ejecutar tests pytest
make test-ci    # Ejecutar suite completa de CI
make lint       # Verificar formato de código
make format     # Auto-formatear código
make clean      # Eliminar archivos generados
```

### Métricas de Equidad

| Métrica | Fórmula | Ideal | Interpretación |
|---------|---------|-------|----------------|
| **Diferencia de Paridad Estadística (SPD)** | P(Ŷ=1\|no_priv) - P(Ŷ=1\|priv) | 0 | Diferencia en tasas de predicción positiva |
| **Impacto Dispar (DI)** | P(Ŷ=1\|no_priv) / P(Ŷ=1\|priv) | 1.0 | Razón de tasas (< 0.8 = impacto adverso) |
| **Diferencia Igualdad de Oportunidades (EOD)** | TPR_no_priv - TPR_priv | 0 | Diferencia en tasas de verdaderos positivos |
| **Diferencia de Odds Promedio (AOD)** | 0.5 × (ΔFPR + ΔTPR) | 0 | Diferencia promedio en tasas de error |

### Estrategias de Mitigación

#### 1. Reponderación (Pre-procesamiento) - Recomendada

Asigna pesos a muestras de entrenamiento para lograr paridad estadística sin cambiar etiquetas.

```
✓ Sin pérdida de accuracy
✓ Simple de implementar
✓ Sin complejidad en inferencia
✓ Fácil de explicar a reguladores
```

#### 2. Igualdad de Odds (Post-procesamiento)

Ajusta predicciones para igualar TPR y FPR entre grupos.

```
✓ Máxima mejora en equidad
✗ Requiere atributo protegido en inferencia
✗ Puede reducir accuracy significativamente
```

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────┐
│               SISTEMA DE AUDITORÍA DE SESGO CREDITICIO                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│   │    CAPA      │    │    CAPA      │    │    CAPA      │             │
│   │   DATOS      │───▶│   MODELO     │───▶│   EQUIDAD    │             │
│   └──────────────┘    └──────────────┘    └──────────────┘             │
│         │                   │                    │                      │
│         ▼                   ▼                    ▼                      │
│   • Cargar Datos      • Entrenar           • Calcular Métricas         │
│   • Preprocesar       • Predecir           • Comparar Grupos           │
│   • Dividir           • Evaluar            • Evaluar Equidad           │
│                                                  │                      │
│                           ┌──────────────────────┴───────┐              │
│                           │      CAPA DE MITIGACIÓN      │              │
│                           ├──────────────────────────────┤              │
│                           │  Reponderación   Igualdad    │              │
│                           │  (Pre-proc)      Odds (Post) │              │
│                           └──────────────────────────────┘              │
│                                          │                              │
│                           ┌──────────────┴───────────────┐              │
│                           │      CAPA DE REPORTES        │              │
│                           │  CSV + Markdown + Análisis   │              │
│                           └──────────────────────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Requisitos

- Python 3.10+
- conda (recomendado) o pip

### Dependencias Principales

- numpy, pandas, scikit-learn
- aif360 (AI Fairness 360)
- aequitas
- pytest (para testing)

---

## Limitaciones

1. **Atributo Protegido Único:** El análisis es univariado; efectos interseccionales no capturados
2. **Clasificación Binaria:** Diseñado para riesgo crediticio binario (bueno/malo)
3. **Datos Históricos:** German Credit refleja prácticas de préstamos de los 1990s
4. **Restricciones Post-procesamiento:** Igualdad de Odds requiere atributos protegidos en inferencia

---

## Referencias

- Bellamy, R. K., et al. (2019). AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning.
- Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination.

---

## Licencia

MIT License

---

## Contributing / Contribuir

1. Fork the repository / Hacer fork del repositorio
2. Create a feature branch / Crear rama de feature
3. Run tests / Ejecutar tests: `make test`
4. Format code / Formatear código: `make format`
5. Submit a pull request / Enviar pull request

# Resumen Ejecutivo: Sistema de Auditoría de Sesgo Crediticio

## Visión General del Proyecto

**Credit Bias Audit** es una solución end-to-end para detectar y mitigar el sesgo algorítmico en modelos de riesgo crediticio. Este sistema permite a las instituciones financieras construir algoritmos de préstamo justos, conformes y precisos.

---

## Valor de Negocio

| Desafío | Nuestra Solución | Impacto |
|---------|------------------|---------|
| Riesgo Regulatorio | Auditoría automática de equidad | Cumplimiento proactivo |
| Demandas por Discriminación | Mitigación de sesgo documentada | Protección legal |
| Riesgo Reputacional | Decisiones de IA transparentes | Confianza del cliente |
| Rendimiento del Modelo | Trade-offs optimizados | Precisión mantenida |

---

## Arquitectura del Sistema

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
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│   │ • Cargar     │    │ • Entrenar   │    │ • Calcular   │             │
│   │ • Preprocesar│    │ • Predecir   │    │   Métricas   │             │
│   │ • Dividir    │    │ • Evaluar    │    │ • Comparar   │             │
│   └──────────────┘    └──────────────┘    └──────────────┘             │
│                                                  │                      │
│                                                  ▼                      │
│                           ┌──────────────────────────────┐              │
│                           │      CAPA DE MITIGACIÓN      │              │
│                           ├──────────────────────────────┤              │
│                           │  ┌─────────┐   ┌──────────┐  │              │
│                           │  │Reponde- │   │Igualdad  │  │              │
│                           │  │ración   │   │de Odds   │  │              │
│                           │  │ (Pre)   │   │ (Post)   │  │              │
│                           │  └─────────┘   └──────────┘  │              │
│                           └──────────────────────────────┘              │
│                                          │                              │
│                                          ▼                              │
│                           ┌──────────────────────────────┐              │
│                           │      CAPA DE REPORTES        │              │
│                           ├──────────────────────────────┤              │
│                           │  • CSV de Métricas           │              │
│                           │  • Reporte Markdown          │              │
│                           │  • Análisis de Trade-offs    │              │
│                           └──────────────────────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Resultados Clave

### Mejora en Equidad

```
Antes de Mitigación          Después de Mitigación (Reponderación)
┌──────────────────┐         ┌──────────────────┐
│  SPD: -8.3%      │  ───▶   │  SPD: -4.5%      │  46% mejora
│  DI:   0.889     │         │  DI:   0.938     │  5.5% mejora
│  EOD: -11.5%     │         │  EOD: -7.0%      │  39% mejora
└──────────────────┘         └──────────────────┘
```

### Preservación de Precisión

| Métrica | Baseline | Con Reponderación | Cambio |
|---------|----------|-------------------|--------|
| Accuracy | 70.5% | 70.5% | **0%** |
| AUC | 0.760 | 0.756 | -0.5% |
| F1 Score | 0.793 | 0.792 | -0.1% |

**Hallazgo Clave:** Ganancias significativas en equidad logradas con prácticamente ninguna pérdida de precisión.

---

## Stack Tecnológico

```
┌─────────────────────────────────────────────────┐
│              STACK TECNOLÓGICO                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  Lenguaje       Python 3.10                     │
│  Framework ML   scikit-learn                    │
│  Equidad        AIF360, Aequitas                │
│  Testing        pytest                          │
│  CI/CD          GitHub Actions                  │
│  Entorno        Conda                           │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## Flujo de Trabajo

```
    ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
    │  DATOS  │────▶│ENTRENAR │────▶│ AUDITAR │────▶│REPORTAR │
    └─────────┘     └─────────┘     └─────────┘     └─────────┘
         │               │               │               │
         ▼               ▼               ▼               ▼
    Descargar &     Baseline &      Métricas       CSV + MD
    Preprocesar     Mitigado        Equidad        Reportes
```

### Interfaz de Línea de Comandos

```bash
# Auditoría completa en un comando
python scripts/run_audit.py \
    --dataset german \
    --protected-attr sex \
    --mitigation all \
    --out-dir reports/
```

---

## Alineación con Regulaciones

| Regulación | Requisito | Cómo lo Abordamos |
|------------|-----------|-------------------|
| **ECOA** | Sin discriminación en crédito | Métricas de equidad + mitigación |
| **Fair Housing Act** | Trato igualitario | Análisis de impacto dispar |
| **GDPR Art. 22** | Explicabilidad | Modelos interpretables + reportes |
| **SR 11-7** | Gestión de riesgo de modelo | Rastro completo de auditoría |

---

## Opciones de Despliegue

### Opción A: Reponderación (Recomendada)
- **Mejor para:** Sistemas de producción
- **Trade-off:** Balance equidad-precisión
- **Complejidad:** Solo tiempo de entrenamiento
- **Impacto en precisión:** Ninguno

### Opción B: Igualdad de Odds
- **Mejor para:** Requisitos de máxima equidad
- **Trade-off:** Prioriza equidad
- **Complejidad:** Ajuste en tiempo de inferencia
- **Impacto en precisión:** -7%

---

## Métricas del Proyecto

| Métrica | Valor |
|---------|-------|
| Cobertura de Tests | Suite de tests completa |
| Documentación | Model Card + Data Sheet |
| Reproducibilidad | Seeds fijos + pipeline CI |
| Modularidad | 5 módulos independientes |

---

## Inicio Rápido

```bash
# Setup (una vez)
conda env create -f environment.yml
conda activate credit-bias-audit

# Ejecutar auditoría
make data    # Descargar dataset
make run     # Ejecutar auditoría completa
make test    # Verificar funcionamiento
```

---

## Outputs Generados

```
reports/
├── metrics.csv      # Todas las métricas en formato estructurado
└── report.md        # Reporte de auditoría legible
    ├── Configuración
    ├── Tabla Comparativa de Resultados
    ├── Análisis Detallado por Estrategia
    ├── Análisis de Trade-offs
    ├── Limitaciones
    └── Recomendaciones
```

---

## Equipo y Contacto

**Tipo de Proyecto:** Portafolio / Open Source
**Dominio:** IA Responsable / Equidad en ML
**Repositorio:** [credit-bias-audit](https://github.com/oscgonz19/credit-bias-audit)

---

## Próximos Pasos

1. **Evaluar** el sistema con tus propios datos
2. **Personalizar** umbrales de equidad para tu contexto
3. **Integrar** en pipelines ML existentes
4. **Monitorear** métricas de equidad en producción

---

*Este resumen ejecutivo proporciona una visión de alto nivel. Ver Apéndice Técnico para detalles de implementación.*

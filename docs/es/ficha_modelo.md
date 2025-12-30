# Ficha del Modelo: Clasificación de Riesgo Crediticio

## Detalles del Modelo

- **Tipo de Modelo:** Regresión Logística (baseline)
- **Framework:** scikit-learn
- **Versión:** 1.0
- **Fecha:** 2024
- **Licencia:** MIT

## Uso Previsto

### Usos Primarios Previstos
- Demostración educativa de auditoría de equidad en modelos de riesgo crediticio
- Investigación sobre técnicas de detección y mitigación de sesgo
- Demostración de portafolio de prácticas de ML responsable

### Usuarios Primarios Previstos
- Científicos de datos aprendiendo sobre equidad en ML
- Investigadores estudiando sesgo algorítmico
- Estudiantes en cursos de IA responsable

### Usos Fuera del Alcance
- **NO destinado para decisiones de crédito en producción**
- No debe usarse para aprobación/denegación real de préstamos
- No validado para cumplimiento regulatorio

## Datos de Entrenamiento

### German Credit Dataset
- **Fuente:** Repositorio UCI Machine Learning
- **Tamaño:** 1,000 instancias
- **Características:** 20 atributos (demográficos, financieros)
- **Etiqueta:** Riesgo crediticio (bueno/malo)
- **Atributos Protegidos:** sexo, edad, estatus de trabajador extranjero

Ver `hoja_datos.md` para documentación detallada de los datos.

## Datos de Evaluación

- 20% de holdout de los datos de entrenamiento
- División estratificada para mantener balance de clases
- Mismo preprocesamiento que datos de entrenamiento

## Métricas

### Métricas de Rendimiento
| Métrica | Descripción |
|---------|-------------|
| Accuracy | Corrección general de predicciones |
| Balanced Accuracy | Recall promedio entre clases |
| AUC | Área bajo la curva ROC |
| F1 Score | Media armónica de precisión y recall |

### Métricas de Equidad
| Métrica | Descripción | Valor Ideal |
|---------|-------------|-------------|
| Diferencia de Paridad Estadística | Diferencia en tasas positivas | 0 |
| Impacto Dispar | Razón de tasas positivas | 1.0 (aceptable: 0.8-1.25) |
| Diferencia de Igualdad de Oportunidades | Diferencia en TPR | 0 |
| Diferencia de Odds Promedio | Promedio de diferencias de TPR y FPR | 0 |

## Consideraciones Éticas

### Limitaciones Conocidas
1. **Sesgo Histórico:** Los datos de entrenamiento reflejan decisiones de préstamo históricas que pueden incorporar sesgos sociales
2. **Discriminación por Proxy:** Las características pueden correlacionarse con atributos protegidos
3. **Atributo Protegido Único:** El análisis se enfoca en un atributo a la vez; efectos interseccionales no capturados
4. **Tamaño de Muestra:** Algunos subgrupos tienen representación limitada

### Riesgos y Daños
- Podría perpetuar discriminación histórica si se despliega
- Puede tener diferentes tasas de error entre grupos demográficos
- Las mitigaciones de post-procesamiento pueden no generalizar a nuevas poblaciones

### Estrategias de Mitigación Implementadas
1. **Reponderación (Pre-procesamiento):** Ajusta pesos muestrales para lograr paridad estadística en entrenamiento
2. **Igualdad de Odds (Post-procesamiento):** Ajusta predicciones para igualar TPR y FPR entre grupos

## Advertencias y Recomendaciones

### Recomendaciones de Uso
1. Este modelo es solo para propósitos educativos/demostración
2. Siempre auditar equidad antes de cualquier consideración de despliegue
3. Considerar múltiples definiciones de equidad según el contexto
4. Involucrar a stakeholders en definir trade-offs aceptables
5. Monitorear cambios distribucionales en despliegue

### Advertencias Técnicas
- Los resultados dependen de la definición del atributo protegido
- Diferentes semillas aleatorias pueden producir diferentes métricas de equidad
- El post-procesamiento requiere acceso a atributos protegidos en inferencia

## Información Adicional

### Contacto
- Repositorio: [credit-bias-audit](https://github.com/oscgonz19/credit-bias-audit)

### Referencias
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning.
- Kamiran, F., & Calders, T. (2012). Data preprocessing techniques for classification without discrimination.
- Bellamy, R. K., et al. (2019). AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias.

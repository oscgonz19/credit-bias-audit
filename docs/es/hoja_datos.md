# Hoja de Datos: German Credit Dataset

## Motivación

### Propósito
El dataset German Credit fue creado para estudiar clasificación de riesgo crediticio. Esta hoja de datos documenta su uso en auditoría de equidad.

### Creadores
- Dataset original: Prof. Dr. Hans Hofmann, Institut für Statistik und Ökonometrie, Universität Hamburg
- Donado al Repositorio UCI ML: 1994

### Financiamiento
Investigación académica

## Composición

### Instancias
- **Total de instancias:** 1,000
- **Tipo de instancia:** Solicitantes de crédito individuales
- **Período de tiempo:** Histórico (Alemania pre-1994)

### Características

#### Numéricas (7)
| Característica | Descripción | Rango |
|----------------|-------------|-------|
| duration | Duración del préstamo en meses | 4-72 |
| credit_amount | Monto del préstamo en DM | 250-18,424 |
| installment_commitment | Tasa de cuota (% de ingreso) | 1-4 |
| residence_since | Años en residencia actual | 1-4 |
| age | Edad en años | 19-75 |
| existing_credits | Número de créditos existentes | 1-4 |
| num_dependents | Número de dependientes | 1-2 |

#### Categóricas (13)
| Característica | Categorías |
|----------------|------------|
| checking_status | <0, 0<=X<200, >=200, sin cuenta |
| credit_history | sin créditos, todos pagados, existentes pagados, retrasados, críticos |
| purpose | auto nuevo, auto usado, muebles, radio/tv, electrodomésticos, reparaciones, educación, vacaciones, reentrenamiento, negocio, otros |
| savings_status | <100, 100<=X<500, 500<=X<1000, >=1000, sin ahorros conocidos |
| employment | desempleado, <1, 1<=X<4, 4<=X<7, >=7 |
| other_parties | ninguno, co-solicitante, garante |
| property_magnitude | bienes raíces, seguro de vida, auto, sin propiedad conocida |
| other_payment_plans | banco, tiendas, ninguno |
| housing | alquiler, propio, gratis |
| job | desempleado/no calificado no residente, no calificado residente, calificado, alta calificación/gerencia |
| own_telephone | ninguno, sí |
| foreign_worker | sí, no |
| marital_status | soltero, divorciado/separado, casado/viudo |

#### Atributos Protegidos
| Atributo | Valores | Notas |
|----------|---------|-------|
| sex | hombre, mujer | Derivado de personal_status_sex |
| age_cat | joven (<25), mayor (>=25) | Derivado de age |
| foreign_worker | sí, no | Característica original |

#### Etiqueta
- **credit-risk:** bueno (700), malo (300)
- Desbalance de clases: 70% bueno, 30% malo

### Datos Faltantes
- Sin valores faltantes en versión procesada
- Codificación original usó códigos específicos para "desconocido"

### Relaciones
- Las instancias son solicitudes de crédito independientes
- No hay individuos duplicados conocidos

### Divisiones
- Práctica estándar: 80% train, 20% test
- Estratificado por etiqueta para mantener balance de clases

## Proceso de Recolección

### Recolección de Datos
- Recolectado de registros bancarios alemanes
- Metodología específica de recolección no documentada

### Estrategia de Muestreo
- No documentada; parece ser muestra por conveniencia
- Puede no ser representativo de todos los solicitantes de crédito

### Marco Temporal
- Datos históricos anteriores a 1994
- Instantánea única, no longitudinal

### Revisión Ética
- Sin revisión IRB o ética documentada
- Práctica común para datasets históricos

## Preprocesamiento

### Preprocesamiento Original
- Atributos simbólicos codificados con códigos (A11, A12, etc.)
- Atributos numéricos escalados

### Nuestro Preprocesamiento
1. Decodificar atributos simbólicos a etiquetas legibles
2. Extraer sexo del campo combinado personal_status_sex
3. Crear categoría de edad binaria (joven/mayor con corte en 25)
4. Binarizar atributos protegidos y etiqueta
5. Crear variables dummy para características categóricas

## Usos

### Usos Previstos
- Benchmark para algoritmos de scoring crediticio
- Investigación y educación en equidad
- Demostraciones de auditoría de sesgo en ML

### Usos Inapropiados
- **Decisiones de crédito en producción** (datos son históricos y específicos de Alemania)
- Generalizar a otros países/períodos de tiempo
- Evaluación de crédito a nivel individual

### Usos Previos
- Ampliamente usado en literatura de equidad en ML
- Destacado en ejemplos del toolkit AIF360
- Benchmark común en papers académicos

## Distribución

### Licencia
- Dominio público vía Repositorio UCI ML
- Sin restricciones de uso

### Acceso
- Repositorio UCI ML: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
- AIF360: Disponible a través de `aif360.sklearn.datasets`

## Mantenimiento

### Curador
- El Repositorio UCI ML mantiene el dataset
- Sin actualizaciones continuas (datos históricos estáticos)

### Versionado
- Versión única; sin actualizaciones planificadas

## Limitaciones y Sesgos

### Sesgos Conocidos
1. **Sesgo histórico:** Refleja prácticas de préstamo de Alemania de los 1990s
2. **Desbalance demográfico:** Más solicitantes hombres que mujeres
3. **Especificidad geográfica:** Contexto bancario alemán
4. **Obsolescencia temporal:** Las condiciones económicas han cambiado

### Brechas de Representación
- Representación limitada de algunos grupos demográficos
- Trabajadores extranjeros potencialmente subrepresentados
- Solicitantes jóvenes (<25) son minoría

### Problemas de Medición
- Sexo derivado de codificación de estado civil (posibles errores)
- Umbral de edad de 25 para "joven" es arbitrario
- Etiqueta de riesgo crediticio basada en evaluaciones históricas (potencialmente sesgada)

## Consideraciones Éticas

### Atributos Sensibles
- Sexo, edad, estatus de trabajador extranjero son protegidos en muchas jurisdicciones
- Usar estas características directamente puede violar leyes anti-discriminación

### Privacidad
- Dataset está anonimizado
- Sin información directamente identificable
- Riesgo de re-identificación considerado bajo debido a antigüedad de los datos

### Consentimiento
- Documentación de consentimiento original no disponible
- Práctica estándar para datasets públicos de esta era

## Notas Adicionales

### Citación
```
@misc{german_credit,
  author = {Hofmann, Hans},
  title = {Statlog (German Credit Data)},
  year = {1994},
  publisher = {UCI Machine Learning Repository},
  url = {https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)}
}
```

### Datasets Relacionados
- Adult Census Income (uso similar en investigación de equidad)
- COMPAS (equidad en justicia criminal)
- Bank Marketing (dominio similar)

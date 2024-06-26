### README

# Sistema de Clasificación de Noticias Reales y Falsas

## Objetivo del Trabajo
El objetivo de este proyecto es desarrollar un sistema de clasificación de noticias en línea, capaz de identificar de manera confiable si una noticia es real o falsa. Esto se logrará mediante la implementación de un modelo de aprendizaje automático que procese el texto de las noticias y las clasifique con un alto grado de precisión.

## Autores
- Estrada Fernández, Sara Gabriela - U20211A109
- Gonzales Astoray, Andrea Abigail - U20211C561
- Ramirez Sarmiento, Joseph Rafael - U20211C828

## Descripción del Caso de Uso
La proliferación de información en línea ha transformado la forma en que accedemos a las noticias, pero también ha incrementado la difusión de noticias falsas. Estas pueden influir negativamente en la opinión pública, generar confusión y desconfianza en los medios de comunicación y en las instituciones. Un sistema efectivo de clasificación de noticias reales y falsas puede:
- Mejorar la calidad y confiabilidad de la información consumida por el público.
- Reducir la propagación de desinformación.
- Fortalecer la transparencia de los medios.
- Contribuir a decisiones informadas.
- Fomentar un entorno digital más confiable.

### Preguntas Clave
1. ¿La noticia es real o falsa?
2. ¿Cuáles son las principales características que indican su veracidad?
3. ¿Cuál es el nivel de confianza en la clasificación?

### Implementación del Sistema
- **Recopilación de Datos:** Reunir un conjunto de datos de noticias reales y falsas.
- **Extracción de Características:** Analizar el estilo de redacción, uso del lenguaje, sesgos, coherencia lógica, etc.
- **Diseño del Modelo:** Entrenar un modelo de clasificación (red neuronal, árbol de decisión) para aprender y predecir la veracidad.
- **Evaluación y Optimización:** Probar y mejorar el modelo con datos de prueba para aumentar su precisión.
- **Implementación Escalable:** Desarrollar una solución que procese un gran volumen de noticias en tiempo real.

## Descripción del Conjunto de Datos
El conjunto de datos utilizado proviene de Kaggle y se compone de dos archivos:
- **Fake.csv:** 23,502 artículos de noticias falsas.
- **True.csv:** 21,417 artículos de noticias verdaderas.

### Variables del Dataset
| Columna | Descripción |
| ------- | ----------- |
| Title   | Título del artículo de noticias |
| Text    | Cuerpo o descripción del texto del artículo de noticias |
| Subject | Tema o categoría del artículo de noticias |
| Date    | Fecha de publicación del artículo de noticias |

Las características del dataset permiten un análisis profundo de los artículos, facilitando la comprensión de los contenidos y particularidades de noticias verdaderas y falsas. Este dataset proporciona una base sólida para aplicar técnicas de análisis exploratorio de datos (EDA) y aprendizaje automático en la detección de noticias falsas.

## Conclusiones
La implementación de este sistema de clasificación de noticias reales y falsas:
- Mejora la calidad y confiabilidad de la información.
- Reduce la desinformación y noticias falsas en línea.
- Fortalece la integridad de los medios de comunicación.
- Contribuye a una toma de decisiones más informada.
- Fomenta un entorno digital más saludable y confiable.

## Licencia
Este proyecto está bajo la licencia MIT.

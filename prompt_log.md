# Registro del proceso de desarrollo

Este documento registra las decisiones tomadas durante la construcción del paper, el notebook y las figuras, siguiendo el flujo de entrada, loop y salida solicitado en la tarea.

## 1. Definición de la entrada

Se partió de la tarea "un paper LaTeX reproducible sobre SHAP", que pedía elegir un modelo clasificador de imágenes, entre una y tres imágenes de prueba, y una pregunta explicativa concreta.

Se discutieron varias familias de modelos de Hugging Face (ResNet-50, ResNet-101/152, ConvNeXt, EfficientNet, MobileNet), evaluando el balance entre precisión, velocidad de inferencia y compatibilidad con SHAP. Dado que SHAP requiere cientos de evaluaciones del modelo por imagen, se priorizó una arquitectura convolucional liviana. Se seleccionó `facebook/convnext-tiny-224`, disponible en Hugging Face, preentrenado sobre ImageNet.

Como imágenes de prueba se usaron tres fotografías de gatos propios, correspondientes a tres individuos distintos con patrones de pelaje diferenciados (tuxedo blanco y negro, atigrado con blanco, atigrado sólido). Se corrigió durante la conversación un error inicial de mapeo entre nombres de archivo y contenido de las fotografías, verificando cada imagen de forma individual antes de continuar.

La pregunta definida fue la siguiente. ¿El modelo se basa en rasgos anatómicos del gato para clasificarlo, o se apoya en atajos visuales como el color del pelaje o elementos del fondo? Esta pregunta se enmarca en el concepto de *shortcut learning*, documentado en la literatura de explicabilidad.

## 2. Loop de construcción

1. Se instaló la librería `shap` sobre un entorno que ya contaba con `torch` y `transformers`.
2. Se implementó un primer script que cargaba el modelo mediante un pipeline de `transformers` y lo pasaba directamente a `shap.Explainer`. La ejecución de las predicciones fue correcta, pero el cálculo de valores SHAP falló, ya que el wrapper interno de SHAP para pipelines de Hugging Face (`TransformersPipeline`) está diseñado para pipelines de texto y no acepta arreglos de imagen en formato numérico.
3. Se corrigió el script reemplazando el pipeline por una función de predicción propia, que aplica el preprocesamiento del `AutoImageProcessor` correspondiente y llama directamente al modelo (`AutoModelForImageClassification`), devolviendo probabilidades mediante softmax. Esta función se pasó como modelo a `shap.Explainer`, junto con un enmascarador de imagen (`shap.maskers.Image` con el método `inpaint_telea`).
4. Se ejecutó el análisis sobre las tres imágenes, generando las predicciones y los mapas de atribución SHAP correspondientes a las clases con mayor probabilidad.

## 3. Salida

El paquete entregable incluye el archivo `paper.tex` con la redacción del análisis, las figuras generadas en la carpeta `figures/`, el notebook reproducible en `notebooks/shap_analysis.ipynb`, el archivo de referencias `references.bib`, y este registro de prompts y limitaciones.

## Limitaciones del proceso

El análisis se restringe a una única arquitectura de modelo y a tres imágenes, por lo que las conclusiones tienen un alcance exploratorio y no permiten generalizaciones estadísticas. Asimismo, el cálculo de los valores SHAP es aproximado, dado que SHAP utiliza muestreo para estimar las contribuciones de cada región en lugar de un cálculo exacto sobre todas las combinaciones posibles.

import cv2
import numpy as np

# Cargar imagen
imagen = cv2.imread("imagenes/lineas_esquinas.jpg")

# Verificar que la imagen exista
if imagen is None:
    print("Error: no se pudo cargar la imagen.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Convertir a tipo float32
gris_float = np.float32(gris)

# Detectar esquinas mediante Harris
esquinas = cv2.cornerHarris(
    gris_float,
    2,
    3,
    0.04
)

# Dilatar para hacer visibles las esquinas
esquinas = cv2.dilate(
    esquinas,
    None
)

# Crear copia
resultado = imagen.copy()

# Umbral para identificar esquinas
umbral = 0.05 * esquinas.max()

# Marcar esquinas
resultado[esquinas > umbral] = [0, 0, 255]

# Mostrar resultados
cv2.imshow(
    "Imagen original NC = 0142",
    imagen
)

cv2.imshow(
    "Esquinas detectadas NC = 0142",
    resultado
)

# Guardar resultado
cv2.imwrite(
    "resultados/ejemplo2_esquinas.jpg",
    resultado
)

# Contar esquinas aproximadas
cantidad_esquinas = np.sum(
    esquinas > umbral
)

print("Deteccion de esquinas terminada.")
print("Cantidad aproximada de puntos detectados:",
      cantidad_esquinas)

print("Resultado guardado en:")
print("resultados/ejemplo2_esquinas.jpg")

# Esperar una tecla
cv2.waitKey(0)

# Cerrar ventanas
cv2.destroyAllWindows()
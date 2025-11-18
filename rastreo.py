# Importa la librería YOLO al principio de tu script
from ultralytics import YOLO
import numpy as np
import cv2
import time
from collections import OrderedDict # Para manejar los objetos

# (Aquí iría todo tu código existente: clase captura_video, hex2bgr, Pipelines_ImageProcessing)

#################################################################
# CLASE DE PIPELINE DE RASTREO (TRACKING)
#################################################################

class TrackerPipeline:
    def __init__(self, max_disappeared=30, base_tolerance_px=50, depth_scale_factor=0.3):
        """
        Inicializa el rastreador de centroides.

        Args:
            yolo_model (YOLO): El modelo YOLOv8 (pose o detección) ya cargado.
            max_disappeared (int): N.º de fotogramas para dar de baja un ID perdido.
            base_tolerance_px (int): Tolerancia base de distancia (en píxeles) para coincidencia.
            depth_scale_factor (float): Factor para escalar la tolerancia con el tamaño de la caja.
                                       (Ajustar este valor es CRÍTICO).
        """
        self.yolo_model = YOLO('DSP_Deteccion_Rastreo_Personas/model_weights/yolo11n-pose.pt')
        
        # Almacenamiento de objetos activos
        self.objects = OrderedDict()
        
        # ID único para el próximo objeto a registrar
        self.next_object_ID = 0
        
        # N.º de fotogramas que un objeto puede "desaparecer" antes de ser eliminado
        self.max_disappeared = max_disappeared
        
        # --- Parámetros clave para tu requisito ---
        
        # Distancia base mínima para el rastreo
        self.base_tolerance = base_tolerance_px
        
        # Cuánto influye el ancho de la caja en la tolerancia
        # tolerancia_total = base_tolerance + (ancho_caja * depth_scale_factor)
        self.depth_scale = depth_scale_factor
        
        # Almacén de colores para los IDs
        self.id_colors = {}

    def _get_color_for_id(self, object_id):
        """Genera un color único y consistente para cada ID."""
        if object_id not in self.id_colors:
            # Generar un color aleatorio basado en el ID
            np.random.seed(object_id * 42) # Semilla consistente
            color = np.random.randint(0, 255, size=3).tolist()
            self.id_colors[object_id] = (int(color[0]), int(color[1]), int(color[2]))
        return self.id_colors[object_id]

    def _register(self, centroid, box):
        """Registra un nuevo objeto (persona) con un nuevo ID."""
        object_id = self.next_object_ID
        self.objects[object_id] = {
            "centroid": centroid,
            "box": box, # Almacenamos la caja (x1, y1, x2, y2)
            "disappeared": 0
        }
        self.next_object_ID += 1
        print(f"[Tracker] Registrado nuevo ID: {object_id}")

    def _deregister(self, object_id):
        """Elimina un objeto que ha desaparecido."""
        print(f"[Tracker] ID {object_id} eliminado (desaparecido).")
        del self.objects[object_id]
        if object_id in self.id_colors:
            del self.id_colors[object_id] # Limpiar color

    def process(self, frame):
        """
        El método principal del pipeline. Recibe un fotograma,
        realiza la detección y actualiza el rastreo.
        """
        
        # --- 1. DETECCIÓN (Con YOLO) ---
        # Ejecutar la inferencia. Usamos 'person' (clase 0 en COCO).
        # Ajusta 'classes=0' si tu modelo YOLO es diferente (ej. solo rostros).
        results = self.yolo_model(frame, classes=0, verbose=False) 
        
        # Almacenamos los centroides y cajas de este fotograma
        new_detections = []
        for r in results:
            for box in r.boxes:
                # Obtenemos solo detecciones de 'persona' con confianza > 0.5
                if box.cls[0] == 0 and box.conf[0] > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Calcular centroide
                    cX = int((x1 + x2) / 2.0)
                    cY = int((y1 + y2) / 2.0)
                    new_detections.append(((cX, cY), (x1, y1, x2, y2)))

        # --- 2. MANEJO DE OBJETOS ---

        # Si no hay detecciones, marcamos todos los objetos existentes como "desaparecidos"
        if len(new_detections) == 0:
            ids_to_deregister = []
            for object_id in self.objects.keys():
                self.objects[object_id]["disappeared"] += 1
                if self.objects[object_id]["disappeared"] > self.max_disappeared:
                    ids_to_deregister.append(object_id)
            
            for object_id in ids_to_deregister:
                self._deregister(object_id)
                
            return frame # Devuelve el fotograma sin dibujos

        # Listas para el proceso de coincidencia
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid]["centroid"] for oid in object_ids]
        object_boxes = [self.objects[oid]["box"] for oid in object_ids]
        
        new_centroids = [d[0] for d in new_detections]
        new_boxes = [d[1] for d in new_detections]

        # --- 3. LÓGICA DE COINCIDENCIA (MATCHING) ---
        
        # Calculamos la distancia L2 (Euclidiana) entre todos los pares
        # de centroides (viejos vs nuevos)
        if len(object_centroids) > 0 and len(new_centroids) > 0:
            # Convertimos a numpy para cálculo vectorizado
            old_centroids_np = np.array(object_centroids)
            new_centroids_np = np.array(new_centroids)
            
            # Matriz de Distancia: dists[i, j] es la distancia entre 
            # el objeto 'i' y la detección 'j'.
            dists = np.linalg.norm(old_centroids_np[:, np.newaxis, :] - new_centroids_np[np.newaxis, :, :], axis=2)
            
            # Índices de las filas (objetos) ordenados por distancia
            rows = dists.min(axis=1).argsort()
            
            # Índices de las columnas (detecciones) ordenados por distancia
            cols = dists.argmin(axis=1)[rows]

        else:
            # Si no hay objetos previos, todas las detecciones son nuevas
            rows, cols = [], []

        
        used_rows = set()
        used_cols = set()
        
        # --- 4. APLICAR TOLERANCIA DINÁMICA ---
        
        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # ESTA ES TU LÓGICA CLAVE:
            object_id = object_ids[row]
            box_width = self.objects[object_id]["box"][2] - self.objects[object_id]["box"][0]
            
            # La tolerancia es la base + (ancho * factor)
            dynamic_tolerance = self.base_tolerance + (box_width * self.depth_scale)
            
            distance = dists[row, col]

            # Si la distancia está dentro de nuestra tolerancia dinámica
            if distance < dynamic_tolerance:
                # Coincidencia encontrada
                self.objects[object_id]["centroid"] = new_centroids[col]
                self.objects[object_id]["box"] = new_boxes[col]
                self.objects[object_id]["disappeared"] = 0
                
                used_rows.add(row)
                used_cols.add(col)

        # --- 5. REGISTRAR Y DESREGISTRAR OBJETOS ---

        # A) Objetos que no coincidieron (o no se usaron)
        unused_rows = set(range(len(object_centroids))).difference(used_rows)
        ids_to_deregister = []
        for row in unused_rows:
            object_id = object_ids[row]
            self.objects[object_id]["disappeared"] += 1
            if self.objects[object_id]["disappeared"] > self.max_disappeared:
                ids_to_deregister.append(object_id)
        
        for object_id in ids_to_deregister:
            self._deregister(object_id)

        # B) Nuevas detecciones que no coincidieron
        unused_cols = set(range(len(new_centroids))).difference(used_cols)
        for col in unused_cols:
            self._register(new_centroids[col], new_boxes[col])

        # --- 6. DIBUJAR RESULTADOS ---
        
        for (object_id, data) in self.objects.items():
            (x1, y1, x2, y2) = data["box"]
            color = self._get_color_for_id(object_id)
            
            # Dibujar la caja
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Dibujar el ID
            text = f"ID: {object_id}"
            y_pos = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(frame, text, (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame
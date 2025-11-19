# Importa la librería YOLO al principio de tu script
from ultralytics import YOLO
import numpy as np
import cv2
import time
from collections import OrderedDict # Para manejar los objetos
from collections import defaultdict

# (Aquí iría todo tu código existente: clase captura_video, hex2bgr, Pipelines_ImageProcessing)

#################################################################
# CLASE DE PIPELINE DE RASTREO (TRACKING)
#################################################################

class TrackerPipeline:
    def __init__(self, max_disappeared=60, base_tolerance_px=50, depth_scale_factor=0.3):
        """
        Inicializa el rastreador de centroides.

        Args:
            yolo_model (YOLO): El modelo YOLOv8 (pose o detección) ya cargado.
            max_disappeared (int): N.º de fotogramas para dar de baja un ID perdido.
            base_tolerance_px (int): Tolerancia base de distancia (en píxeles) para coincidencia.
            depth_scale_factor (float): Factor para escalar la tolerancia con el tamaño de la caja.
                                       (Ajustar este valor es CRÍTICO).
        """
        self.yolo_model = YOLO('DSP_Deteccion_Rastreo_Personas/model_weights/yolo11n.pt')
                # Verificar si está usando GPU o CPU
        device = self.yolo_model.device
        print(f"El modelo está corriendo en: {device}")
        
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


        # Store the track history
        self.track_history = defaultdict(lambda: [])

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

    def CentroidesTracker(self, frame):
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
    
    def YOLOtracker(self, frame):
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        result = self.yolo_model.track(frame, persist=True, classes=0, verbose=False)[0]

        # Get the boxes and track IDs
        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()

            # Visualize the result on the frame
            frame = result.plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = self.track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=hex2bgr("#FFED94"), thickness=5)


        return frame

class OpticalFlowTracker:
    def __init__(self, max_points=100):
        """
        Inicializa el rastreador de Flujo Óptico usando CascadeClassifier para la detección.
        
        Args:
            cascade_path (str): Ruta al archivo XML del clasificador de Haar (ej. 'haarcascade_fullbody.xml').
            max_points (int): Número máximo de puntos a rastrear simultáneamente.
        """
        
        # Cargar el CascadeClassifier
        self.detector_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

        # --- Parámetros del Optical Flow (Lucas-Kanade) ---
        self.feature_params = dict(maxCorners=max_points,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Estado interno
        self.old_gray = None
        self.p0 = None
        self.mask = None
        self.color = np.random.randint(0, 255, (max_points, 3))
        self.max_points = max_points
        
        # Para desvanecer las estelas de optical flow
        self.decay_factor = 0.95

        # Estado del Rastreador de Centroides
        self.next_object_ID = 0
        self.objects = {}       # {ID: {'centroid': (x,y), 'disappeared': int, 'box_wh': (w,h)}}
        self.max_disappeared = 15 # Frames antes de eliminar un ID
        self.tracking_tolerance = 120 # Distancia máxima en píxeles para coincidencia

        # Métodos de ayuda para la gestión de IDs
        self.id_colors = {}

    def _get_color_for_id(self, object_id):
        if object_id not in self.id_colors:
            np.random.seed(object_id * 42)
            color = np.random.randint(0, 255, size=3).tolist()
            self.id_colors[object_id] = (int(color[0]), int(color[1]), int(color[2]))
        return self.id_colors[object_id]

    def _register(self, centroid, box):
        object_id = self.next_object_ID
        x, y, w, h = box
        self.objects[object_id] = {
            "centroid": centroid,
            "disappeared": 0,
            "box_wh": (w, h)
        }
        self.next_object_ID += 1
        # print(f"[Tracker] Registrado nuevo ID: {object_id}")

    def _deregister(self, object_id):
        # print(f"[Tracker] ID {object_id} eliminado.")
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.id_colors:
            del self.id_colors[object_id]

    def _detect_people_mask(self, frame_gray):
        """
        Crea una máscara a partir de las detecciones del CascadeClassifier.
        """
        
        # Detección con Haar (ajusta los parámetros según la necesidad, 1.1 y 5 son comunes)
        detections = self.detector_cascade.detectMultiScale(
            frame_gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        mask = np.zeros_like(frame_gray)
        
        # Dibujar rectángulos blancos en la máscara donde se detectó una persona
        for (x, y, w, h) in detections:
            # Dibujamos un rectángulo relleno (color 255)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            
        return mask, detections

    def process(self, frame):
        """
        Pipeline principal. Ahora asigna un ID persistente a cada detección.
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Detección (Haar) y Preparación
        detection_mask, current_detections_haar = self._detect_people_mask(frame_gray)
        
        # Convertir detecciones Haar (x, y, w, h) a lista de (centroid, box_coords)
        new_detections = []
        new_centroids = []
        for (x, y, w, h) in current_detections_haar:
            cX, cY = int(x + w / 2), int(y + h / 2)
            new_centroids.append((cX, cY))
            new_detections.append({'centroid': (cX, cY), 'box': (x, y, w, h)})
        
        # --- 2. RASTREO DE CENTROIDES (ASIGNACIÓN DE ID) ---
        
        object_ids = list(self.objects.keys())
        old_centroids = np.array([self.objects[oid]["centroid"] for oid in object_ids])
        
        matched_new_indices = set()
        
        # a) Buscar coincidencias
        if len(self.objects) > 0 and len(new_centroids) > 0:
            new_centroids_np = np.array(new_centroids)
            
            # Matriz de Distancia entre centros viejos y nuevos
            dists = np.linalg.norm(old_centroids[:, np.newaxis, :] - new_centroids_np[np.newaxis, :, :], axis=2)
            
            # Recorrer objetos antiguos
            for row in range(len(object_ids)):
                object_id = object_ids[row]
                
                # Encontrar la nueva detección más cercana al objeto actual
                closest_new_index = np.argmin(dists[row, :])
                min_distance = dists[row, closest_new_index]
                
                # Si la distancia está dentro de la tolerancia y no se ha usado
                if min_distance < self.tracking_tolerance and closest_new_index not in matched_new_indices:
                    # Coincidencia encontrada: Actualizar objeto
                    new_det = new_detections[closest_new_index]
                    self.objects[object_id]["centroid"] = new_det['centroid']
                    self.objects[object_id]["disappeared"] = 0
                    self.objects[object_id]["box_wh"] = new_det['box'][2:] # (w, h)
                    matched_new_indices.add(closest_new_index)
                else:
                    # No coincidió: Marcar como desaparecido
                    self.objects[object_id]["disappeared"] += 1

        # b) Registrar nuevos y desregistrar perdidos
        
        # Desregistrar objetos que han desaparecido por mucho tiempo
        ids_to_deregister = [oid for oid in object_ids if self.objects[oid]["disappeared"] > self.max_disappeared]
        for oid in ids_to_deregister:
            self._deregister(oid)

        # Registrar nuevas detecciones no coincidentes
        for col, det in enumerate(new_detections):
            if col not in matched_new_indices:
                self._register(det['centroid'], det['box'])


        # --- 3. INICIALIZACIÓN DE PUNTOS LK (OPTICAL FLOW) ---
        
        # Re-inicializamos si no hay puntos o si la cantidad de puntos cae demasiado
        if self.p0 is None or len(self.p0) < self.max_points // 4:
            if len(current_detections_haar) > 0:
                self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=detection_mask, **self.feature_params)
            self.old_gray = frame_gray.copy()
            if self.p0 is None:
                return frame
        
        
        # --- 4. CÁLCULO Y DIBUJO DEL FLUJO ÓPTICO Y LAS CAJAS CON ID ---
        
        if self.p0 is not None and len(self.p0) > 0:
            
            # ... (Cálculo LK y selección de puntos buenos)
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]

            # Aplicar decaimiento a la máscara
            if self.mask is None: self.mask = np.zeros_like(frame)
            self.mask = cv2.addWeighted(self.mask, self.decay_factor, self.mask, 0, 0) 
            
            # Dibujar flujo
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                color = self.color[i % self.max_points].tolist()
                self.mask = cv2.line(self.mask, (int(c), int(d)), (int(a), int(b)), color, 2)
                frame = cv2.circle(frame, (int(a), int(b)), 3, color, -1)
            
            # 5. Dibujar caja y ID persistente
            for object_id, data in self.objects.items():
                cX, cY = data['centroid']
                w, h = data['box_wh']
                x1, y1 = int(cX - w / 2), int(cY - h / 2)
                x2, y2 = int(cX + w / 2), int(cY + h / 2)
                
                color = self._get_color_for_id(object_id)
                
                # Dibujar rectángulo (usamos el color del ID)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Dibujar ID text encima de la caja
                text = f"ID: {object_id}"
                y_pos = y1 - 10 if y1 - 10 > 10 else y1 + 25 # Posición superior o debajo si es muy alto
                cv2.putText(frame, text, (x1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
            # Combinar la imagen original con la máscara de líneas
            frame = cv2.add(frame, self.mask)

            # 6. Actualizar estado LK
            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
        
        return frame

def hex2bgr(hex_color): 
    """Convierte una cadena HEX (#RRGGBB) a una tupla BGR para OpenCV."""
    # Eliminar el '#' si está presente
    hex_color = hex_color.lstrip('#')
    
    # Convierte de HEX a RGB (en decimal)
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Reordena a BGR para OpenCV
    bgr = (rgb[2], rgb[1], rgb[0])
    return bgr
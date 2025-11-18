import cv2
import time
import threading
import queue
from collections import deque
import numpy as np
from ultralytics import YOLO


class captura_video:
    def __init__(self, fps=60.0, camera=cv2.VideoCapture(0), video_path="DSP_Deteccion_Rastreo_Personas/videos/"):
        self.fps = fps
        self.cam = camera
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)) # 640 px
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 480 px
        print(f"Resolución de la cámara: {self.frame_width}x{self.frame_height}")
        # Define the codec and create VideoWriter object
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_path = video_path
        # Queues para la comunicación entre hilos
        self.frame_queue = queue.Queue()
        self.video_queue = queue.Queue()

    def record(self, save_video=False, pipe_filtrado=None):
        """Inicia la captura de video desde la cámara, mostrando en pantalla y opcionalmente guardando en disco."""
        # Iniciar los hilos
        reader_thread = threading.Thread(target=self.camera_reader, args=(save_video,pipe_filtrado,))
        writer_thread = threading.Thread(target=self.video_writer)

        reader_thread.start()
        if save_video:
            writer_thread.start()

        fps_text = "FPS: 0.00"
        fps_buffer = deque(maxlen=20)
        while True:
            try:
                prev_time = time.time()
                # Intenta obtener un fotograma de la cola de visualización
                frame = self.frame_queue.get(timeout=1)

                # 3. Muestra el texto de FPS sobre el fotograma
                cv2.putText(frame, 
                            fps_text, 
                            (10, 30),                         # Posición (x, y)
                            cv2.FONT_HERSHEY_SIMPLEX,         # Tipo de fuente
                            1,                                # Escala de la fuente
                            (255, 255, 255),                      # Color (BGR: Blanco)
                            2)                                # Grosor del texto
                
                cv2.imshow('Camera', frame)

                if cv2.waitKey(1) == ord('q'):
                    break
                
                curr_time = time.time()
                time_diff = curr_time - prev_time

                fps_actual = 1 / time_diff
                fps_buffer.append(fps_actual)
                # Formatea el texto (limitando a 2 decimales)
                fps_text = f"FPS: {np.mean(fps_buffer):.1f}"

            except queue.Empty:
                # Si no hay fotogramas, significa que el hilo de lectura terminó
                break


        # Limpieza
        print("Cerrando...")
        self.cam.release()
        cv2.destroyAllWindows()
        # Esperar a que los hilos terminen (aunque ya deberían haberlo hecho)
        reader_thread.join()
        if save_video:
            writer_thread.join()
        print("Listo.")


    def camera_reader(self, save_video, pipe_filtrado):
        """Lee fotogramas de la cámara tan rápido como sea posible."""
        print("Iniciando hilo de lectura...")
        delayed = 0
        start_recording = time.time()
        while True:
            start = time.time()
            ret, frame = self.cam.read()
            #end = time.time()
            #print("Tiempo en obtener frame:", end-start) #0.053 segundos aprox, lo cual es un cuello de botella desde raíz.
            if not ret:
                break
            
            # Pasa el fotoframa por el pipeline de procesamiento
            filtered_frame = pipe_filtrado(frame) if pipe_filtrado is not None else frame

            # Pone el fotograma en las colas para mostrar y grabar
            self.frame_queue.put(filtered_frame)
            if save_video:
                self.video_queue.put(frame)

            # Wait for a short period to control the frame rate
            end = time.time()
            pipe_time = end - start
            wait = 1/self.fps - pipe_time if (1/self.fps - pipe_time) > 0 else 0
            delayed += 1 if wait == 0 else 0
            time.sleep(wait)

        # Print recording summary
        record_time = time.time() - start_recording
        print(f"Frames delayed due to processing time: {delayed*100/(record_time*self.fps):.2f}%")
        print("Finalizando hilo de lectura.")


    def video_writer(self):
        """Toma fotogramas de la cola y los escribe en el disco."""
        print("Iniciando hilo de escritura...")
        out = cv2.VideoWriter(self.video_path+'output.mp4', self.fourcc, fps=self.fps, frameSize=(self.frame_width, self.frame_height))
        while True:
            try:
                # Espera a que haya un fotograma en la cola (con un timeout)
                frame = self.video_queue.get(timeout=1)
                out.write(frame)
            except queue.Empty:
                # Si la cola está vacía por 1 segundo, asumimos que la lectura terminó.
                break
        print("Finalizando hilo de escritura.")
        out.release()

def hex2bgr(hex_color): 
    """Convierte una cadena HEX (#RRGGBB) a una tupla BGR para OpenCV."""
    # Eliminar el '#' si está presente
    hex_color = hex_color.lstrip('#')
    
    # Convierte de HEX a RGB (en decimal)
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Reordena a BGR para OpenCV
    bgr = (rgb[2], rgb[1], rgb[0])
    return bgr

class Pipelines_ImageProcessing:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml') #haarcascade_frontalface_default
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

        self.pose_model = YOLO('model_weights/yolo11n-pose.pt') 

    def edges(self, frame):
        """Pipeline de procesamiento de fotogramas para suvizado y detección de bordes."""
        start = time.time()
        # Convierte a escala de grises
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Aplica un desenfoque gaussiano
        #processed_frame = cv2.GaussianBlur(processed_frame, (3, 3), 0)
        #processed_frame = cv2.blur(processed_frame, (3, 3))
        #processed_frame = cv2.medianBlur(processed_frame, 3)
        processed_frame = cv2.bilateralFilter(processed_frame, 5, 30, 50) # Busca reducir ruido cuidando mantener bordes. Calibrado con Canny
        #processed_frame = cv2.bilateralFilter(processed_frame, 13, 75, 75) # Calibrado con Sobel
        # Aplica método de detección de bordes
        processed_frame = cv2.Canny(processed_frame, 35, 60)
        #sobelx = cv2.Sobel(processed_frame, cv2.CV_64F, 1, 0, ksize=13)
        #sobely = cv2.Sobel(processed_frame, cv2.CV_64F, 0, 1, ksize=13)
        #processed_frame = cv2.magnitude(sobelx*0.45, sobely*0.45)
        end = time.time()
        #print(f"Tiempo de procesamiento del frame: {(end - start):.2} segundos")
        return processed_frame

    def faceComponentsCV2(self, frame):
        """Pipeline de procesamiento de fotogramas para detección de cara o perfil, ojos y boca usando OpenCV."""
        start = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 5, 30, 50)
        
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        profile = False
        if len(faces) == 0:
            profile = True
            faces = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        eyes = self.eyes_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30)) 

        smile = self.smile_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=7, minSize=(30, 20), maxSize=(80,40)) 
        
        face_x, face_y, face_w, face_h = 0,0,0,0
        for (x, y, w, h) in faces:
            face_x, face_y, face_w, face_h = x, y, w, h
            if profile:
                color = "#650EE8"
                cv2.rectangle(frame, (x, y), (x + w, y + h), hex2bgr(color), 2)
                cv2.putText(frame, 'profile', (x, y-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, hex2bgr(color), 1)
            else:
                color = "#4E75E8"
                cv2.rectangle(frame, (x, y), (x + w, y + h), hex2bgr(color), 2)
                cv2.putText(frame, 'face', (x, y-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, hex2bgr(color), 1)
            break

        eyes_count = 0
        for (x, y, w, h) in eyes:
            if y > face_y and y < face_y+face_h*2/3 and x > face_x and x < face_x+face_w:
                if eyes_count < 2:
                    eyes_count += 1
                else:
                    break
                color = "#C0E80E"
                cv2.putText(frame, 'eye', (x, y-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, hex2bgr(color), 1)
                x = x + w//2
                y = y + h//2
                cv2.circle(frame, (x, y), w//2, hex2bgr(color), 2)
                

        for (x, y, w, h) in smile:
            # Asegurar que la sonrisa esté dentro de la región segura de la cara y filtrar falsas detecciones
            if y > face_y+(face_h)*5/8 and y < face_y+face_h*5/6 and x > face_x+face_w*1/8 and x < face_x+face_w*7/8: 
                color = "#AC8D59"
                cv2.rectangle(frame, (x, y), (x + w, y + h), hex2bgr(color), 2)
                cv2.putText(frame, 'smile', (x, y-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, hex2bgr(color), 1)
                break
        
        end = time.time()
        #print(f"Tiempo de ejecución del pipeline: {(end - start):.4} segundos")

        return frame
    
    def faceComponentsYOLO(self, frame):
        """
        Pipeline de procesamiento de fotogramas para detección de pose usando YOLO Pose.
        Extrae y dibuja el ojo izquierdo.
        """
        start = time.time()

        # Ejecutar la inferencia de pose. 
        # 'verbose=False' evita que imprima información en cada frame.
        results = self.pose_model(frame, verbose=False)

        # Iterar sobre los resultados (normalmente solo hay un 'result')
        for result in results:
            
            # 1. Acceder al objeto de keypoints
            keypoints = result.keypoints

            # 2. Verificar si se detectó al menos una persona
            if keypoints and keypoints.shape[0] > 0:
                
                # keypoints.xy es un tensor de forma [num_personas, 17_keypoints, 2 (x,y)]
                # Iteramos sobre cada persona detectada en el frame
                for person_kps in keypoints.xy:
                    
                    # 3. Obtener el keypoint del OJO IZQUIERDO (Índice 1)
                    # El estándar COCO define: 0=nariz, 1=ojo_izq, 2=ojo_der
                    nose_coords = person_kps[0]
                    left_eye_coords = person_kps[1]
                    right_eye_coords = person_kps[2]
                    nose = Keypoint(nose_coords, "nose")
                    left_eye = Keypoint(left_eye_coords, "left_eye")
                    right_eye = Keypoint(right_eye_coords, "right_eye")
                    radio = abs(left_eye.x - right_eye.y) // 4

                    color = "#C0E80E"
                    if left_eye.x > 0 and left_eye.y  > 0:
                        # (Opcional) Dibujar un círculo sobre el ojo en el frame
                        cv2.circle(frame, (left_eye.x, left_eye.y), radius=radio, color=hex2bgr(color), thickness=1)
                        cv2.putText(frame, left_eye.label, (left_eye.x, left_eye.y-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, hex2bgr(color), 1)
                    
                    if right_eye.x > 0 and right_eye.y  > 0:
                        # (Opcional) Dibujar un círculo sobre el ojo en el frame
                        cv2.circle(frame, (right_eye.x, right_eye.y), radius=radio, color=hex2bgr(color), thickness=1)
                        cv2.putText(frame, right_eye.label, (right_eye.x, right_eye.y-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, hex2bgr(color), 1)

                    smile_x = right_eye.x - abs(right_eye.x - left_eye.x)//6 - left_eye.y + right_eye.y
                    smile_y = nose.y + (nose.y - left_eye.y)*2//3
                    smile_w = abs(right_eye.x - left_eye.x) *4//3
                    smile_h = smile_w // 2
                    
                    profile = False
                    if nose.x < right_eye.x or nose.x > left_eye.x:
                        profile = True

                    if smile_x > 0 and smile_y > 0 and not profile:
                        color = "#AC8D59"
                        cv2.rectangle(frame, (smile_x, smile_y), (smile_x + smile_w, smile_y + smile_h), hex2bgr(color), 2)
                        cv2.putText(frame, 'smile', (smile_x, smile_y-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, hex2bgr(color), 1)

                    face_x = right_eye.x - abs(right_eye.x - left_eye.x)*5//6
                    face_y = right_eye.y - abs(right_eye.y - nose.y)*2
                    face_w = abs(right_eye.x - left_eye.x)*16//6
                    face_h = face_w*7//6

                    profile_x = right_eye.x - abs(right_eye.x - left_eye.x)*2//3
                    profile_y = right_eye.y - abs(right_eye.y - nose.y)*2
                    profile_w = abs(right_eye.x - left_eye.x) *4
                    profile_h = profile_w *6//5

                    if profile:
                        color = "#650EE8"
                        cv2.rectangle(frame, (profile_x, profile_y), (profile_x + profile_w, profile_y + profile_h), hex2bgr(color), 2)
                        cv2.putText(frame, 'profile', (profile_x, profile_y-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, hex2bgr(color), 1)
                    else:
                        color = "#4E75E8"
                        cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), hex2bgr(color), 2)
                        cv2.putText(frame, 'face', (face_x, face_y-10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, hex2bgr(color), 1)
                    break


                    

        end = time.time()
        #print(f"Tiempo de ejecución del pipeline YOLO Pose: {(end - start):.4} segundos")

        # Devolver el frame procesado
        return frame

class Keypoint:
    def __init__(self, coords, label):
        self.x = int(coords[0])
        self.y = int(coords[1])
        self.label = label


if __name__ == "__main__":
    captura = captura_video(fps=16.0)
    pipelines = Pipelines_ImageProcessing()
    captura.record(save_video=False, pipe_filtrado=pipelines.faceComponentsYOLO)

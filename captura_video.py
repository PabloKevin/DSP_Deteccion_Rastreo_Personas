import cv2
import time
import threading
import queue
from collections import deque
import numpy as np
from ultralytics import YOLO
from deteccion import Pipelines_ImageProcessing
import os


def hex2bgr(hex_color): 
    """Convierte una cadena HEX (#RRGGBB) a una tupla BGR para OpenCV."""
    # Eliminar el '#' si está presente
    hex_color = hex_color.lstrip('#')
    
    # Convierte de HEX a RGB (en decimal)
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Reordena a BGR para OpenCV
    bgr = (rgb[2], rgb[1], rgb[0])
    return bgr

class captura_video:
    def __init__(self, fps=60.0, video_path="DSP_Deteccion_Rastreo_Personas/videos/"):
        self.fps = fps
        self.cam = cv2.VideoCapture(0)
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
        
        i=0
        path = None
        while True:
            path = self.video_path+"output"+str(i)+".mp4"
            if not os.path.isfile(path):
                break
            else:
                i+=1


        out = cv2.VideoWriter(path, self.fourcc, fps=self.fps, frameSize=(self.frame_width, self.frame_height))
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






if __name__ == "__main__":
    captura = captura_video(fps=16.0)
    pipelines = Pipelines_ImageProcessing()
    captura.record(save_video=False, pipe_filtrado=pipelines.faceComponentsYOLO)

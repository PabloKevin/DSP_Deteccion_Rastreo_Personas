import cv2
import time
import threading
import queue
from collections import deque
import numpy as np

class captura_video:
    def __init__(self, fps=60.0, camera=cv2.VideoCapture(0), video_path="DSP_Deteccion_Rastreo_Personas/videos/"):
        self.fps = fps
        self.cam = camera
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Define the codec and create VideoWriter object
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_path = video_path
        # Queues para la comunicación entre hilos
        self.frame_queue = queue.Queue()
        self.video_queue = queue.Queue()

    def record(self, save_video=False, pipe_filtrado=None):
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
                            (255, 255, 255),                      # Color (BGR: Verde)
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

    
def pipe(frame):
    """Pipeline de procesamiento de fotogramas."""
    #start = time.time()
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
    #end = time.time()
    #print(f"Tiempo de procesamiento del frame: {(end - start):.2} segundos")
    return processed_frame

if __name__ == "__main__":
    captura = captura_video(fps=30.0)
    captura.record(save_video=True, pipe_filtrado=pipe)

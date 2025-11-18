from captura_video import captura_video
from deteccion import Pipelines_ImageProcessing
from rastreo import TrackerPipeline

def apartado32():
    captura = captura_video(fps=16.0)
    pipelines = Pipelines_ImageProcessing()
    captura.record(save_video=False, pipe_filtrado=pipelines.edges)

def apartado331():
    captura = captura_video(fps=16.0)
    pipelines = Pipelines_ImageProcessing()
    captura.record(save_video=False, pipe_filtrado=pipelines.faceComponentsCV2)

def apartado332():
    captura = captura_video(fps=16.0)
    pipelines = Pipelines_ImageProcessing()
    captura.record(save_video=False, pipe_filtrado=pipelines.faceComponentsYOLO)

def apartado34():
    captura = captura_video(fps=16.0)
    pipelines = TrackerPipeline()
    captura.record(save_video=False, pipe_filtrado=pipelines.process)

if __name__ == "__main__":
    while True:
        apartado = input("\n\nSelecciones apartado a ejecutar: ")
        if apartado == "32":
            apartado32()
        elif apartado == "331":
            apartado331()
        elif apartado == "332":
            apartado332()
        elif apartado == "34":
            apartado34()
        elif apartado == "exit":
            break
        else: 
            print("Opción no válida. Intente de nuevo. Si desea salir, escriba 'exit'.")
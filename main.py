from captura_video import captura_video
from deteccion import Pipelines_ImageProcessing
from rastreo import TrackerPipeline, OpticalFlowTracker

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

def apartado340():
    pipe1 = Pipelines_ImageProcessing()
    pipe2 = TrackerPipeline()
    def deteccionRastreo(frame):
        frame = pipe1.faceComponentsYOLO(frame)
        frame = pipe2.CentroidesTracker(frame)
        return frame
    captura = captura_video(fps=16.0)
    captura.record(save_video=False, pipe_filtrado=deteccionRastreo)

def apartado341():
    captura = captura_video(fps=16.0)
    #pipelines = TrackerPipeline()
    #captura.record(save_video=False, pipe_filtrado=pipelines.OpticalFlowTracker)
    pipelines = OpticalFlowTracker()
    captura.record(save_video=False, pipe_filtrado=pipelines.process)

def apartado342():
    captura = captura_video(fps=16.0)
    pipelines = TrackerPipeline()
    captura.record(save_video=False, pipe_filtrado=pipelines.YOLOtracker)
    
    

if __name__ == "__main__":
    while True:
        apartado = input("\n\nSelecciones apartado a ejecutar: ")
        if apartado == "32":
            apartado32()
        elif apartado == "331":
            apartado331()
        elif apartado == "332":
            apartado332()
        elif apartado == "340":
            apartado340()
        elif apartado == "341":
            apartado341()
        elif apartado == "342":
            apartado342()
        elif apartado in ["exit", "e"]:
            break
        else: 
            print("Opción no válida. Intente de nuevo. Si desea salir, escriba 'exit'.")
import cv2
import time

class captura_video:
    def __init__(self, fps=60.0, camera=cv2.VideoCapture(0), video_path="DSP_Deteccion_Rastreo_Personas/videos/"):
        self.fps = fps
        self.cam = camera
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Define the codec and create VideoWriter object
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_path = video_path

    def record(self, save_video=False):
        if save_video:
            out = cv2.VideoWriter(self.video_path+'output.mp4', self.fourcc, fps=self.fps, frameSize=(self.frame_width, self.frame_height))
        delayed = 0
        start_recording = time.time()
        while True:
            start = time.time()
            ret, frame = self.cam.read()

            if save_video:
                # Write the frame to the output file
                out.write(frame)

            # Display the captured frame
            cv2.imshow('Camera', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord('q'):
                break

            # Wait for a short period to control the frame rate
            pipe_time = time.time() - start
            wait = 1/self.fps - pipe_time if (1/self.fps - pipe_time) > 0 else 0
            delayed += 1 if wait == 0 else 0
            time.sleep(wait)

        # Release the capture and writer objects
        self.cam.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()

        # Print recording summary
        record_time = time.time() - start_recording
        print(f"Frames delayed due to processing time: {delayed*100/(record_time*self.fps):.2f}%")

if __name__ == "__main__":
    captura = captura_video(fps=30.0)
    captura.record(save_video=True)
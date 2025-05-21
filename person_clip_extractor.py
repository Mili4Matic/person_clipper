import argparse
import json
from pathlib import Path
import cv2
from ultralytics import YOLO


def _ms(frame_index: int, fps: float) -> int:
    """
    Convert frame index to milliseconds.
    Args:
        frame_index (int): Frame index.
        fps (float): Frames per second.
    Returns:
        int: Time in milliseconds.
    """
    return int((frame_index / fps) * 1000)



def extract_single_person_clip_segment(
        video_path: str | Path,
        output_json: str | Path,
        model_weight: str | Path = "yolov8n.pt",
        conf_thresh: float = 0.3,
):
    
    video_path = Path(video_path)
    if output_json is not None:
        output_json = path(output_json)

    # Load the YOLO model
    model = YOLO(model_weight)

    # Load the video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) # or 30.0 si me da fallo le metemos este valor como default
    cap.release()


    segments: list[dict] = []
    active_id: int | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    frame_index: int = 0


    for result in model.track(
        source = str(video_path),
        stream = True,
        classes = [0],  # Person class
        conf = conf_thresh,
        persist = True,
    ):
        # Sacamos los ids de las personas que aparecen en el frame
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().tolist()
            unique_ids = list(set(ids))
        else:
            unique_ids = []


        # Logica de control de los ids

        if len(unique_ids) == 1:
        # Hay un solo id en el frame
            pid = unique_ids[0]
            
            if active_id is None:
                # Si no hay id activo, lo activamos
                active_id = pid
                start_frame = frame_index
            if pid == active_id:
                # Si el id activo es el mismo que el de la persona, lo guardamos
                end_frame = frame_index

            else:
                segments.append({
                    "person_id": active_id,
                    "start_ms": _ms(start_frame, fps),
                    "end_ms": _ms(end_frame, fps),
                })

                active_id = None
                start_frame = None
                end_frame = None

            frame_index += 1


        # Si hay un id activo y un frame de fin, lo guardamos
        if active_id is not None and end_frame is not None:
            segments.append({
                "person_id": active_id,
                "start_ms": _ms(start_frame, fps),
                "end_ms": _ms(end_frame, fps),
            })


        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as fh:
                json.dump(segments, fh, indent=2, ensure_ascii=False) 
                print(f"Saved segments to {output_json}")

        return segments
    


def _cli():
    parser = argparse.ArgumentParser(
        description="Extract single person clip segments from a video using YOLOv8."
    )
    parser.add_argument("--input", required=True, help ="Ruta al video de entrada")
    parser.add_argument("--output", required=True, help="Ruta al archivo JSON de salida de los timestamps")
    parser.add_argument("--model", default="yolov8n.pt", help="Pesos del modelo YOLOv8n, se puede utilizar las versions s, m etc")
    parser.add_argument("--conf", type=float, default=0.3, help="Umbral de confianza para detectar una persona")
    args = parser.parse_args()
    clips = extract_single_person_clip_segment(
        video_path=args.input,
        output_json=args.output,
        model_weight=args.model,
        conf_thresh=args.conf,
    )
    print(f"Se han detectado y extraido {len(clips)} clips from {args.input} y guardado en {args.output}")


if __name__ == "__main__":
    _cli()
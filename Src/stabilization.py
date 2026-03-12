import cv2
import os
from stabilo import Stabilizer


def stabilize_video(
    video_path: str,
    output_path: str,
    use_stabilization: bool = True,
    reset_ref_every: int = 0,
    codec: str = "mp4v"
):
    """
    Stabilize a video using the same approach as in your original script.

    Parameters
    ----------
    video_path : str
        Path to the input video.
    output_path : str
        Path to the output stabilized video.
    use_stabilization : bool
        If False, the video is copied without stabilization.
    reset_ref_every : int
        If > 0, resets the stabilizer reference frame every N frames.
        If 0, keeps the first frame as the reference for the whole video.
    codec : str
        FourCC codec for output video. Default is "mp4v".
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError("Could not read the first frame of the video.")

    h, w = first_frame.shape[:2]

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {output_path}")

    stabilizer = None
    if use_stabilization:
        stabilizer = Stabilizer()
        stabilizer.set_ref_frame(first_frame.copy())

    frame_idx = 0
    processed = 0

    while True:
        if frame_idx == 0:
            frame = first_frame
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

        if use_stabilization and stabilizer is not None:
            try:
                stabilizer.stabilize(frame)
                stabilized = stabilizer.warp_cur_frame()
                output_frame = stabilized if stabilized is not None else frame
            except Exception as e:
                print(f"[Warning] Stabilization failed at frame {frame_idx}: {e}")
                output_frame = frame
        else:
            output_frame = frame

        writer.write(output_frame)

        # Same logic as your original script:
        # optionally reset the reference every N frames
        if (
            use_stabilization
            and stabilizer is not None
            and reset_ref_every > 0
            and frame_idx % reset_ref_every == 0
            and frame_idx != 0
        ):
            stabilizer.set_ref_frame(output_frame.copy())

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}")

        processed += 1
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Done. Processed {processed} frames.")
    print(f"Stabilized video saved to: {output_path}")


if __name__ == "__main__":
    # ----------------------------
    # PUT YOUR PATHS HERE
    # ----------------------------
    video_path = r"C:/Users/makss/Git/Galatsi-Semester-Project/Dataset/Galatsi_Data_Semester_Project_archive/DJI_0314_D2_S4_S1.mp4"
    output_path = r"C:/Users/makss/Git/Galatsi-Semester-Project/Dataset/Galatsi_Data_Semester_Project_stabilized/DJI_0314_D2_S4_S1.mp4"

    stabilize_video(
        video_path=video_path,
        output_path=output_path,
        use_stabilization=True,
        reset_ref_every=0  # 0 = keep first frame as reference for all video
    )
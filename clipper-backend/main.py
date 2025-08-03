import pickle
import glob
import cv2
import ffmpegcv
import numpy as np
from tqdm import tqdm
import modal
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import boto3
import pathlib
import uuid
import whisperx
import subprocess
import time
import json
from google import genai
import shutil
import pysubs2


class ProcessVideoRequest(BaseModel):
    s3_key: str


image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"])
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["mkdir -p /usr/share/fonts/truetype/custom",
                   "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
                   "fc-cache -f -v"])
    .add_local_dir("asd", "/asd", copy=True))

app = modal.App("ai-video-clipper", image=image)

volume = modal.Volume.from_name(
    "ai-video-clipper-model-cache",
    create_if_missing=True
)

mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()

def create_vertical_video(tracks, scores, pyframes_path, pyavi_path, audio_path, output_path, frame_rate=25):
    # Create a vertical video from the given tracks and scores  
    target_width = 1080
    target_height = 1920
    
    flist = glob.glob(os.path.join(pyframes_path, "*.jpg"))
    flist.sort()

    faces = [[] for _ in range(len(flist))]

    for track_idx, track in enumerate(tracks):
        score_array = scores[track_idx]
        for frame_idx, frame in enumerate(track["track"]["frame"].tolist()):
            slice_start = max(frame_idx - 30, 0)
            slice_end = min(frame_idx + 30, len(score_array))
            score_slice = score_array[slice_start:slice_end]
            avg_score = float(np.mean(score_slice) if len(score_slice) > 0 else 0)

            faces[frame].append(
                {'track': track_idx,
                 'score': avg_score,
                 's': track["proc_track"]["s"][frame_idx],
                 'x': track["proc_track"]["x"][frame_idx],
                 'y': track["proc_track"]["y"][frame_idx]})

    temporary_video_path = os.path.join(pyavi_path, "temporary_vertical_video.mp4")

    vout = None
    for frame_idx, fname in tqdm(enumerate(flist), total=len(flist), desc="Creating vertical video"):
        img = cv2.imread(fname)
        if img is None:
            continue

        current_faces = faces[frame_idx]

        max_score_face = max(current_faces, key=lambda face: face['score']) if current_faces else None

        if max_score_face and max_score_face['score'] < 0:
            max_score_face = None

        if vout is None:
            vout = ffmpegcv.VideoWriterNV(
                file=temporary_video_path,
                codec=None,
                fps=frame_rate,
                resize=(target_width, target_height))
            
        if max_score_face:
            mode = "crop"
        else:
            mode = "resize"

        if mode == "resize":
            scale = target_width / img.shape[1]
            resized_height = int(img.shape[0] * scale)
            resized_image = cv2.resize(img, (target_width, resized_height), interpolation=cv2.INTER_AREA)

            scale_for_bg = max(target_width / img.shape[1], target_height / img.shape[0])
            bg_width = int(img.shape[1] * scale_for_bg)
            bg_height = int(img.shape[0] * scale_for_bg)

            blurred_background = cv2.resize(img, (bg_width, bg_height))
            blurred_background = cv2.GaussianBlur(blurred_background, (121, 121), 0) # change these for fiddling with the blur

            crop_x = (bg_width - target_width) // 2
            crop_y = (bg_height - target_height) // 2 # floored division

            blurred_background = blurred_background[crop_y:crop_y + target_height, crop_x:crop_x + target_width]

            center_y = (target_height - resized_height) // 2
            blurred_background[center_y:center_y + resized_height, :] = resized_image

            # Write the vertical image to the video
            vout.write(blurred_background)
        elif mode == "crop":
            scale = target_height / img.shape[0]
            resized_image = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame_width = resized_image.shape[1]

            center_x = int(max_score_face["x"] * scale if max_score_face else frame_width // 2)
            top_x = max(
                min(center_x - target_width // 2, frame_width - target_width),
                0)
            
            image_cropped = resized_image[0:target_height, top_x:top_x + target_width]

            vout.write(image_cropped)

    if vout:
        vout.release()

    ffmpeg_command = (f"ffmpeg -y -i {temporary_video_path} -i {audio_path} "
                      f"-c:v h264 -preset fast -crf 23 -c:a aac -b:a 128k "
                      f"{output_path}")
    subprocess.run(
        ffmpeg_command,
        shell=True,
        text=True,
        check=True
    )

def create_subtitles_with_ffmpeg(
        transcript_segments: list,
        clip_start: float,
        clip_end: float,
        clip_video_path: str,
        output_path: str,
        max_words: int = 5
):
    temp_dir = os.path.dirname(output_path)
    subtitle_path = os.path.join(temp_dir, "subtitles.ass")

    # clip_segments = [segment for segment in transcript_segments
    #                  if segment['start'] is not None and
    #                  segment['end'] is not None and
    #                  segment['end'] > clip_start and
    #                  segment['start'] < clip_end]
    
    clip_segments = [segment for segment in transcript_segments
                     if segment.get("start") is not None
                     and segment.get("end") is not None
                     and segment.get("end") > clip_start
                     and segment.get("start") < clip_end
                     ]
    
    # Group words into chunks of the max words
    subtitles = []
    current_words = []
    current_start = None
    current_end = None

    for segment in clip_segments:
        # word = segment['word'].strip()
        # seg_start = segment['start']
        # seg_end = segment['end']

        word = segment.get("word", "").strip()
        seg_start = segment.get("start")
        seg_end = segment.get("end")

        if not word or seg_start is None or seg_end is None:
            continue

        start_rel = max(0.0, seg_start - clip_start)
        # end_rel = min(clip_end - clip_start, seg_end - clip_start)
        end_rel = max(0.0, seg_end - clip_start)

        # print(start_rel, end_rel, word)

        if end_rel <= 0:
            continue

        if not current_words: # For the very first word of the subtitle
            current_start = start_rel
            current_end = end_rel
            current_words = [word]
        elif len(current_words) >= max_words: # If we reach the max words limit, save the current subtitle
            # subtitles.append({
            #     'start': current_start,
            #     'end': current_end,
            #     'text': ' '.join(current_words)
            # })
            subtitles.append(
                (current_start, current_end, ' '.join(current_words)))
            current_start = start_rel
            current_end = end_rel
            current_words = [word]
        else: # If we are still within the max words limit, just append the word
            current_words.append(word)
            current_end = end_rel

        # print(current_words, current_start, current_end)

    if current_words:
        # subtitles.append({
        #     'start': current_start,
        #     'end': current_end,
        #     'text': ' '.join(current_words)
        # })
        subtitles.append(
            (current_start, current_end, ' '.join(current_words)))

    # print(subtitles)
    subs = pysubs2.SSAFile()

    subs.info["WrapStyle"] = 0
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920
    subs.info["ScriptType"] = "v4.00+"

    style_name = "Default"
    new_style = pysubs2.SSAStyle()
    new_style.fontname = "Anton"
    new_style.fontsize = 140
    new_style.primarycolor = pysubs2.Color(255, 255, 255)
    new_style.outline = 2.0
    new_style.shadow = 2.0
    new_style.shadowcolor = pysubs2.Color(0, 0, 0, 128)
    new_style.alignment = 2
    new_style.marginl = 50
    new_style.marginr = 50
    new_style.marginv = 400
    new_style.spacing = 0.0

    subs.styles[style_name] = new_style

    for i, (start, end, text) in enumerate(subtitles):
        start_time = pysubs2.make_time(s=start)
        end_time = pysubs2.make_time(s=end)
        line = pysubs2.SSAEvent(
            start=start_time,
            end=end_time,
            text=text,
            style=style_name,
        )
        subs.events.append(line)

    # subs.save(subtitle_path, encoding="utf-8")
    subs.save(subtitle_path)

    # Write the subtitles to a file
    # with open(subtitle_path, "w") as f:
    #     for subtitle in subtitles:
    #         f.write(f"{subtitle['start']} --> {subtitle['end']}\n")
    #         f.write(f"{subtitle['text']}\n\n")

    # Create the final video with subtitles
    ffmpeg_command = (f"ffmpeg -i {clip_video_path} -vf \"ass={subtitle_path}\" "
                      f"-c:v h264 -preset fast -crf 23 {output_path}")
    subprocess.run(
        ffmpeg_command,
        shell=True,
        check=True
    )

def process_clip(
        base_dir: str, # the base directory to work in
        video_path: str, # the path to the video file
        s3_key: str, # the s3 key to save the clip to
        start_time: float, # the start time of the viral clip
        end_time: float, # the end time of the viral clip
        index: int, # the index of the viral clip
        transcript_segments: list): # the transcript segments of the entire video
    print(f"Processing video: {video_path}")
    print(f"Processing clip from {start_time} to {end_time}")
    clip_name = f"clip_{index}"
    # uuid/originalVideo.mp4 to uuid/clip_index.mp4
    s3_key_dir = os.path.dirname(s3_key)
    output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"

    print(f"Output S3 key: {output_s3_key}")

    clip_dir = base_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    # Path 1 - Segment Path: Original Clip from Start To End
    clip_segment_path = clip_dir / f"{clip_name}_segment.mp4"
    # Path 2 - Vertical Clip Path: Original Clip from Start To End, but in vertical format
    vertical_mp4_path = clip_dir / "pyavi" / f"{clip_name}_vertical.mp4"
    # Path 3 - Subtitle Path: Original Clip from Start To End, but with subtitles
    subtitle_mp4_path = clip_dir / "pyavi" / f"{clip_name}_subtitle.mp4"

    # We need to create three folder paths because of the way LR-ASD works:
    # https://github.com/Junhua-Liao/LR-ASD?tab=readme-ov-file#evaluate-on-columbia-asd-dataset
    # https://www.youtube.com/watch?v=a7rLzt2ZWk0
    (clip_dir / "pywork").mkdir(exist_ok=True)
    pyframes_path = clip_dir / "pyframes"
    pyavi_path = clip_dir / "pyavi"
    audio_path = clip_dir / "pyavi" / "audio.wav"
    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)

    duration = end_time - start_time
    cut_command = (f"ffmpeg -i {video_path} -ss {start_time} -t {duration} "
                   f"{clip_segment_path}")
    subprocess.run(
        cut_command,
        shell=True,
        capture_output=True,
        text=True,
        check=True
    )

    extract_audio_command = (f"ffmpeg -i {clip_segment_path} "
                             f"-vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}")
    subprocess.run(
        extract_audio_command,
        shell=True,
        capture_output=True,
        check=True
    )

    shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")

    columbia_asd_command = (f"python Columbia_test.py --videoName {clip_name} "
                            f"--videoFolder {str(base_dir)} "
                            f"--pretrainModel weight/finetuning_TalkSet.model")
    
    print(f"Running Columbia ASD command: {columbia_asd_command}")
    columbia_start = time.time()
    subprocess.run(
        columbia_asd_command,
        cwd="/asd",
        shell=True
    )
    columbia_end = time.time()
    print(f"Columbia ASD command completed in {columbia_end - columbia_start:.2f} seconds")

    tracks_path = clip_dir / "pywork" / "tracks.pckl"
    scores_path = clip_dir / "pywork" / "scores.pckl"
    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError(
            "tracks.pckl or scores.pckl not found. Skipping clip processing."
        )
    
    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)

    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    print(f"Creating vertical video from {tracks_path} and {scores_path}")
    cvv_start = time.time()
    create_vertical_video(
        tracks, scores, pyframes_path, pyavi_path, audio_path, vertical_mp4_path)
    cvv_end = time.time()
    print(f"Vertical video created in {cvv_end - cvv_start:.2f} seconds")

    print(f"Creating subtitle video from {vertical_mp4_path} to {output_s3_key}")
    svv_start = time.time()
    create_subtitles_with_ffmpeg(
        transcript_segments,
        start_time,
        end_time,
        vertical_mp4_path,
        subtitle_mp4_path,
        max_words=5)
    svv_end = time.time()
    print(f"Subtitle video created in {svv_end - svv_start:.2f} seconds")

    print(f"Saving final video with subtitles to {output_s3_key}")
    s3_client = boto3.client("s3")
    s3_client.upload_file(
        subtitle_mp4_path,
        "ai-podcast-clipper-raju",
        output_s3_key
    )


@app.cls(
    gpu="L40S",
    timeout=1200,
    retries=3,
    scaledown_window=20,
    volumes={mount_path: volume},
    secrets=[modal.Secret.from_name("ai-podcast-clipper-secret")],
)
class AiPodcastClipper:
    @modal.enter()
    def load_model(self):
        print("Loading models")

        # Load the WhisperX model to be used for transcription
        self.whisperx_model = whisperx.load_model(
            "large-v2", device="cuda", compute_type="float16")

        # Load the alignment model to align transcription and audio
        self.alignment_model, self.metadata = whisperx.load_align_model(
            language_code="en", device="cuda")

        print("Transcription and alignment models loaded...")

        print("Loading Gemini client...")

        # Load the Gemini client to identify viral moments
        # Make sure to set the GEMINI_API_KEY environment variable
        self.gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        print("Gemini client loaded...")

    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        # Audio extraction section
        audio_path = base_dir / "audio.wav"

        # Extract audio from video, and save it to the location 'audio_path'
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"

        # As we need to run this command in the terminal, we need to use the shell=True
        subprocess.run(
            extract_cmd,
            shell=True,
            check=True,
            capture_output=True)
        # Audio extraction section completed

        print("Starting transcription with WhisperX...")
        start_time = time.time()

        # Load the audio file into WhisperX
        audio = whisperx.load_audio(str(audio_path))

        # Transcribe the audio file by calling the transcribe method of the whisperx model
        # batch_size is the number of audio chunks to process at once
        # We can also set the language code to "en" to transcribe the audio in English
        # The batch_size is tested for the GPU used, which is L40S in this case
        result = self.whisperx_model.transcribe(audio, batch_size=16)

        # Align the transcription results with the audio file
        result = whisperx.align(
            result["segments"],
            self.alignment_model,
            self.metadata,
            audio,
            device="cuda",
            return_char_alignments=False,
        )

        duration = time.time() - start_time
        print(
            f"Transcription and alignment completed in {duration:.2f} seconds")

        # print(json.dumps(result, indent=2))

        # Create a list to store the segments
        segments = []

        # Check if the word_segments are present in the result
        if "word_segments" in result:
            # Iterate over the word_segments
            for word_segment in result["word_segments"]:
                # Append the word_segment to the segments list
                # The word_segment contains the start and end time of the word, and the word itself
                segments.append({
                    "start": word_segment["start"],
                    "end": word_segment["end"],
                    # The word_segment contains the word itself
                    "word": word_segment["word"]
                })

        # print(json.dumps(segments, indent=2))

        # Return the segments as a JSON string
        return json.dumps(segments)

    def identify_viral_moments(self, transcript_segments: dict):
        print("Starting identification of viral moments using Gemini 2.5 Flash...")
        start_time = time.time()
        response = self.gemini_client.models.generate_content(model="gemini-2.5-flash", contents="""
    This is a podcast video transcript consisting of word, along with each words's start and end time. I am looking to create clips between a minimum of 30 and maximum of 60 seconds long. The clip should never exceed 60 seconds.

    Your task is to find and extract stories, or question and their corresponding answers from the transcript.
    Each clip should begin with the question and conclude with the answer.
    It is acceptable for the clip to include a few additional sentences before a question if it aids in contextualizing the question.

    Please adhere to the following rules:
    - Ensure that clips do not overlap with one another.
    - Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
    - Only use the start and end timestamps provided in the input. modifying timestamps is not allowed.
    - Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{"start": seconds, "end": seconds}, ...clip2, clip3]. The output should always be readable by the python json.loads function.
    - Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.

    Avoid including:
    - Moments of greeting, thanking, or saying goodbye.
    - Non-question and answer interactions.

    If there are no valid clips to extract, the output should be an empty list [], in JSON format. Also readable by json.loads() in Python.

    The transcript is as follows:\n\n""" + str(transcript_segments))
        duration = time.time() - start_time
        print(f"Viral moments identification completed in {duration:.2f} seconds")
        # print(f"Identified moments response: ${response.text}")
        return response.text

        # # Create a list to store the viral moments
        # viral_moments = []

        # # Iterate over the transcript segments
        # for segment in transcript_segments:
        #     # Append the segment to the viral moments list
        #     viral_moments.append(segment)

        # # Return the viral moments
        # return viral_moments

    @modal.fastapi_endpoint(method="POST")
    def process_video(
            self,
            request: ProcessVideoRequest,
            token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        
        # Check if the token is correct
        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"})
        
        # Check if the request body contains the s3_key
        if not request.s3_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="s3_key is required in the request body")
        # Check if the s3_key is a valid string
        if not isinstance(request.s3_key, str) or not request.s3_key.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="s3_key must be a non-empty string")
        
        # Get the s3_key from the request
        s3_key = request.s3_key

        # Create a unique run ID
        run_id = str(uuid.uuid4())

        # Create a base directory folder to store the video and the transcript
        base_dir = pathlib.Path("/tmp") / run_id

        # Create the base directory
        base_dir.mkdir(parents=True, exist_ok=True)

        # Download the video file from S3
        video_path = base_dir / "input.mp4"

        # Create an S3 client
        s3_client = boto3.client("s3")

        # Download the video file from S3
        s3_client.download_file(
            "ai-podcast-clipper-raju",
            s3_key,
            str(video_path))

        # 1. Transcription
        transcript_segments_json = self.transcribe_video(base_dir, video_path)
        transcript_segments = json.loads(transcript_segments_json)
        # The JSON returned contains the start time and end time and word of each segment

        # 2. Identify the viral moments for clips using Gemini
        print("Identifying viral moments...")
        identified_viral_moments = self.identify_viral_moments(
            transcript_segments)
        print("Viral moments identified...")

        clean_json_string = identified_viral_moments.strip()

        if clean_json_string.startswith("```json"):
            clean_json_string = clean_json_string[(
                len("```json")):].strip()

        if clean_json_string.endswith("```"):
            clean_json_string = clean_json_string[:-3].strip()

        clip_moments = json.loads(clean_json_string)
        if not clip_moments or not isinstance(clip_moments, list):
            print("Error: No valid clips identified or the format is incorrect.")
            clip_moments = []

        print(clip_moments)

        # 3. Process the clips
        for index, moment in enumerate(clip_moments[:1]): # Limit to first 2 clips for testing
            if "start" in moment and "end" in moment:
                print(f"Creating clip {index + 1} from {moment['start']} to {moment['end']}")
                process_clip(base_dir, # the directory to work in
                             video_path, # the path to the video file
                             s3_key, # the s3 key to save the clip to
                             moment['start'], # the start time of the viral clip
                             moment['end'], # the end time of the viral clip
                             index, # the index of the viral clip
                             transcript_segments) # the transcript segments of the entire video

        print("All clips processed successfully.")

        if base_dir.exists():
            print(f"Cleaning temp directory after processing is completed... {base_dir}...")
            shutil.rmtree(base_dir, ignore_errors=True)


@app.local_entrypoint()
def main():
    import requests

    ai_podcast_clipper = AiPodcastClipper()

    url = ai_podcast_clipper.process_video.get_web_url()

    payload = {
        "s3_key": "test1/vod1.mp4"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123123"
    }

    response = requests.post(url, json=payload, headers=headers)

    response.raise_for_status()
    result = response.json()
    print(result)

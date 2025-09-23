# ===============================================================================
#                            IMPORTS
# ===============================================================================
import os
import json
import subprocess
import string
import glob
import shutil

# ===============================================================================
#                            CONFIGURATION
# ===============================================================================
# --- Core Settings ---
LANGUAGE = "de"  # "en" or "de"
INPUT_TYPE = "spoken" # "vocals" or "spoken"

# --- Alignment Settings ---
# "word": Standard word-by-word alignment.
# "word_separated": Alternative word-by-word alignment. May improve quality.
# "segment_manual": For phrase-by-phrase highlighting. Use '|' in the .txt file to separate phrases.
# "segment_auto": Let the aligner automatically create segments based on punctuation.
ALIGNMENT_TYPE = "segment_manual"

# --- Video Settings ---
# Set to a color name (e.g., "black", "blue") or "image" to use an image file from your WORK_DIR.
VIDEO_BACKGROUND = "#7e727bff"
VIDEO_RESOLUTION = "1280x720"

# --- File Paths ---
NEMO_DIR_PATH = "/workspace/NeMo"
WORK_DIR = "/workspace/NeMo/tools/nfa_wrdlvl_pipeline/WORK_DIR"
# Directory for input media files for batch processing.
INPUT_DIR = os.path.join(WORK_DIR, "input")


# --- Model Selection ---
# This will be automatically selected based on LANGUAGE and INPUT_TYPE
# but you can override it here if you want.
PRETRAINED_MODEL = "" # e.g., "stt_en_fastconformer_hybrid_large_pc, stt_en_conformer_ctc_xlarge

# --- Subtitle Styling ---
ASS_FONTSIZE = "22"
VERTICAL_ALIGNMENT = "bottom"
TEXT_ALREADY_SPOKEN_RGB = "#4D4D4D"
TEXT_BEING_SPOKEN_RGB = "#1100FF"
TEXT_NOT_YET_SPOKEN_RGB = "#FFFFFF"

# ===============================================================================
#                            CORE FUNCTIONS
# ===============================================================================
def setup_environment(work_dir, nemo_dir):
    """Sets up the environment for the pipeline."""
    os.makedirs(work_dir, exist_ok=True)
    print(f"Project files will be stored in: {work_dir}")
    
    if not os.path.exists(nemo_dir):
        raise FileNotFoundError(f"Could not find NeMo directory at {nemo_dir}")
    print(f"NeMo toolkit found at: {nemo_dir}")
    

def hex_to_rgb(color):
    if isinstance(color, list):
        return color  # It's already an RGB list
    if isinstance(color, str):
        hex_color = color.lstrip('#')
        return list(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return color # Return as is if it's not a list or a string


def select_model(language, input_type, override_model):
    """Selects the pretrained model based on language and input type."""
    
    model_name = ""
    if override_model:
        model_name = override_model
    else:
        model_map = {
            "en": {
                "spoken": "stt_en_fastconformer_hybrid_large_pc",
                "vocals": "stt_en_fastconformer_hybrid_large_pc",
            },
            "de": {
                "spoken": "stt_de_fastconformer_hybrid_large_pc",
                "vocals": "stt_de_fastconformer_hybrid_large_pc",
            }
        }
        try:
            model_name = model_map[language][input_type]
        except KeyError:
            raise ValueError(f"No model found for language='{language}' and input_type='{input_type}'. You can set a PRETRAINED_MODEL manually in the configuration.")

    if "transducer" in model_name:
        raise ValueError(
            f"\n\nThe selected model '{model_name}' is a 'Transducer' model.\n"
            f"The NeMo Forced Aligner tool does not support Transducer models.\n"
            f"Please choose a CTC-based model (e.g., models with 'conformer_ctc', 'citrinet', or 'fastconformer_hybrid' in their name).\n"
        )
    
    print(f"Selected model: {model_name}")
    return model_name

def prepare_media_files(work_dir, video_background, video_resolution, input_media_path,
transcription_only=False):
    """Prepares a single media file for processing."""
    print(f"Starting media preparation for: {input_media_path}")
    output_basename = os.path.splitext(os.path.basename(input_media_path))[0]
    mono_wav_path = os.path.join(work_dir, f"{output_basename}_temp16.wav")

    # Always create the mono WAV file for the aligner.
    print(f"Creating 16-bit mono WAV: '{mono_wav_path}'")
    subprocess.run(
        ["ffmpeg", "-i", input_media_path, "-acodec", "pcm_s16le", "-ac", "1", "-n", mono_wav_path], # -n don't overwrite 16BIT WAV
        check=True, capture_output=True, text=True
    )

    if transcription_only:
        # For transcription, we only need the WAV. The final video path is irrelevant.
        print("Transcription mode: Skipping video preparation.")
        return input_media_path, output_basename, mono_wav_path

    # Determine the path of the video to be used for subtitling.
    video_extensions = ['mp4', 'mov', 'avi', 'mkv']
    _, ext = os.path.splitext(input_media_path)
    is_video = ext.lower().lstrip('.') in video_extensions

    if is_video:
        # If the input is a video, we will use it directly for subtitling. No re-encoding.
        print("Input is a video. It will be used directly for the final subtitled file.")
        video_path_for_subtitling = input_media_path
    else:
        # If the input is audio, we must create a video with a background.
        video_path_for_subtitling = os.path.join(work_dir, f"{output_basename}.mp4")
        if not os.path.exists(video_path_for_subtitling):
            print(f"Input is audio. Creating a background video at:\n{video_path_for_subtitling}")
            image_files = glob.glob(os.path.join(os.path.dirname(input_media_path),f"{output_basename}.*g"))                         + glob.glob(os.path.join(os.path.dirname(input_media_path),f"{output_basename}.*G"))

            if video_background == 'image' and image_files:
                image_path = image_files[0]
                print(f"Creating video from image '{image_path}'...")
                subprocess.run(["ffmpeg", "-loop", "1", "-framerate", "30", "-i",
                                            image_path, "-i", input_media_path, "-c:v", "libx264", "-preset", "p1", "-tune", "hq",
                                            "-cq", "19", "-c:a", "libmp3lame", "-b:a", "192k", "-vf", 
                                            f"scale={video_resolution}:force_original_aspect_ratio=decrease,pad={video_resolution}:-1:-1:color=black",
                                            "-shortest", "-n", video_path_for_subtitling], 
                                            check=True, capture_output=True, text=True)
            else:
                ffmpeg_color = f"0x{video_background.lstrip('#')}"
                print(f"Creating video with '{video_background}' background...")
                subprocess.run(["ffmpeg", "-f", "lavfi", "-i",
f"color=c={ffmpeg_color}:s={video_resolution}", "-i", input_media_path, "-c:v",
"libx264", "-c:a", "copy", "-shortest", "-n", video_path_for_subtitling], check=True,
capture_output=True, text=True)
        else:
            print(f"Background video already exists: {video_path_for_subtitling}")

    return video_path_for_subtitling, output_basename, mono_wav_path

def get_transcript(media_filepath):
    """Looks for a .txt sidecar file for the given media file."""
    base, _ = os.path.splitext(media_filepath)
    transcript_path = base + ".txt"
    if os.path.exists(transcript_path):
        print(f"Found transcript file: {transcript_path}")
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def prepare_text(raw_text, alignment_type):
    separator = "⠀ "
    
    if alignment_type == "word":
        punc_to_remove = string.punctuation.replace('!', '').replace('?', '').replace('-', '')
        processed_text = raw_text.translate(str.maketrans('', '', punc_to_remove))
        words = processed_text.split()
        return " ".join(words)

    elif alignment_type == "word_separated":
        punc_to_remove = string.punctuation.replace('!', '').replace('?', '').replace('-', '')
        processed_text = raw_text.translate(str.maketrans('', '', punc_to_remove))
        words = processed_text.split()
        return separator.join(words)
        
    elif alignment_type == "segment_manual":
        processed_text = raw_text.replace('|', separator)
        parts = processed_text.split(separator)
        cleaned_parts = [" ".join(part.split()) for part in parts]
        return separator.join(cleaned_parts)

    elif alignment_type == "segment_auto":
        return raw_text.replace('|', '').replace('⠀', '')
        
    else:
        raise ValueError(f"Unknown alignment_type: {alignment_type}")

def create_manifest(work_dir, text, mono_wav_path, output_basename):
    manifest_filepath = os.path.join(work_dir, f"{output_basename}_manifest.json")
    manifest_data = {
        "audio_filepath": mono_wav_path,
        "text": text if text is not None else ""
    }
    with open(manifest_filepath, 'w', encoding="utf-8") as f:
        line = json.dumps(manifest_data)
        f.write(line + "\n")
    print(f"Manifest file created at: {manifest_filepath}")
    return manifest_filepath

def run_forced_alignment(work_dir, nemo_dir, pretrained_model, use_pred_text, alignment_type,
vertical_alignment, ass_fontsize, text_already_spoken_rgb, text_being_spoken_rgb, text_not_yet_spoken_rgb, manifest_filepath):
    output_dir = os.path.join(work_dir, "nfa_output")

    text_already_spoken_rgb_list = hex_to_rgb(text_already_spoken_rgb)
    text_being_spoken_rgb_list = hex_to_rgb(text_being_spoken_rgb)
    text_not_yet_spoken_rgb_list = hex_to_rgb(text_not_yet_spoken_rgb)

    command = ["python", f"{nemo_dir}/tools/nemo_forced_aligner/align.py",
                f"pretrained_name={pretrained_model}",
                f"manifest_filepath={manifest_filepath}",
                f"output_dir={output_dir}",
                f"align_using_pred_text={use_pred_text}",
                f"ass_file_config.fontsize={ass_fontsize}", 
                f"ass_file_config.vertical_alignment={vertical_alignment}",
                f"ass_file_config.text_already_spoken_rgb={text_already_spoken_rgb_list}",
                f"ass_file_config.text_being_spoken_rgb={text_being_spoken_rgb_list}",
                f"ass_file_config.text_not_yet_spoken_rgb={text_not_yet_spoken_rgb_list}"]

    if alignment_type == "word":
        command.append(f"ass_file_config.resegment_text_word_by_word=true")
    elif alignment_type in ["word_separated", "segment_manual"]:
        command.append(f"additional_segment_grouping_separator=['⠀']")
    elif alignment_type == "segment_auto":
        # Only use line splitting as a fallback if we are predicting text AND the model is not punctuation-capable.
        if use_pred_text and "_pc" not in pretrained_model:
            print("Model is not punctuation-capable and no transcript is provided. Using line splitting for auto segmentation.")
            command.extend([
                f"ass_file_config.resegment_text_to_fill_space=true",
                f"ass_file_config.max_lines_per_segment=2"
            ])
        else:
            # If a transcript is provided, or if the model is punctuation-capable, use punctuation for segmentation.
            print("Using punctuation for auto segmentation.")
            command.append(f"additional_segment_grouping_separator=['.', ',', ':', '?', '!', '...']")

    print("Running NeMo Forced Aligner...")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("NeMo Forced Aligner finished successfully.")
    except subprocess.CalledProcessError as e:
        print("NeMo Forced Aligner failed.")
        print("Return code:", e.returncode)
        print("Stderr:", e.stderr)
        raise e

def save_predicted_text(work_dir, output_basename, input_dir):
    """Finds the predicted CTM file and saves the text to a sidecar .txt file."""
    ctm_path = os.path.join(work_dir, "nfa_output", "ctm", "words", f"{output_basename}.ctm")
    sidecar_txt_path = os.path.join(input_dir, f"{output_basename}.txt")

    if not os.path.exists(ctm_path):
        print(f"Warning: Could not find CTM file at {ctm_path} to save predicted text.")
        return

    words = []
    with open(ctm_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                words.append(parts[4])
    
    predicted_text = " ".join(words)
    
    with open(sidecar_txt_path, 'w', encoding='utf-8') as f:
        f.write(predicted_text)
    
    print(f"Saved predicted transcript to: {sidecar_txt_path}")

def rename_alignment_outputs(work_dir, output_basename, mono_wav_path):
    """Renames the output files from the NeMo Forced Aligner to match the original media name."""
    print("Renaming alignment output files...")
    nfa_output_dir = os.path.join(work_dir, "nfa_output")
    original_stem = os.path.splitext(os.path.basename(mono_wav_path))[0]

    for dirpath, _, filenames in os.walk(nfa_output_dir):
        for filename in filenames:
            if filename.startswith(original_stem):
                old_path = os.path.join(dirpath, filename)
                new_filename = filename.replace(original_stem, output_basename)
                new_path = os.path.join(dirpath, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed '{old_path}' to '{new_path}'")
    
def create_srt_file(work_dir, mp4_path):
    input_basename = os.path.splitext(os.path.basename(mp4_path))[0]
    ass_word_path = os.path.join(work_dir, "nfa_output", "ass", "words", f"{input_basename}.ass")
    srt_path = os.path.join(work_dir, f"{input_basename}.srt")

    if not os.path.exists(ass_word_path):
        print("No word-level ASS file found to create SRT from. Skipping this step.")
        return

    print("Creating SRT file...")
    subprocess.run(["ffmpeg", "-y", "-i", ass_word_path, "-c:s", "text", srt_path], check=True, capture_output=True, text=True)
    print(f"SRT file created at: {srt_path}")
      
def generate_subtitled_video(work_dir, video_path_for_subtitling):
    """
    Merges word and token subtitles into a single MKV file without re-encoding.
    """
    # Use the original filename for the output, but with a new extension and suffix
    base_name_with_ext = os.path.basename(video_path_for_subtitling)
    base_name_no_ext = os.path.splitext(base_name_with_ext)[0]
    output_mkv_path = os.path.join(work_dir, f"{base_name_no_ext}_subtitled.mkv")

    ass_token_path = os.path.join(work_dir, "nfa_output", "ass", "tokens", f"{base_name_no_ext}.ass")
    ass_word_path = os.path.join(work_dir, "nfa_output", "ass", "words", f"{base_name_no_ext}.ass")

    has_token_subs = os.path.exists(ass_token_path)
    has_word_subs = os.path.exists(ass_word_path)

    if not has_token_subs and not has_word_subs:
        print("No subtitle files found to generate video from. Skipping this step.")
        return

    print(f"Generating single MKV file with all available subtitles at: {output_mkv_path}")

    # Base command with the main video input
    command = ["ffmpeg", "-y", "-i", video_path_for_subtitling]
    
    # Dynamically add subtitle files as inputs
    subtitle_inputs = []
    if has_word_subs:
        subtitle_inputs.append(ass_word_path)
    if has_token_subs:
        subtitle_inputs.append(ass_token_path)
        
    for sub_path in subtitle_inputs:
        command.extend(["-i", sub_path])

    # Map video, audio, and then the new subtitle tracks
    command.extend([
        "-map", "0:v:0",      # Map video from the first input
        "-map", "0:a:0?",     # Map audio from the first input (optional, if it exists)
    ])

    for i in range(len(subtitle_inputs)):
        command.extend(["-map", str(i + 1)]) # Map the subtitle streams

    # Copy all streams without re-encoding and set metadata for the new subtitle tracks
    command.extend(["-c", "copy"])
    
    # Start subtitle metadata index at 0
    metadata_stream_index = 0 
    if has_word_subs:
        command.extend([f"-metadata:s:s:{metadata_stream_index}", "title=Words"])
        metadata_stream_index += 1
    if has_token_subs:
        command.extend([f"-metadata:s:s:{metadata_stream_index}", "title=Tokens"])

    command.append(output_mkv_path)

    try:
        print("Running ffmpeg to create final MKV...")
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully created final subtitled MKV: '{output_mkv_path}'")
    except subprocess.CalledProcessError as e:
        print("--- FFMPEG MERGE FAILED ---")
        print("Could not merge subtitles into the original video container.")
        print("The separate .ass and .srt subtitle files have still been generated.")
        print("Stderr:", e.stderr)

def move_files_to_done_directory(work_dir, output_basename):
    """Moves all generated files for a run into a 'done' subdirectory."""
    print(f"Moving files for '{output_basename}' to done directory...")
    done_dir = os.path.join(work_dir, "done", output_basename)
    os.makedirs(done_dir, exist_ok=True)

    files_in_work_dir = glob.glob(os.path.join(work_dir, f"{output_basename}*"))

    for src_path in files_in_work_dir:
        if os.path.isfile(src_path):
            file_name = os.path.basename(src_path)
            dst_path = os.path.join(done_dir, file_name)
            shutil.move(src_path, dst_path)
            print(f"Moved '{src_path}' to '{dst_path}'")

    nfa_output_dir = os.path.join(work_dir, "nfa_output")
    if os.path.exists(nfa_output_dir):
        for dirpath, _, filenames in os.walk(nfa_output_dir):
            for filename in filenames:
                if filename.startswith(output_basename):
                    src_path = os.path.join(dirpath, filename)
                    relative_path = os.path.relpath(dirpath, work_dir)
                    dst_dir_for_file = os.path.join(done_dir, relative_path)
                    os.makedirs(dst_dir_for_file, exist_ok=True)
                    dst_path = os.path.join(dst_dir_for_file, filename)
                    shutil.move(src_path, dst_path)
                    print(f"Moved '{src_path}' to '{dst_path}'")

# ===============================================================================
#                                   RUN
# ===============================================================================
def main():
    """Main function to run the batch processing."""
    # 1) Setup
    # Create input and done directories
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(WORK_DIR, "done"), exist_ok=True)
    setup_environment(WORK_DIR, NEMO_DIR_PATH)
    
    global PRETRAINED_MODEL
    PRETRAINED_MODEL = select_model(LANGUAGE, INPUT_TYPE, PRETRAINED_MODEL)

    # Find all media files in the input directory
    media_files = []
    media_extensions = ['mp4', 'mov', 'avi', 'mkv', 'mp3', 'aac', 'm4a', 'flac', 'wma', 'wav']
    for ext in media_extensions:
        media_files.extend(glob.glob(os.path.join(INPUT_DIR, f'*.{ext.lower()}')))
        media_files.extend(glob.glob(os.path.join(INPUT_DIR, f'*.{ext.upper()}')))

    print(f"Found {len(media_files)} media file(s) to process in '{INPUT_DIR}'")

    # 2) Loop through each media file
    for media_filepath in media_files:
        print(f"--- Starting processing for: {media_filepath} ---")
        
        try:
            # Check for transcript
            transcript = get_transcript(media_filepath)
            
            # Determine if we are in transcription or alignment mode
            if transcript:
                print("Found existing transcript. Running in alignment mode.")
                use_pred_text_for_this_file = False
                raw_text_for_this_file = transcript
                # Prepare all media files, including video
                output_video_path, output_basename, mono_wav_path = prepare_media_files(WORK_DIR, VIDEO_BACKGROUND, VIDEO_RESOLUTION, media_filepath, transcription_only=False)
            else:
                print("No transcript found. Running in transcription mode to generate one.")
                use_pred_text_for_this_file = True
                raw_text_for_this_file = "" # Pass empty string for prediction
                # Only prepare the WAV file, no need to create a video yet
                output_video_path, output_basename, mono_wav_path = prepare_media_files(WORK_DIR, VIDEO_BACKGROUND, VIDEO_RESOLUTION, media_filepath, transcription_only=True)

            # Prepare inputs for this specific file
            formatted_text = prepare_text(raw_text_for_this_file, ALIGNMENT_TYPE)
            manifest_filepath = create_manifest(WORK_DIR, formatted_text, mono_wav_path, output_basename)

            # Run alignment
            run_forced_alignment(
                WORK_DIR,
                NEMO_DIR_PATH,
                PRETRAINED_MODEL,
                use_pred_text_for_this_file,
                ALIGNMENT_TYPE,
                VERTICAL_ALIGNMENT,
                ASS_FONTSIZE,
                TEXT_ALREADY_SPOKEN_RGB,
                TEXT_BEING_SPOKEN_RGB,
                TEXT_NOT_YET_SPOKEN_RGB,
                manifest_filepath,
            )

            # Rename outputs
            rename_alignment_outputs(WORK_DIR, output_basename, mono_wav_path)

            # Post-processing
            if use_pred_text_for_this_file:
                # We were in transcription mode, so save the predicted text and stop for this file
                save_predicted_text(WORK_DIR, output_basename, INPUT_DIR)
                print(f"--- Finished transcription for: {media_filepath}. Please edit the generated .txt file and run again for alignment. ---")
                # No further processing for this file, continue to the next one
                continue
            else:
                # We were in alignment mode, so generate video and clean up
                create_srt_file(WORK_DIR, output_video_path)
                generate_subtitled_video(WORK_DIR, output_video_path)
                print(f"--- Finished alignment for: {media_filepath} ---")
            
            # Always move the generated files to the done directory to keep the main WORK_DIR clean
            move_files_to_done_directory(WORK_DIR, output_basename)

        except Exception as e:
            print(f"!!! An error occurred while processing {media_filepath}: {e} !!!")
            # Optionally, you can add more robust error handling here
            continue

    print("✅ Batch processing finished!")

if __name__ == "__main__":
    main()

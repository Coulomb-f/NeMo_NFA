# ===============================================================================
#                                   RUN
# ===============================================================================
import os
import glob

# 1) Setup
# Create input and done directories
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "done"), exist_ok=True)
setup_environment(WORK_DIR, NEMO_DIR_PATH)
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

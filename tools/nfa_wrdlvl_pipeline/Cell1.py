# ===============================================================================
#                            CONFIGURATION
# ===============================================================================
import os

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

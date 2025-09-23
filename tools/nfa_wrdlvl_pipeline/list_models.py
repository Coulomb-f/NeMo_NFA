import nemo.collections.asr as nemo_asr

# Get the list of all pretrained ASR models
all_models = nemo_asr.models.ASRModel.list_available_models()

# --- Optional: Filter for CTC models required by NFA ---
ctc_models = [model for model in all_models if "ctc" in model.pretrained_model_name]

print("--- All Available ASR Models ---")
for model in all_models:
    print(model)

print("\n\n--- Filtered for CTC Models (for NFA) ---")
for model in ctc_models:
    print(model)
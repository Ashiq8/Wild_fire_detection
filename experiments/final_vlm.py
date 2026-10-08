import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# ========================================
# Final VLM - Wildfire Scene Description
# ========================================

MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"

IMAGE_PATH = "data/images/image_0.jpg"
OUTPUT_PATH = "outputs/vlm/sample_0000_description.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("================================")
print("VLM Wildfire Scene Analysis")
print("================================")
print(f"Device: {DEVICE}")
print(f"Image : {IMAGE_PATH}")

# -----------------------------
# Load image
# -----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")

# -----------------------------
# Load VLM
# -----------------------------
print("\nLoading SmolVLM...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

model = model.to(DEVICE)
model.eval()

print("SmolVLM loaded.")

# -----------------------------
# Prompt
# -----------------------------
prompt = """
Analyze this wildfire image.

Describe briefly:
1. Whether visible wildfire/fire is present.
2. Where the fire appears to be located.
3. Any visible smoke or flames.
4. The surrounding vegetation or terrain.

Do not identify people or objects that are not clearly visible.
Give a concise description in 3-5 sentences.
"""

# -----------------------------
# Prepare input
# -----------------------------
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

inputs = processor(
    text=text,
    images=[image],
    return_tensors="pt",
)

inputs = {
    k: v.to(DEVICE) if hasattr(v, "to") else v
    for k, v in inputs.items()
}

# -----------------------------
# Generate
# -----------------------------
print("\nGenerating VLM description...")

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
    )

generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]

# Remove prompt if it appears in decoded output
if "Assistant:" in generated_text:
    generated_text = generated_text.split("Assistant:", 1)[1].strip()

print("\n================================")
print("VLM OUTPUT")
print("================================")
print(generated_text)

# -----------------------------
# Save output
# -----------------------------
from pathlib import Path

Path("outputs/vlm").mkdir(
    parents=True,
    exist_ok=True,
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(generated_text)

print("\n================================")
print("VLM COMPLETE")
print("================================")
print(f"Saved : {OUTPUT_PATH}")
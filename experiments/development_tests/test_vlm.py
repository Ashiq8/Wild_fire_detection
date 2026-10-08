import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Loading processor...")

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print("Loading VLM...")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    model.eval()

    print("VLM loaded successfully.")

    # Load real FLAME image
    image = Image.open(
        "data/images/image_0.jpg"
    ).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": (
                        "Analyze this wildfire image. "
                        "Describe what you see and identify "
                        "any visible fire or smoke regions."
                    )
                }
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt"
    )

    inputs = inputs.to(device)

    print("Generating explanation...")

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=150
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )

    print("\n========== VLM OUTPUT ==========\n")
    print(generated_text[0])


if __name__ == "__main__":
    main()
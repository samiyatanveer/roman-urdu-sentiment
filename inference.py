from transformers import pipeline

# Load the fine-tuned model
model_path = "./results/checkpoint-2023"  # This number might change, we'll confirm after training

pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Test example
test_texts = [
    "Yeh bilkul amazing hai",
    "Bilkul bakwaas tha",
    "Okay theek hai"
]

for text in test_texts:
    result = pipe(text)
    print(f"Text: {text}")
    print(f"Result: {result}\n")
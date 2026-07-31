BASE_MODEL_PATH = "/kaggle/input/llama-3.2/transformers/3b-instruct/1"
LORA_PATH = "output/"
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
COMPLETE_PHRASE = "Answer:"
BASE_PROMPT = f"Reddit moderation: Does the comment violate the rule? Answer '{POSITIVE_ANSWER}' or '{NEGATIVE_ANSWER}' only."
BASE_MODEL_PATH = "/kaggle/input/berts/transformers/default/1/ettin-encoder-400m"

LORA_PATH = "output_deberta/"  # just a folder name; now stores full finetuned model
DATA_PATH = "/kaggle/input/jigsaw-agile-community-rules/"

POSITIVE_ANSWER = "Yes"
NEGATIVE_ANSWER = "No"
COMPLETE_PHRASE = "Answer:"
BASE_PROMPT = "Reddit moderation: Does the comment violate the rule? Answer 'Yes' or 'No' only."
# app/convert_model.py
from transformers import TFBertForSequenceClassification, BertTokenizer

# Path to the folder containing tf_model.preproc
model_name_or_path = "C:/Users/user/Downloads/3e5-77acc-20250912T133637Z-1-001/3e5-77acc/Citisense/model"

# Load the TensorFlow model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
model = TFBertForSequenceClassification.from_pretrained(model_name_or_path, from_tf=True)

# Save the model in PyTorch format
model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("Model successfully converted to PyTorch format!")
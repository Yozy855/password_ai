# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("DunnBC22/codebert-base-Password_Strength_Classifier")
model = AutoModelForSequenceClassification.from_pretrained("DunnBC22/codebert-base-Password_Strength_Classifier")


#test it and compare with ours. if its better use this one
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("DunnBC22/codebert-base-Password_Strength_Classifier")
model = AutoModelForSequenceClassification.from_pretrained("DunnBC22/codebert-base-Password_Strength_Classifier")




'''import pandas as pd
mine = pd.read_csv("password_strength_with_explanations.csv")
kaggle = pd.read_csv("path/to/kaggle/password-strength-classifier-dataset.csv")  # after download

print("mine shape:", mine.shape)
print("kaggle shape:", kaggle.shape)
print(mine['strength'].value_counts(), "\n")
print(kaggle['strength'].value_counts())



from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import joblib
import pandas as pd
from sklearn.metrics import classification_report

# load HF model
model_name = "DunnBC22/codebert-base-Password_Strength_Classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
pipe = TextClassificationPipeline(model=hf_model, tokenizer=tokenizer, return_all_scores=False)

# load your test set
df_test = pd.read_csv("your_test.csv")   # contains columns 'password' and 'strength' (int or label)
preds = []
for pw in df_test['password'].tolist():
    out = pipe(pw)[0]  # {'label': 'LABEL_0', 'score': 0.99} or possibly 'Weak' depending on card
    label = out['label']
    preds.append(label)

print(classification_report(df_test['strength'], preds))



'''
#test it and compare with ours. if its better use this one
# main.py
import joblib
#from password_classifier import predict_strength
from brute_force import analyze_password, format_time
#from feedback_model import generate_feedback
from sft_feedback_model import generate_feedback

CRACK_THRESHOLD = 60 * 60 * 24 * 7  # 1 week

model = joblib.load("password_model.pkl")
vectorizer = joblib.load("password_vectorizer.pkl")


def classify_password(password: str) -> str:
    '''inputs = tokenizer(password, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        label_id = torch.argmax(logits).item()'''
    X = vectorizer.transform([password])
    label_id = model.predict(X)[0]

    labels = ["weak", "medium", "strong"]  # must match your training
    return labels[label_id]


def main():

    brute_too_short_time = False
    detected_in_list = False
    pwd = input("Enter password: ")

    ml_label = classify_password(pwd)
    brute_seconds = analyze_password(pwd)

    print(f"ML Strength: {ml_label}")

    if brute_seconds is None:
        print("Detected in common-passwords list--so didn't go through brute force")
        detected_in_list = True
    else:
        print(f"Bruteforce seconds: {brute_seconds}")
        print(f"Bruteforce time: {format_time(brute_seconds)}")
        if brute_seconds < CRACK_THRESHOLD:
            print("Bruteforce time is less than threshold (a week)")
            brute_too_short_time = True

    # Trigger feedback logic
    if ml_label == "weak" or brute_too_short_time or detected_in_list:
        print("\nPassword flagged as weak, common, or easy to crack. (Feedback model will run here)")
    
        from sft_feedback_model import generate_feedback

        print(generate_feedback(pwd)) 

    else:
        print("\nPassword is acceptable.")


#load feedback model


if __name__ == "__main__":
    main()

from password_model import predict_strength, train_model

def estimate_bruteforce_time(pw, guesses_per_second=1e6):
    """
    Estimate worst-case brute-force time for a password.
    This is theoretical, for educational/security analysis only.
    """
    has_lower = any(c.islower() for c in pw)
    has_upper = any(c.isupper() for c in pw)
    has_digit = any(c.isdigit() for c in pw)
    has_symbol = any(not c.isalnum() for c in pw)

    charset_size = 0
    if has_lower:
        charset_size += 26
    if has_upper:
        charset_size += 26
    if has_digit:
        charset_size += 10
    if has_symbol:
        # rough estimate for common printable symbols
        charset_size += 32

    length = len(pw)
    if charset_size == 0 or length == 0:
        return None  # invalid

    # total possible combinations
    N = charset_size ** length
    # expected guesses ~ N/2
    expected_guesses = N / 2.0
    seconds = expected_guesses / guesses_per_second
    return seconds

def format_time(seconds):
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f} minutes"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.2f} hours"
    days = hours / 24
    if days < 365:
        return f"{days:.2f} days"
    years = days / 365
    return f"{years:.2e} years"

def analyze_password(pw, model, vectorizer, guesses_per_second=1e6):
    strength = predict_strength(pw, model=model, vectorizer=vectorizer)
    t_seconds = estimate_bruteforce_time(pw, guesses_per_second)
    t_str = format_time(t_seconds)

    print(f"Password: {pw}")
    print(f"  ML Strength Label: {strength}")
    print(f"  Estimated brute-force time (@ {guesses_per_second:.0f} guesses/sec): {t_str}")
    print()

if __name__ == "__main__":
    print("Training model (this may take a moment)...")
    model, vectorizer = train_model(sample_n=50000)

    print("\n--- Brute-force Analysis ---")
    test_passwords = [
        "password123",
        "Summer2025!",
        "Tg!93xQ#zA",
        "Yozy!2025CS!!"
    ]

    for pw in test_passwords:
        analyze_password(pw, model, vectorizer)
    
    # Optional: interactive
    while True:
        user_pw = input("Enter a password to analyze (or 'q' to quit): ")
        if user_pw.lower().strip() == 'q':
            break
        analyze_password(user_pw, model, vectorizer)

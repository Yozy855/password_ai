

import pandas as pd

COMMON_PWDS = set()

def load_common_passwords(path="ignis-1K.csv"):
    global COMMON_PWDS
    df = pd.read_csv(path, header=None)
    COMMON_PWDS = set(p.lower().strip() for p in df[0].astype(str))

load_common_passwords()


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

def analyze_password(pw, guesses_per_second=1e6):
     # 1) Check dictionary hits first
    if pw.lower() in COMMON_PWDS:
        print(f"Password: {pw}")
        print("  ❌ This password appears in the common-passwords list!")
        print("  Attackers crack it instantly.\n")
        return  # no need to bother with brute-force
    
    t_seconds = estimate_bruteforce_time(pw, guesses_per_second)
    #t_str = format_time(t_seconds)

    return t_seconds#t_str
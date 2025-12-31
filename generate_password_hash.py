#!/usr/bin/env python3
"""
Password Hash Generator
=======================
Use this script to generate password hashes for new users.

Usage:
    python generate_password_hash.py

Then copy the hash into your secrets.toml file.
"""

import hashlib
import secrets
import string


def generate_hash(password: str) -> str:
    """Generate a SHA-256 hash of the password."""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_random_password(length: int = 16) -> str:
    """Generate a secure random password."""
    alphabet = string.ascii_letters + string.digits + "!@#$%&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def main():
    print("=" * 50)
    print("Password Hash Generator")
    print("=" * 50)
    print()
    
    print("Options:")
    print("  1. Hash an existing password")
    print("  2. Generate a new random password + hash")
    print()
    
    choice = input("Choose (1 or 2): ").strip()
    
    if choice == "1":
        print()
        password = input("Enter password to hash: ")
        hash_value = generate_hash(password)
        
        print()
        print("=" * 50)
        print("Results:")
        print("=" * 50)
        print(f"Password: {password}")
        print(f"Hash:     {hash_value}")
        print()
        print("Add this to your secrets.toml:")
        print()
        email = input("User email (for the config): ").strip() or "user@example.com"
        name = input("User name (optional): ").strip() or "New User"
        company = input("Company name (optional): ").strip() or ""
        
        print()
        print("-" * 50)
        print(f'[users."{email}"]')
        print(f'password_hash = "{hash_value}"')
        print(f'name = "{name}"')
        if company:
            print(f'company = "{company}"')
        print('plan = "standard"')
        print("-" * 50)
        
    elif choice == "2":
        print()
        password = generate_random_password()
        hash_value = generate_hash(password)
        
        print()
        print("=" * 50)
        print("Generated Credentials:")
        print("=" * 50)
        print(f"Password: {password}")
        print(f"Hash:     {hash_value}")
        print()
        print("⚠️  Send the PASSWORD to the user.")
        print("⚠️  Add the HASH to your secrets.toml.")
        print()
        
        email = input("User email (for the config): ").strip() or "user@example.com"
        name = input("User name (optional): ").strip() or "New User"
        company = input("Company name (optional): ").strip() or ""
        
        print()
        print("-" * 50)
        print("For secrets.toml:")
        print("-" * 50)
        print(f'[users."{email}"]')
        print(f'password_hash = "{hash_value}"')
        print(f'name = "{name}"')
        if company:
            print(f'company = "{company}"')
        print('plan = "standard"')
        print("-" * 50)
        print()
        print("Email template:")
        print("-" * 50)
        print(f"Subject: Your {name} account is ready")
        print()
        print(f"Hi {name.split()[0] if name else 'there'},")
        print()
        print("Your Local Growth Estimator account is ready!")
        print()
        print(f"Email: {email}")
        print(f"Password: {password}")
        print()
        print("Log in at: [YOUR_APP_URL]")
        print()
        print("Please change your password after first login.")
        print("-" * 50)
    
    else:
        print("Invalid choice. Run the script again.")


if __name__ == "__main__":
    main()

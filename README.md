# Local Growth Estimator - Deployment Guide

## Quick Start

### 1. Prepare Your Repository

Create a new **private** GitHub repository with these files:

```
your-repo/
├── app.py                    # Entry point (gated access)
├── ppc_estimator_app.py      # Main estimator logic
├── generate_password_hash.py # Helper for adding users
├── requirements.txt          # Dependencies
├── .streamlit/
│   └── secrets.toml          # Local secrets (DO NOT COMMIT)
└── README.md
```

> ⚠️ **IMPORTANT:** Never commit `secrets.toml` to Git. Add it to `.gitignore`.

### 2. Set Up Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `app.py`
6. Click "Deploy"

### 3. Configure Secrets (User Credentials)

In Streamlit Cloud:
1. Go to your app dashboard
2. Click ⚙️ Settings → Secrets
3. Paste your secrets configuration:

```toml
[users."customer@agency.com"]
password_hash = "YOUR_HASH_HERE"
name = "Customer Name"
company = "Their Agency"
plan = "pro"
```

### 4. Generate Password Hashes

Run the helper script locally:

```bash
python generate_password_hash.py
```

Choose option 2 to generate a random password + hash:
- Send the **password** to the customer
- Add the **hash** to Streamlit Cloud secrets

---

## Adding New Users (Your Workflow)

When a new customer pays:

1. **Generate credentials:**
   ```bash
   python generate_password_hash.py
   ```

2. **Add to Streamlit Cloud:**
   - Go to app Settings → Secrets
   - Add the new `[users."email"]` block
   - Save

3. **Email the customer:**
   - Send them their email + password
   - Link to your app URL

4. **Track in your system:**
   - Add to your CRM/spreadsheet
   - Note their plan type and signup date

---

## Waitlist Integration

1. **Create a Tally.so form** (or Typeform):
   - Email (required)
   - Agency name
   - How they heard about you
   - What channels they serve (PPC/SEO/Maps)

2. **Update `app.py`:**
   - Replace `WAITLIST_FORM_URL` with your form URL
   - Replace `COMPANY_NAME` with your company
   - Replace `SUPPORT_EMAIL` with your email

3. **Set up notifications:**
   - Tally/Typeform can email you on each submission
   - Or connect to Zapier → your CRM

---

## Pricing Strategy (Suggestion)

### Early Access / Validation Phase
- **Price:** $49-99/month
- **Goal:** Get 10 paying users to validate demand
- **Process:** Manual invoicing via Stripe Payment Links

### Growth Phase (10+ users)
- **Starter:** $49/mo - Basic access
- **Pro:** $99/mo - All features + priority support
- **Agency:** $199/mo - Multiple team members (future)

---

## Security Notes

1. **Passwords are hashed** - You never store plain text
2. **Sessions expire** - Default 24 hours (configurable)
3. **Secrets are encrypted** - Streamlit Cloud encrypts at rest
4. **Private repo** - Keep your code private

### What This DOESN'T Have (Yet)
- Password reset flow (you'll reset manually)
- Email verification
- Usage tracking/limits
- Billing integration

These can be added when you have enough users to justify the complexity.

---

## Customization

### Change Session Timeout
In `app.py`:
```python
SESSION_TIMEOUT_HOURS = 24  # Change this
```

### Add Plan-Based Features
In `ppc_estimator_app.py`, check the plan:
```python
user_info = get_user_info(st.session_state.username)
if user_info['plan'] == 'pro':
    # Show pro features
```

### Custom Domain
1. Upgrade to Streamlit Teams ($250/mo) for custom domains
2. Or use a reverse proxy (Cloudflare, nginx) in front of the default URL

---

## Troubleshooting

### "Invalid email or password"
- Check that the email in secrets matches exactly (case-sensitive)
- Regenerate the hash and verify it matches

### "Main application file not found"
- Ensure `ppc_estimator_app.py` is in the same directory as `app.py`
- Check the file name is exact

### Users can't log in after adding to secrets
- Secrets changes require app restart
- Go to app dashboard → Reboot app

---

## Next Steps After Validation

Once you have 10+ paying users:

1. **Add Stripe Billing** - Automate payments
2. **Add User Database** - PostgreSQL for user management
3. **Add Password Reset** - Self-service via email
4. **Consider Full Rebuild** - Next.js + proper backend

But don't build any of that until you've validated people will pay.

---

## Support

Questions? Issues?  
Create an issue in your repo or email [your support email].

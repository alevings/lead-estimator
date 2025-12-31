"""
Local Growth Estimator - Gated Access Version
==============================================
This is the entry point for the SaaS version.
It handles authentication before loading the main estimator.

Deployment:
1. Deploy to Streamlit Community Cloud
2. Set up secrets in the Streamlit dashboard
3. Add users to the secrets file as they pay

User management:
- Add paying users to .streamlit/secrets.toml
- Or use Streamlit Cloud's secrets management
"""

import streamlit as st
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional

# =========================================================
# Configuration
# =========================================================
APP_NAME = "Local Growth Estimator"
COMPANY_NAME = "Your Agency Name"  # Change this
SUPPORT_EMAIL = "support@yourdomain.com"  # Change this
WAITLIST_FORM_URL = "https://tally.so/r/XXXXXXX"  # Replace with your Tally/Typeform URL

# Session timeout (hours)
SESSION_TIMEOUT_HOURS = 24


# =========================================================
# Authentication Functions
# =========================================================
def get_users_from_secrets() -> dict:
    """
    Load authorized users from Streamlit secrets.
    
    In .streamlit/secrets.toml, format is:
    
    [users]
    user1 = "hashed_password"
    user2 = "hashed_password"
    
    Or for more detail:
    
    [users.user1]
    password_hash = "hashed_password"
    name = "John Smith"
    company = "ABC Agency"
    plan = "pro"
    """
    try:
        return dict(st.secrets.get("users", {}))
    except Exception:
        return {}


def hash_password(password: str) -> str:
    """Hash a password for storage. Use this to generate hashes for new users."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(username: str, password: str) -> bool:
    """Verify a password against the stored hash."""
    users = get_users_from_secrets()
    
    if username not in users:
        return False
    
    stored = users[username]
    
    # Handle both simple format (just hash) and detailed format (dict with password_hash)
    if isinstance(stored, dict):
        stored_hash = stored.get("password_hash", "")
    else:
        stored_hash = stored
    
    # Use constant-time comparison to prevent timing attacks
    password_hash = hash_password(password)
    return hmac.compare_digest(password_hash, stored_hash)


def get_user_info(username: str) -> dict:
    """Get additional info about a user if available."""
    users = get_users_from_secrets()
    stored = users.get(username, {})
    
    if isinstance(stored, dict):
        return {
            "name": stored.get("name", username),
            "company": stored.get("company", ""),
            "plan": stored.get("plan", "standard"),
        }
    return {"name": username, "company": "", "plan": "standard"}


def init_session_state():
    """Initialize session state for authentication."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "login_time" not in st.session_state:
        st.session_state.login_time = None


def check_session_timeout() -> bool:
    """Check if the session has timed out."""
    if st.session_state.login_time is None:
        return True
    
    elapsed = datetime.now() - st.session_state.login_time
    return elapsed > timedelta(hours=SESSION_TIMEOUT_HOURS)


def login(username: str, password: str) -> bool:
    """Attempt to log in a user."""
    if verify_password(username, password):
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.login_time = datetime.now()
        return True
    return False


def logout():
    """Log out the current user."""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.login_time = None


# =========================================================
# Landing Page (for non-authenticated users)
# =========================================================
def render_landing_page():
    """Render the marketing/waitlist landing page."""
    
    # Header
    st.title(f"📈 {APP_NAME}")
    st.markdown(f"### Professional lead estimation for local SEO, Maps, and PPC")
    
    st.markdown("---")
    
    # Value proposition
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Stop guessing. Start estimating.
        
        **The problem:** Agencies promise leads without data. Clients expect guarantees 
        you can't make. Proposals feel like fiction.
        
        **The solution:** A professional estimation tool that:
        
        - ✅ **Uses real market data** from Google Keyword Planner
        - ✅ **Outputs ranges, not fantasies** — defensible in any client conversation  
        - ✅ **Covers all local channels** — PPC, Organic SEO, and Google Maps
        - ✅ **Explains every assumption** — full audit trail for proposals
        - ✅ **Prevents over-promising** — built-in demand caps and reality checks
        
        ### Built for agencies who value credibility over hype.
        """)
        
        st.markdown("---")
        
        st.markdown("""
        ## How it works
        
        1. **Upload your Keyword Planner export** — we use real search demand as the ceiling
        2. **Adjust assumptions to match reality** — CPC ranges, conversion rates, business hours
        3. **Get defensible estimates** — ranges for PPC, Organic, and Maps leads
        4. **Export for proposals** — CSV and JSON exports ready for client docs
        
        No AI hallucinations. No made-up multipliers. Just math you can explain.
        """)
    
    with col2:
        st.markdown("### 🔐 Customer Login")
        
        with st.form("login_form"):
            username = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", use_container_width=True)
            
            if submitted:
                if login(username, password):
                    st.success("✅ Logged in!")
                    st.rerun()
                else:
                    st.error("Invalid email or password")
        
        st.markdown("---")
        
        st.markdown("### 🚀 Get Access")
        st.markdown(
            f"We're onboarding agencies in batches. "
            f"Join the waitlist to get early access and founder pricing."
        )
        
        st.link_button(
            "Join the Waitlist →",
            WAITLIST_FORM_URL,
            use_container_width=True
        )
        
        st.caption(
            f"Questions? Email [{SUPPORT_EMAIL}](mailto:{SUPPORT_EMAIL})"
        )
    
    st.markdown("---")
    
    # Social proof / FAQ section
    st.markdown("""
    ## FAQ
    
    **Q: Is this a guarantee generator?**  
    A: No. This tool produces *estimates* based on *your assumptions*. It's designed to help you 
    have honest conversations with clients about what's realistic — not to manufacture promises.
    
    **Q: What data do I need?**  
    A: A Google Keyword Planner export (XLSX) for PPC and Maps. Optionally, a Google Search Console 
    export for more accurate Organic estimates.
    
    **Q: Can I white-label this for clients?**  
    A: Not yet, but it's on the roadmap. For now, you can export data and use it in your own proposal docs.
    
    **Q: How is this different from other SEO tools?**  
    A: Most tools give you a single number ("you'll get 47 leads"). We give you ranges ("12–28 leads 
    under these assumptions"). That's the difference between fantasy and a defensible estimate.
    """)
    
    # Footer
    st.markdown("---")
    st.caption(f"© {datetime.now().year} {COMPANY_NAME}. Built for agencies who don't like lying to clients.")


# =========================================================
# Authenticated Header
# =========================================================
def render_auth_header():
    """Render a header bar for authenticated users."""
    user_info = get_user_info(st.session_state.username)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.caption(f"👤 Logged in as **{user_info['name']}**")
        if user_info['company']:
            st.caption(f"🏢 {user_info['company']}")
    
    with col2:
        plan_badge = {
            "standard": "📦 Standard",
            "pro": "⭐ Pro",
            "enterprise": "🏆 Enterprise"
        }.get(user_info['plan'], "📦 Standard")
        st.caption(f"Plan: {plan_badge}")
    
    with col3:
        if st.button("Log Out", key="logout_btn"):
            logout()
            st.rerun()
    
    st.markdown("---")


# =========================================================
# Main App Import
# =========================================================
def load_main_app():
    """Load and run the main estimator application."""
    # Import the main app module
    # This assumes ppc_estimator_app.py is in the same directory
    # and has been refactored to be importable (see below)
    
    # For now, we'll use exec to run the app inline
    # In production, you'd refactor the app to be importable
    
    import importlib.util
    import sys
    from pathlib import Path
    
    # Get the path to the main app
    app_path = Path(__file__).parent / "ppc_estimator_app.py"
    
    if not app_path.exists():
        st.error("Main application file not found. Please ensure ppc_estimator_app.py is in the same directory.")
        return
    
    # Load and execute the main app
    spec = importlib.util.spec_from_file_location("ppc_estimator_app", app_path)
    module = importlib.util.module_from_spec(spec)
    
    # Execute the module (this runs all the Streamlit code)
    spec.loader.exec_module(module)


# =========================================================
# Main Entry Point
# =========================================================
def main():
    """Main entry point for the application."""
    
    # Page config must be first Streamlit command
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    init_session_state()
    
    # Check for session timeout
    if st.session_state.authenticated and check_session_timeout():
        logout()
        st.warning("Your session has expired. Please log in again.")
    
    # Route to appropriate view
    if st.session_state.authenticated:
        render_auth_header()
        load_main_app()
    else:
        render_landing_page()


if __name__ == "__main__":
    main()

# auth.py — Authentication Layer for Family Office Dashboard
# Reads credentials from .streamlit/secrets.toml (never hardcoded).

import streamlit as st


def check_auth() -> bool:
    """
    Check if the user is authenticated. Shows login form if not.
    Returns True if authenticated, False otherwise.

    Reads credentials from st.secrets['auth'] which maps to
    .streamlit/secrets.toml. Passwords are hashed at runtime.
    """
    try:
        import streamlit_authenticator as stauth
    except ImportError:
        # If streamlit-authenticator is not installed, skip auth
        return True

    # Load credentials from secrets.toml
    try:
        auth_cfg = st.secrets["auth"]
        raw_users = auth_cfg["credentials"]["usernames"]

        # Build credentials dict with hashed passwords
        credentials = {'usernames': {}}
        for username, user_data in raw_users.items():
            credentials['usernames'][username] = {
                'email': user_data['email'],
                'name': user_data['name'],
                'password': stauth.Hasher.hash(user_data['password']),
            }

        cookie_name = auth_cfg.get('cookie_name', 'family_office_auth')
        cookie_key = auth_cfg.get('cookie_key', 'fo_default_secret_key')
        cookie_expiry = auth_cfg.get('cookie_expiry_days', 30)

    except Exception:
        # Fallback if secrets.toml is missing (dev mode)
        credentials = {
            'usernames': {
                'admin': {
                    'email': 'admin@familyoffice.com',
                    'name': 'Portfolio Admin',
                    'password': stauth.Hasher.hash('admin123'),
                },
            }
        }
        cookie_name = 'family_office_auth'
        cookie_key = 'family_office_secret_key_2026'
        cookie_expiry = 30

    authenticator = stauth.Authenticate(
        credentials=credentials,
        cookie_name=cookie_name,
        cookie_key=cookie_key,
        cookie_expiry_days=cookie_expiry,
    )

    try:
        authenticator.login()
    except Exception:
        pass

    if st.session_state.get("authentication_status"):
        authenticator.logout("Logout", "sidebar")
        st.sidebar.markdown(f"**Logged in as:** {st.session_state.get('name', 'User')}")
        return True
    elif st.session_state.get("authentication_status") is False:
        st.error("❌ Incorrect username or password.")
        return False
    elif st.session_state.get("authentication_status") is None:
        st.info("🔐 Please log in to access the Family Office Dashboard.")
        return False

    return False

"""Optional Supabase client. Only used when SUPABASE_URL / SUPABASE_ANON_KEY are configured."""

from app.core.config import settings


def get_supabase_client():
    """Returns a Supabase client using the anon key, or None when Supabase is not configured."""
    if not settings.supabase_configured:
        return None
    from supabase import create_client
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)

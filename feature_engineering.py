import re
import socket
import ssl
import datetime
from urllib.parse import urlparse

# ─────────────────────────────────────────────
#  FEATURE EXTRACTION  (30 features – must
#  exactly match the columns the scaler/model
#  were trained on)
# ─────────────────────────────────────────────

FEATURE_COLUMNS = [
    "having_IP_Address", "URL_Length", "Shortining_Service",
    "having_At_Symbol", "double_slash_redirecting", "Prefix_Suffix",
    "having_Sub_Domain", "SSLfinal_State", "Domain_registeration_length",
    "Favicon", "port", "HTTPS_token", "Request_URL", "URL_of_Anchor",
    "Links_in_tags", "SFH", "Submitting_to_email", "Abnormal_URL",
    "Redirect", "on_mouseover", "RightClick", "popUpWidnow", "Iframe",
    "age_of_domain", "DNSRecord", "web_traffic", "Page_Rank",
    "Google_Index", "Links_pointing_to_page", "Statistical_report",
]


# ── individual feature helpers ──────────────────────────────────────────────

def _having_IP_Address(url: str) -> int:
    netloc = urlparse(url).netloc
    return -1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", netloc) else 1


def _URL_Length(url: str) -> int:
    n = len(url)
    if n < 54:
        return 1
    if n <= 75:
        return 0
    return -1


def _Shortining_Service(url: str) -> int:
    shorteners = [
        "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co",
        "is.gd", "buff.ly", "adf.ly", "shorte.st",
    ]
    return -1 if any(s in url.lower() for s in shorteners) else 1


def _having_At_Symbol(url: str) -> int:
    return -1 if "@" in url else 1


def _double_slash_redirecting(url: str) -> int:
    # look for // after position 7 (skip the http://)
    return -1 if url.rfind("//") > 7 else 1


def _Prefix_Suffix(url: str) -> int:
    return -1 if "-" in urlparse(url).netloc else 1


def _having_Sub_Domain(url: str) -> int:
    netloc = urlparse(url).netloc
    # remove www
    parts = netloc.replace("www.", "").split(".")
    if len(parts) <= 2:
        return 1
    if len(parts) == 3:
        return 0
    return -1


def _SSLfinal_State(url: str) -> int:
    try:
        netloc = urlparse(url).netloc.split(":")[0]
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=netloc) as s:
            s.settimeout(3)
            s.connect((netloc, 443))
        return 1
    except Exception:
        return -1


def _Domain_registeration_length(url: str) -> int:
    """Return 1 (>1 year) or -1 without making external calls.
    We fall back to a heuristic: TLD-only domains registered for
    commercial use tend to be renewed; unknown → conservative -1."""
    try:
        import whois  # optional dependency
        netloc = urlparse(url).netloc
        domain = whois.whois(netloc)
        exp = domain.expiration_date
        if isinstance(exp, list):
            exp = exp[0]
        if exp and (exp - datetime.datetime.now()).days > 365:
            return 1
        return -1
    except Exception:
        return -1


def _Favicon(url: str) -> int:
    # Simplified: assume favicon matches domain → 1
    return 1


def _port(url: str) -> int:
    netloc = urlparse(url).netloc
    if ":" in netloc:
        try:
            port = int(netloc.split(":")[1])
            risky = {21, 22, 23, 445, 1433, 1521, 3306, 3389, 8080, 8443}
            return -1 if port in risky else 1
        except ValueError:
            return -1
    return 1


def _HTTPS_token(url: str) -> int:
    # 'https' in the domain part (not scheme) is a red flag
    netloc = urlparse(url).netloc
    return -1 if "https" in netloc.lower() else 1


def _Request_URL(url: str) -> int:
    # We can't fetch the page easily; conservative positive
    return 1


def _URL_of_Anchor(url: str) -> int:
    return 1  # simplified


def _Links_in_tags(url: str) -> int:
    return 1  # simplified


def _SFH(url: str) -> int:
    return 1  # simplified


def _Submitting_to_email(url: str) -> int:
    return -1 if "mailto:" in url.lower() else 1


def _Abnormal_URL(url: str) -> int:
    netloc = urlparse(url).netloc
    # hostname should appear in the URL path as well for normal sites
    return 1 if netloc and netloc in url else -1


def _Redirect(url: str) -> int:
    # Count occurrences of //
    return 0 if url.count("//") > 1 else 1


def _on_mouseover(url: str) -> int:
    return 1  # simplified


def _RightClick(url: str) -> int:
    return 1  # simplified


def _popUpWidnow(url: str) -> int:
    return 1  # simplified


def _Iframe(url: str) -> int:
    return 1  # simplified


def _age_of_domain(url: str) -> int:
    try:
        import whois
        netloc = urlparse(url).netloc
        domain = whois.whois(netloc)
        created = domain.creation_date
        if isinstance(created, list):
            created = created[0]
        if created and (datetime.datetime.now() - created).days > 180:
            return 1
        return -1
    except Exception:
        return -1


def _DNSRecord(url: str) -> int:
    try:
        netloc = urlparse(url).netloc.split(":")[0]
        socket.gethostbyname(netloc)
        return 1
    except Exception:
        return -1


def _web_traffic(url: str) -> int:
    return 0  # unknown without external API


def _Page_Rank(url: str) -> int:
    return -1  # unknown without external API


def _Google_Index(url: str) -> int:
    return 1  # assume indexed (conservative)


def _Links_pointing_to_page(url: str) -> int:
    return 0  # unknown


def _Statistical_report(url: str) -> int:
    # Cross-check against some known bad TLD patterns
    bad_tlds = [".tk", ".ml", ".ga", ".cf", ".gq"]
    netloc = urlparse(url).netloc.lower()
    return -1 if any(netloc.endswith(t) for t in bad_tlds) else 1


# ── public API ──────────────────────────────────────────────────────────────

def extract_features(url: str) -> list:
    """Return a list of 30 integer features in training-column order."""
    return [
        _having_IP_Address(url),
        _URL_Length(url),
        _Shortining_Service(url),
        _having_At_Symbol(url),
        _double_slash_redirecting(url),
        _Prefix_Suffix(url),
        _having_Sub_Domain(url),
        _SSLfinal_State(url),
        _Domain_registeration_length(url),
        _Favicon(url),
        _port(url),
        _HTTPS_token(url),
        _Request_URL(url),
        _URL_of_Anchor(url),
        _Links_in_tags(url),
        _SFH(url),
        _Submitting_to_email(url),
        _Abnormal_URL(url),
        _Redirect(url),
        _on_mouseover(url),
        _RightClick(url),
        _popUpWidnow(url),
        _Iframe(url),
        _age_of_domain(url),
        _DNSRecord(url),
        _web_traffic(url),
        _Page_Rank(url),
        _Google_Index(url),
        _Links_pointing_to_page(url),
        _Statistical_report(url),
    ]


def get_feature_dict(url: str) -> dict:
    """Return features as a named dict (useful for display/debugging)."""
    values = extract_features(url)
    return dict(zip(FEATURE_COLUMNS, values))


def extra_security_checks(url: str) -> float:
    """
    Rule-based risk boost on top of ML score.
    Returns a float 0.0 – 0.40 to add to model probability.
    """
    score = 0.0

    if not url.startswith("https"):
        score += 0.10

    keywords = ["login", "verify", "update", "secure", "account",
                 "bank", "paypal", "signin", "confirm", "password",
                 "credential", "ebay", "amazon", "microsoft", "apple"]
    hits = sum(1 for w in keywords if w in url.lower())
    score += min(hits * 0.04, 0.20)

    netloc = urlparse(url).netloc
    if netloc.count(".") > 3:
        score += 0.10

    if re.search(r"\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}", netloc):
        score += 0.10

    return min(score, 0.40)

"""Prompts for LLM-based evaluations."""

SIGNATURE_EXTRACTION_PROMPT = """Extract the email signature from this email. The signature is the sign-off at the end, which may include:
- Name
- Title/role
- Contact info (phone, email)
- Company name

Return ONLY the signature text, nothing else. If there is no signature, return exactly: NO_SIGNATURE

Email:
{email}"""

# Stricter prompt - requires business contact info
SIGNATURE_EXTRACTION_PROMPT_STRICT = """Extract the FORMAL BUSINESS SIGNATURE from this email.

A formal business signature MUST contain contact information such as:
- Email address
- Phone number
- Company name or website

The following are NOT signatures - return NO_SIGNATURE for these:
- Just a name or initials (e.g., "John", "JD", "Thanks, Mike")
- Casual sign-offs (e.g., "Love, Mom", "Cheers", "Best wishes")
- Names without any contact info

Return ONLY the signature text, nothing else. If there is no formal business signature, return exactly: NO_SIGNATURE

Email:
{email}"""

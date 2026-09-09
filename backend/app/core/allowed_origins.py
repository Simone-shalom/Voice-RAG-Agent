import os


def get_allowed_origins() -> list[str]:
    return [
        origin.strip()
        for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ]

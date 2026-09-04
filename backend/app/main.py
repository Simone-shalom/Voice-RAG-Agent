from fastapi import FastAPI

app = FastAPI(title="Voice Knowledge Agent")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

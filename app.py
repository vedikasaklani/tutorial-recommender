import io
import json
import logging
import os
import threading

from flask import Flask, Response, jsonify, request
from pipeline import recommend

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB; guards against memory-exhaustion uploads


@app.after_request
def add_cors_headers(resp):
    # Only allow browser extensions to call this, not arbitrary websites the
    # user happens to have open while the local server is running.
    origin = request.headers.get("Origin", "")
    if origin.startswith(("chrome-extension://", "moz-extension://")):
        resp.headers["Access-Control-Allow-Origin"] = origin
    return resp


@app.route("/")
def index():
    return "PDF-to-Video Recommender API. POST a PDF file to /recommend."


@app.route("/recommend", methods=["POST"])
def recommend_route():
    file = request.files.get("pdf")
    if not file:
        return jsonify({"error": "missing 'pdf' file"}), 400

    pdf_bytes = io.BytesIO(file.read())
    result = {}

    def run():
        try:
            result["videos"] = recommend(pdf_bytes)
        except Exception as e:
            logger.error("recommend failed: %s", e, exc_info=True)
            result["error"] = f"{type(e).__name__}: {e}"

    def generate():
        # ponytail: pipeline can take minutes (HF + YouTube calls); a plain
        # request would sit idle long enough for the browser to kill the
        # connection, so send heartbeat whitespace (valid leading JSON
        # whitespace) while the real work runs on a thread.
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        while thread.is_alive():
            thread.join(timeout=10)
            if thread.is_alive():
                yield " "
        yield json.dumps({"error": result["error"]} if "error" in result else result["videos"])

    return Response(generate(), mimetype="application/json")


if __name__ == "__main__":
    app.run()

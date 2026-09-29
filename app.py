from flask import Flask, render_template_string
from youtube_api import links  # runs the pipeline once at startup

app = Flask(__name__)

PAGE = """<!doctype html><title>Recommendations</title>
<body style="font-family:sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem">
<h1>Recommended videos</h1>
{% for v in videos %}<p><a href="{{ v.link }}" target="_blank">{{ v.title|safe }}</a></p>{% endfor %}"""

@app.route("/")
def index():
    return render_template_string(PAGE, videos=links.values())

if __name__ == "__main__":
    app.run()

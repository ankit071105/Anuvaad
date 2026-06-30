"""
app.py
======
A tiny Flask server so you can use the model from your browser — entirely on
localhost, no external services. It loads the trained checkpoint once at
startup and exposes one JSON endpoint.

Run:  python app.py
Then open http://127.0.0.1:5000
"""
from flask import Flask, render_template, request, jsonify

from config import cfg

app = Flask(__name__)

# Lazy global so the (slow) model load happens once.
_translator = None
_load_error = None


def get_translator():
    global _translator, _load_error
    if _translator is None and _load_error is None:
        try:
            from translate import Translator
            _translator = Translator()
        except FileNotFoundError as e:
            _load_error = (
                "No trained model found. Run `python tokenizer_train.py` then "
                "`python train.py` first. (" + str(e) + ")"
            )
        except Exception as e:        # pragma: no cover
            _load_error = f"Failed to load model: {e}"
    return _translator


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    mode = data.get("mode", "beam")
    if not text:
        return jsonify({"error": "Please enter some English text."}), 400

    tr = get_translator()
    if tr is None:
        return jsonify({"error": _load_error}), 503

    try:
        if mode == "greedy":
            out = tr.greedy_decode(text)
        else:
            out = tr.beam_decode(text, beam_size=4)
        return jsonify({"translation": out})
    except Exception as e:           # pragma: no cover
        return jsonify({"error": f"Translation failed: {e}"}), 500


if __name__ == "__main__":
    print("Loading model (first request may take a moment)...")
    # Pre-warm so the first user request isn't slow; ignore errors here.
    get_translator()
    app.run(host="127.0.0.1", port=5000, debug=False)

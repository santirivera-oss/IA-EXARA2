from flask import Flask, request, jsonify
import ollama

app = Flask(__name__)

@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    model = data.get("model", "gemma")
    try:
        response = ollama.chat(model=model, messages=[{"role":"user","content":prompt}])
        # Retornamos solo el contenido
        content = response.get("message", {}).get("content", str(response))
        return jsonify({"response": content})
    except Exception as e:
        return jsonify({"response": f"⚠️ Error: {e}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    response = {"message": "Hello, Flask!", "status": "success"}
    return jsonify(response)  # Converts dictionary to JSON

if __name__ == "__main__":
    app.run(debug=True)

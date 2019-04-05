import training as m
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    res = request.args.get('s')
    response = m.response(res)

    return jsonify({
        'request': res,
        'status': 200,
        'response': response
    })

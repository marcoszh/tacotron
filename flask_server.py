import argparse
import flask
from flask import Response
# from flask import request
from hparams import hparams, hparams_debug_string
import os
from librosa import effects
from synthesizer import Synthesizer


# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
synthesizer = Synthesizer()





class SynthesisResource:
  def on_get(self, req, res):
    if not req.params.get('text'):
      raise falcon.HTTPBadRequest()
    res.data = synthesizer.synthesize(req.params.get('text'))
    res.content_type = 'audio/wav'



@app.route("/predict", methods=["GET"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "GET":
        if flask.request.args.get("text"):
            # read the image in PIL format
            try:
                text = flask.request.args.get("text")

                aud = synthesizer.synthesize(text)
                # data["predictions"] = aud

            except Exception as ex:
                error_response = {
                    'error_message': "Unexpected error",
                    'stack_trace': str(ex)
                }
                return flask.make_response(flask.jsonify(error_response), 403)

            # indicate that the request was a success
            data["success"] = True
          else:
            return flask.make_response(flask.jsonify(data), 400)

    # return the data dictionary as a JSON response
    return flask.make_response(flask.jsonify(data), 200)

  # if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
  print(("* Loading TF model and Flask starting server..."
      "please wait until server has fully started"))
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Full path to model checkpoint')
  parser.add_argument('--port', type=int, default=9000)
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  print(hparams_debug_string())
  synthesizer.load(args.checkpoint)
  print('model loaded')
  app.run()

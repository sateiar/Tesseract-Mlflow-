import mlflow.pyfunc
import mlflow
import tesserocr
from tesserocr import PyTessBaseAPI
import os
import re
import time
import click
import pandas as pd
import json
from PIL import Image
#add the local db
os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'
experiment_name = "ocrmodel"
mlflow.set_experiment(experiment_name)


class ocrmodel(mlflow.pyfunc.PythonModel):

    def __init__(self, oem=tesserocr.OEM.DEFAULT):
        self.oem = oem
    def ocr (self,data):
        print(data)
        api = PyTessBaseAPI(lang=data[2], psm=data[1],oem=self.oem)
        api.SetImageFile(data[0])
        txt=api.GetUTF8Text()
        confident=api.AllWordConfidences()
        return txt,confident

    def predict(self, context, model_input):
        result = []
        condfident = []
        for data in model_input.values:
            txt , conf=self.ocr(data)
            result.append(txt)
            condfident.append(conf)
        return pd.DataFrame({"text": result,"confident":condfident})

# @click.command(help="Prepare Tesseract for Serving")
# @click.option("--lang", type=click.STRING, default="eng", help="Document Language")
# @click.argument("image_path")
def start(image_path):
    # image_path = "96486_image_19.png"

    model_input = pd.DataFrame({"path": [image_path],"psm":[tesserocr.PSM.SINGLE_BLOCK],"lang":["eng"],"image_read_type":['path']})
    df_input = pd.DataFrame({
        "path": ["96486_image_19.png"],
        "lang": ["eng"],
        "psm": [tesserocr.PSM.SINGLE_BLOCK]
    })
    with mlflow.start_run(run_name="ocrmodel") as run:
        tic = time.time()
        ocr_model = ocrmodel(oem=tesserocr.OEM.DEFAULT)
        duration_training = time.time() - tic
        #     run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        tic = time.time()
        model_path = "ocr_model33"
        mlflow.pyfunc.save_model(path=model_path, python_model=ocr_model)
        loaded_model = mlflow.pyfunc.load_pyfunc(model_path)

        model_input = pd.DataFrame(
            {"path": [image_path], "psm": [tesserocr.PSM.SINGLE_BLOCK], "lang": ["eng"]})
        model_output = loaded_model.predict(model_input)

        #     ocrmodel.predict()
        duration_prediction = time.time() - tic
        mlflow.log_metric("Load Model Time", duration_training)
        mlflow.log_metric("OCR Time", duration_prediction)
        mlflow.log_param("Language", 'eng')
        mlflow.log_param("psm", tesserocr.PSM.SINGLE_BLOCK)
        #     mlflow.pyfunc.log_model(loaded_model, "model")
        mlflow.end_run()

if __name__ == '__main__':
    start(image_path =  "96486_image_19.png")
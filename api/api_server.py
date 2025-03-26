import io
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import mlflow
from api_helpers import resize_image, load_model_from_registry, make_prediction


""" 
run app by running "fastapi run FastAPIserver.py" in terminal.
Go to localhost = 127.0.0.1. Add "/docs" to url to get to API-frontend and check endpoints!
DOES NOT WORK when called from directory of this script! 
Has to be called from parent directory via: uvicorn api_server:app --host 0.0.0.0 --port 8000 (127.0.0.1 for local)
"""


# make app
app = FastAPI(title = "Deploying an ML Model for Pneumonia Detection")

# root
@app.get("/")
def home():
    return "root of this API"

# endpoint for uploading image
@app.post("/upload_image")
# async defines an asynchronous function => These functions can be paused and resumed, 
# allowing other tasks to run while waiting for external operations, such as network requests or file I/O.
async def upload_image_and_integer( 
    file: UploadFile = File(...)
):

    try: 
        # read the uploaded file into memory as bytes
        image_bytes = await file.read()
        
        # convert bytes to a PIL image, then ensure its integrity
        image = Image.open(io.BytesIO(image_bytes))
        image.verify() # can't be used if i want to process the image
    
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    
    # load image again (as it has been deconstructed by .verify())
    image = Image.open(io.BytesIO(image_bytes))

    # convert the PIL image to np.array
    image_array = np.asarray(image)

    # modell laden
    model_name_test = "MobileNet_transfer_learning"  # Small_CNN, MobileNet_transfer_learning, MobileNet_transfer_learning_finetuned
    model_version_test = 1

    # Bild laden
    img = image_array

    # mlflow setting
    "    mlflow server --host 127.0.0.1 --port 8080     "
    mlflow.set_tracking_uri("http://127.0.0.1:8080")


    model, input_shape, input_type = load_model_from_registry(model_name = model_name_test, model_version=model_version_test)


    formatted_image = resize_image(image=img, signature_shape = input_shape, signature_dtype=input_type)


    y_pred = {"prediction": str(make_prediction(model, image_as_array=formatted_image))}

    return y_pred

################# host specification #################

# my localhost adress
host = "127.0.0.1"

# run server
if __name__ == "__main__":
    uvicorn.run(app, host=host, port=8000, root_path="/") 
# GUI at http://127.0.0.1:8000/docs



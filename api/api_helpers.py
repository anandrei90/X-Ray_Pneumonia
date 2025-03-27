import numpy as np
from tensorflow import keras
import mlflow
from PIL import Image
import io
from fastapi import HTTPException


def resize_image(
    image,
    signature_shape,
    signature_dtype
    ):

    # convert image to numpy array
    image_array = np.asarray(image)
    image_array = image_array.reshape((*image_array.shape,1))

    # shape according to signature
    if signature_shape[-1] > 1:
        # populate each channel with the same pixel values
        img_array_tuple = tuple([image_array for i in range(signature_shape[-1])])
        image_array = np.concatenate(img_array_tuple, axis = -1)

    # resizing according to signature_shape
    resized_image = keras.ops.image.resize(
        image_array,
        size = (signature_shape[1], signature_shape[2]),
        interpolation="bilinear",
        )
    
    # converting to numpy and retyping according to signature_type
    image_array = resized_image.numpy().reshape(signature_shape)
    image_array = image_array.astype(signature_dtype)

    return image_array


def load_model_from_registry(model_name, model_version = 1):
    
    # start_loading = time.time()
    print("start loading model")
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    # end_loading = time.time()
    # print("loading time: ", end_loading - start_loading)
    print("model loaded")
    # Signature abrufen
    signature = model.metadata.signature
    input_shape = signature.inputs.to_dict()[0]['tensor-spec']['shape'] 
    input_type = signature.inputs.to_dict()[0]['tensor-spec']['dtype']

    return model, input_shape, input_type

def make_prediction(model, image_as_array):
    
    prediction = model.predict(image_as_array)
    pred_reshaped = float(prediction.flatten())

    return pred_reshaped


def return_verified_image_as_numpy_arr(image_bytes):
        try: 
            
            # convert bytes to a PIL image, then ensure its integrity
            image = Image.open(io.BytesIO(image_bytes))
            image.verify() # can't be used if i want to process the image
        
        except Exception:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
        
        # load image again (as it has been deconstructed by .verify())
        validated_image = Image.open(io.BytesIO(image_bytes))

        # convert the PIL image to np.array
        validated_image_as_numpy = np.asarray(validated_image)
        return validated_image_as_numpy



# if run locally (for tests)
if __name__ == "__main__":
    # modell laden
    model_name_test = "MobileNet_transfer_learning_finetuned"  # Small_CNN, MobileNet_transfer_learning, MobileNet_transfer_learning_finetuned
    model_version_test = 1

    # Bild laden (ist )
    img = Image.open('IM-0001-0001.jpeg')

    # mlflow setting
    "    mlflow server --host 127.0.0.1 --port 8080     "
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    model, input_shape, input_type = load_model_from_registry(model_name = model_name_test, model_version=model_version_test)

    formatted_image = resize_image(image=img, signature_shape = input_shape, signature_dtype=input_type)

    y_pred = make_prediction(model, image_as_array=formatted_image)

    print(y_pred)
import numpy as np
from tensorflow import keras
import mlflow
from PIL import Image
import io
import os
from fastapi import HTTPException
from mlflow import MlflowClient
import csv
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def resize_image(
    image,
    signature_shape,
    signature_dtype
    ):
    """
    This function recieves images (as numpy array) and an mlflow model signature (shape and type) and reformats the array according to the signature.
    
    Parameters
    ----------
    image : numpy array
        image representation as numpy array.
    signature_shape : tuple ( , , , )
        Required to reshape the numpy array according to this tuple shape.
    signature_dtype : python type specification
        Required to specify the datatype of the numpy array elements of the output.
        
    Returns
    -------
    image_array: numpy array
        Reshaped and retyped numpy array
    """

    # convert image to numpy array
    image_array = np.asarray(image)
    image_array = image_array.reshape((*image_array.shape,1))

    # shape according to signature
    if signature_shape[-1] > 1:
        # populate each channel with the same pixel values
        img_array_tuple = tuple([image_array for i in range(signature_shape[-1])])
        image_array = np.concatenate(img_array_tuple, axis = -1)

    # resizing according to signature_shape. Using helper function from keras
    resized_image = keras.ops.image.resize(
        image_array,
        size = (signature_shape[1], signature_shape[2]),
        interpolation="bilinear",
        )
    
    # converting to numpy and retyping according to signature_type
    image_array = resized_image.numpy().reshape(signature_shape)
    image_array = image_array.astype(signature_dtype)

    return image_array

def load_model_from_registry(model_name, alias):
    """
    Is used to load an mlflow model from its registry (i.e., to fetch the corresponding artifact).
    Model is fetched according to given model name and alias. The model and its signature data are returned.

    Parameters
    ----------
    model_name : string
        The registered model's name.
    alias : string
        The registered model's alias.
        
    Returns
    -------
    model: mlflow model
    input_shape: tuple
        Reflects the models required signature shape (specification of data structure for the model input during predictions)
    input_type:
        The models required input data type in element level.
    """

    # start_loading = time.time()
    print("start loading model")
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{alias}")
    # end_loading = time.time()
    # print("loading time: ", end_loading - start_loading)
    print("model loaded")
    # extract signature
    signature = model.metadata.signature
    input_shape = signature.inputs.to_dict()[0]['tensor-spec']['shape'] 
    input_type = signature.inputs.to_dict()[0]['tensor-spec']['dtype']
    
    return model, input_shape, input_type

def make_prediction(model, image_as_array):
    """
    Simple function to return a prediction of a given model on a given input array.

    Parameters
    ----------
    model : mlflow model
        Mlflow model object. Has to be retrieved earlier by pufunc loading (mlflow)
    image_as_array : numpy array
        Image representation.
        
    Returns
    -------
    pred_reshaped: numpy array
        Model prediction as numpy array.
    """

    prediction = model.predict(image_as_array)
    pred_reshaped = float(prediction.flatten())

    return pred_reshaped

def return_verified_image_as_numpy_arr(image_bytes):
    """
    Verification and reformatting function.
    Verifies image type of input. Returns formatted numpy array.

    Parameters
    ----------
    image_bytes : byte object 
        Retrieved from API-endpoint.
        
    Returns
    -------
    validated_image_as_numpy: numpy array
        Image type has been validated, return as numpy array.
    """        
        
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

def get_modelversion_and_tag(model_name, model_alias):
    """
    Fetches modelversion and tag by given model name and alias.
    Both infos are retrieved from the file system of the mlflow registry of the project.

    Parameters
    ----------
    model_name : string
    model_alias : string
        
    Returns
    -------
    version_number : string
        Version number of registered mlflow model (registry model)
    tag : string
        Tag of registered model's version (registry model)
    """ 

    # get absolute path of the project dir
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    # get path of model folder and aliases subfolder (both are used later)
    aliases_path = os.path.join(project_folder ,f"unified_experiment/mlruns/models/{model_name}/aliases")
    
    # aliases_path = os.path.abspath(os.path.join("..",f"unified_experiment/mlruns/models/{model_name}/aliases"))
    model_path = os.path.dirname(aliases_path)
    
    
    # find alias (containing version number) in aliases subfolder, read version number from found file
    alias_file = os.path.join(aliases_path, model_alias)
    with open(alias_file, 'r') as file:
        version_number = int(file.read().strip())
        
    # use version number to find folder of model version
    version_dir = os.path.join(model_path, f"version-{version_number}")
    if not os.path.exists(version_dir):
        raise FileNotFoundError(f"Folder {version_dir} does not exist")
    
    # find subfolder containing tags
    tags_dir = os.path.join(version_dir, "tags")
    if not os.path.exists(tags_dir):
        raise FileNotFoundError(f"folder 'tags' in {version_dir} is missing")
    
    # extract titel from first tag-file found
    tag_files = os.listdir(tags_dir)
    if not tag_files:
        raise FileNotFoundError(f"No tags found in {tags_dir}")
    
    tag = tag_files[0].strip()

    return version_number, tag

def get_performance_indicators(num_steps_short_term = 1):


    # setting the uri 
    client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
        
    # get the three experiments: performance + (baseline, challenger, champion)
    print("getting experiment list")
    experiments = list(client.search_experiments())[:3]
    
    # get experiment names and ids
    print("getting experiment names and ids")
    exp_names = [exp.name for exp in experiments]
    exp_ids = [exp.experiment_id for exp in experiments]
    
    # define an empty dictionary to hold the performance indicators for each experiment
    performance_dictionary = {}
    
    # for loop to calculate perfomance indicators for each experiment/model
    for exp_name, exp_id in zip(exp_names, exp_ids):
        print(f"{exp_name}: getting runs")
        # all runs in the experiment with exp_id, i.e. number of predictions made
        runs = list(client.search_runs(experiment_ids = exp_id))
        
        # run_ids
        run_ids = [run.info.run_id for run in runs]
        
        # extract lists of accuracies, timestamps, and correct prediction labels
        # within the given experiment (0 = no pneumonia, 1 = pneumonia)
        print(f"{exp_name}: starting extraction of accuracies")
        accuracies = [list(client.get_metric_history(run_id = run_id, key = 'accuracy'))[0].value for run_id in run_ids]
        print(f"{exp_name}: starting extraction of time stamps")
        timestamps = [list(client.get_metric_history(run_id = run_id, key = 'accuracy'))[0].timestamp for run_id in run_ids]
        print(f"{exp_name}: starting extraction of input_labels")
        y_true = [list(client.get_metric_history(run_id = run_id, key = 'y_true'))[0].value for run_id in run_ids]
        
        # 1st row is timestamps, 2nd is accuracies and so on
        values_array = np.array([timestamps, accuracies, y_true])
        # sorts according to the timestamps
        values_array = values_array[:, values_array[0].argsort()]
        # get rid of the timestamps row
        values_array = values_array[1:]
        
        print(f"{exp_name}: calc confusion matrix")
        # calculate confusion matrix elements
        true_positives = np.sum((values_array[0] == 1) & (values_array[1] == 1))
        true_negatives = np.sum((values_array[0] == 1) & (values_array[1] == 0))
        false_negatives = np.sum((values_array[0] == 0) & (values_array[1] == 1))
        false_positives = np.sum((values_array[0] == 0) & (values_array[1] == 0))
        
        # save the experiment information in a dictionary
        exp_dictionary ={
            'all-time average accuracy': str(np.mean(values_array[0])),
            'total number of predictions': str(len(accuracies)),
            f'average accuracy for the last {num_steps_short_term} predictions': str(np.mean(values_array[0,-num_steps_short_term:])),
            'pneumonia true positives': str(true_positives),
            'pneumonia true negatives': str(true_negatives),
            'pneumonia false positives': str(false_positives), 
            'pneumonia false negatives': str(false_negatives),
             
        }
        
        # update the dictionary containing the information from the other experiments
        performance_dictionary.update({exp_name: exp_dictionary})
          
    return performance_dictionary

def save_performance_data_csv(alias, timestamp, y_true, y_pred, accuracy, filename, model_version, model_tag):
    """
    Recieves data from a model's prediction to generate performance review. 
    Saves the retrieved data and some additional calculations in a csv-file under a specified path.
    Also returns the data for further processing (i.e. mlflow-logging).

    Parameters
    ----------
    alias : string
        Alias of mlflow registry model version
    timestamp : string
        Contains time of API call.
    y_true : integer (0 or 1)
        True label of image
    y_pred : float (0 <= y_pred <=1)
        Predicted label of image
    accuracy : int (0 or 1)
        accuracy of prediction
    filename: string
        name of predicted file
    model_version : int
        Version number of mlflow registry model version
    model_tag : string
        Tag of mlflow registry model version

    Returns
    -------
    data : dictionary
        Dictionary of data to be logged into csv-file
    """ 

    # take time
    start_time = time.time()

    # get absolute path of the project dir to later find required csv-files
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # get path of folder for performance tracking (where csv files are located)
    tracking_path = os.path.join(project_folder ,f"unified_experiment/performance_tracking")
    
    # make folder, if not existing yet
    os.makedirs(tracking_path, exist_ok=True)
    file_path = os.path.join(tracking_path, f'performance_data_{alias}.csv')
    
    # initializing standard values for cumulative and global values
    log_counter = 1
    cumulative_accuracy = accuracy
    global_accuracy = accuracy
    last_50_accuracy = accuracy
    
    # Calculate consecutive values from last row's values and current values
    if os.path.exists(file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            if rows:
                last_row = rows[-1]
                # get values of last row to determine cumulations and global accuracy
                log_counter = int(last_row['log_counter']) + 1
                cumulative_accuracy = float(last_row['cumulative_accuracy']) + accuracy
                global_accuracy = cumulative_accuracy / log_counter
                # get last values (or less, if not enough rows available)
                num_previous = min(49, log_counter - 1)
                relevant_rows = rows[-num_previous:]
                relevant_accuracies = [float(row['accuracy']) for row in relevant_rows] + [accuracy]
                last_50_accuracy = sum(relevant_accuracies) / len(relevant_accuracies)

    # prepare data for output (formatting)
    data = {
        'log_counter': log_counter,
        'timestamp': timestamp,
        'y_true': y_true,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'cumulative_accuracy': cumulative_accuracy,
        'global_accuracy': global_accuracy,
        'accuracy_last_50_predictions': last_50_accuracy,
        'filename': filename,
        'model_version': model_version,
        'model_tag': model_tag,
        "model_alias": alias
    }
    
    # Check if file exists already
    file_exists = os.path.isfile(file_path)
    
    # Open file in append mode 
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        # Write header only if file is newly created
        if not file_exists:
            writer.writeheader()
        # Append new row
        writer.writerow(data)
    
    end_time = time.time()

    # print runtime and execution confirmation
    print("runtime performance logging: ", end_time-start_time)
    print(f"Data has been saved in {file_path}.")

    return data

def generate_performance_summary_csv(alias, last_n_predictions = 100):
    """
    Fetches logged performance data of model with given alias from corresponding csv-file.
    Fetches global accuracy, number of predictions, and floating average. 
    Additionally calculates confusion matrix of entire history.

    Parameters
    ----------
    alias : string
        Alias of mlflow registry model version.
    last_n_predictions: int
        Controls timeframe of confusion matrix. Only last n predictions will be used to calculate confusion matrix.
        
    Returns
    -------
    summary: dictionary
        Contains performance info of model runs. Dictionary values are strings.
    """     

    # get path of csv-files
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tracking_path = os.path.join(project_folder, "unified_experiment/performance_tracking")
    file_path = os.path.join(tracking_path, f'performance_data_{alias}.csv')

    if not os.path.exists(file_path):
        return "Error: CSV file not found."

    # read csv files
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    if not rows:
        return "Error: CSV file is empty."

    # get values of last prediction (cumulations, averages) to calculate consecutive values
    last_row = rows[-1]
    total_predictions = int(last_row['log_counter'])
    all_time_average = float(last_row['global_accuracy'])
    last_50_average = float(last_row['accuracy_last_50_predictions'])

    # initialize confusion matrix
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    # convert to numpy, restricted to last_n_predictions
    y_true = np.array([float(row['y_true']) for row in rows[-last_n_predictions:]])
    accuracy = np.array([float(row['accuracy']) for row in rows[-last_n_predictions:]])

    # calc confusion matrix
    true_positives = np.sum((y_true == 1) & (accuracy == 1))
    true_negatives = np.sum((y_true == 0) & (accuracy == 1))
    false_positives = np.sum((y_true == 0) & (accuracy == 0))
    false_negatives = np.sum((y_true == 1) & (accuracy == 0))

    # generate result dict
    summary = {
        f"performance csv {alias}": {
            "all-time average accuracy": f"{all_time_average:.4f}",
            "total number of predictions": str(total_predictions),
            "average accuracy last 50 predictions": f"{last_50_average:.4f}",
            "pneumonia true positives": str(true_positives),
            "pneumonia true negatives": str(true_negatives),
            "pneumonia false positives": str(false_positives),
            "pneumonia false negatives": str(false_negatives)
        }
    }

    return summary

def generate_model_comparison_plot(target = "accuracy_last_50_predictions", scaling =  "log_counter"):

    # get absolute path of the project dir
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # get path performance tracking subfolder
    tracking_path = os.path.join(project_folder ,"unified_experiment/performance_tracking")

    # get file paths of model (alias) tracking
    path_champion = os.path.join(tracking_path ,"performance_data_champion.csv")
    path_challenger = os.path.join(tracking_path ,"performance_data_challenger.csv")
    path_baseline = os.path.join(tracking_path ,"performance_data_baseline.csv")

    # open files as dataframes
    df_champion = pd.read_csv(path_champion)
    df_challenger = pd.read_csv(path_challenger)
    df_baseline = pd.read_csv(path_baseline)

    # convert timestamp to datetime
    df_champion['timestamp'] = pd.to_datetime(df_champion['timestamp'])
    df_challenger['timestamp'] = pd.to_datetime(df_challenger['timestamp'])
    df_baseline['timestamp'] = pd.to_datetime(df_baseline['timestamp'])

    # create fig and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    x_axis = scaling
    # plot thre dataframes as traces
    ax.plot(df_champion[x_axis], df_champion[target], label='Champion', color='green', linestyle='-', linewidth=2)
    ax.plot(df_challenger[x_axis], df_challenger[target], label='Challenger', color='blue', linestyle='--', linewidth=2)
    # ax.plot(df_baseline[x_axis], df_baseline[target], label='Baseline', color='red', linestyle=':', linewidth=2)

    # set common axis labels and titles
    ax.set_ylabel(target, fontsize=12)
    ax.set_title(f'Model comparison over time ({target})', fontsize=14)

    # legend
    ax.legend(fontsize=10)

    # set custom axis and title formatting according to scaling
    if scaling == "timestamp":
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        ax.set_xlabel("Time of run", fontsize=12)

    else:
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_major_locator(plt.AutoLocator())
        ax.set_xlabel("Run number", fontsize=12)

    # add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # show plot
    plt.tight_layout()

    return fig

def check_challenger_takeover(last_n_predictions = 20):

    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tracking_path = os.path.join(project_folder, "unified_experiment/performance_tracking")

    file_path_champ = os.path.join(tracking_path, f'performance_data_champion.csv')
    file_path_chall = os.path.join(tracking_path, f'performance_data_challenger.csv')

    # read csv files
    with open(file_path_champ, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    last_rows_champ = rows[-last_n_predictions:]
    last_values_champ = [row['accuracy_last_50_predictions'] for row in last_rows_champ]

    # read csv files
    with open(file_path_chall, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    last_rows_chall = rows[-last_n_predictions:]
    last_values_chall = [row['accuracy_last_50_predictions'] for row in last_rows_chall]

    # converting to numpy
    np_champ = np.array(last_values_champ, dtype=float)
    np_chall = np.array(last_values_chall, dtype=float)

    # compare by calculating difference
    diff = np_champ - np_chall

    # check if all entries negative
    check_if_chall_is_better = np.all(diff < 0)
    print(check_if_chall_is_better)

    return check_if_chall_is_better

def switch_champion_and_challenger():
    # get paths of alias files
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    aliases_path = os.path.join(project_folder, r"unified_experiment\mlruns\models\Xray_classifier\aliases")
    path_challenger = os.path.join(aliases_path, "challenger")
    path_champion = os.path.join(aliases_path, "champion")

    # read csv files
    with open(path_champion, 'r') as file:
        version_number_champion = file.read()
    with open(path_challenger, 'r') as file:
        version_number_challenger = file.read()
    # swap content (i.e. version numbers)
    with open(path_champion, 'w') as file:
        file.write(version_number_challenger)
    with open(path_challenger, 'w') as file:
        file.write(version_number_champion)


# if run locally (for tests)
if __name__ == "__main__":
    # modell laden
    model_name_test = "Xray_classifier"  # Small_CNN, MobileNet_transfer_learning, MobileNet_transfer_learning_finetuned
    model_alias = "baseline"
    
    ' ###################### test: extract version via alias ########### '
    print(get_modelversion_and_tag(model_name=model_name_test, model_alias=model_alias))

    ' #################### test: extract tag of model version #############'
    
    # Bild laden
    img = Image.open(r"C:\Users\pprie\OneDrive\Dokumente\Python_Projekte\95_Xray\data\test\NORMAL\IM-0001-0001.jpeg")

    # mlflow setting
    "    mlflow server --host 127.0.0.1 --port 8080     "
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    model, input_shape, input_type = load_model_from_registry(model_name = model_name_test, alias=model_alias)

    formatted_image = resize_image(image=img, signature_shape = input_shape, signature_dtype=input_type)

    y_pred = make_prediction(model, image_as_array=formatted_image)

    print(y_pred)
    
    print(get_performance_indicators())

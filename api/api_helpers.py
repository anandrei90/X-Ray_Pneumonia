import numpy as np
from tensorflow import keras
import mlflow
from PIL import Image
import io
import os
from fastapi import HTTPException
from mlflow import MlflowClient
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from pathlib import Path
import random
from datetime import datetime

' ##############################################################################################'
' ######################### image preprocessing, model loading, prediction #####################'

def return_verified_image_as_numpy_arr(image_bytes):

    
    '''
    Verification and reformatting function.
    Verifies image type of input. Returns formatted numpy array.
    
    Parameters
    ----------
    
    image_bytes: image as binary stream
        Input Image, converted to bytes.
        
    Returns
    -------
    Validated image in numpy array format. 
    '''   
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
    print("Start loading model")
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{alias}")
    # end_loading = time.time()
    # print("loading time: ", end_loading - start_loading)
    print(f"Model {model_name}@{alias} loaded")
    # extract signature
    signature = model.metadata.signature
    input_shape = signature.inputs.to_dict()[0]['tensor-spec']['shape'] 
    input_type = signature.inputs.to_dict()[0]['tensor-spec']['dtype']
    
    return model, input_shape, input_type

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

def resize_image(
    image,
    signature_shape,
    signature_dtype
    ):

    '''
    Function that resizes a grayscale image such that it agrees 
    with the signature and data type of the ML classifier's input.
    
    Parameters
    ----------
    
    image: PIL image/numpy array
        Image to be resized. Can only be in grayscale.
    signature_shape: tuple
        Shape of the ML classifier input.
    signature_dtype: data type
        Data type of the ML classifier input.
        
    Returns
    -------
    image_array: numpy array
        Reshaped numpy array with signature_dtype entries. 
    '''
    # convert image to numpy array
    image_array = np.asarray(image)
    image_array = image_array.reshape((*image_array.shape,1))
    
    # if ML model input has more than one channel, populate each channel with the same pixel values
    if signature_shape[-1] > 1:
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

def get_image_paths(n_samples):
    '''
    Returns a list of the paths of the images to be classified.
    
    Parameters
    ----------
    n_samples: int
        Number of images to be classified.
    
    Returns
    -------
    selected_images: list of Path objects
        List of image paths. 
    '''
    # Get absolute path of the project dir
    project_folder = Path(__file__).resolve().parent.parent

    # paths to image folders
    normal_folder = project_folder / "data" / "test" / "NORMAL"
    pneumonia_folder = project_folder / "data" / "test" / "PNEUMONIA"
    tracking_csv_path = project_folder / "unified_experiment" / "performance_tracking" / "performance_data_champion.csv"

    # load images of both classes
    normal_images = list(normal_folder.glob("*"))
    pneumonia_images = list(pneumonia_folder.glob("*"))

    # put the images together in one list
    all_images = normal_images + pneumonia_images
    
    # filter out images that were already analysed
    if tracking_csv_path.exists(): 
        # import logging dataframe to get names of already analysed images
        df_performance = pd.read_csv(tracking_csv_path)
        analysed_images = set(df_performance["filename"])

        # filter out already analysed images
        all_images = [image for image in all_images if image.name not in analysed_images]

    if n_samples <= len(all_images):
        # select a random sample from the remaining images
        selected_images = random.sample(all_images, n_samples)
    else:
        print("Chosen no. of images larger than remaining images.")
        print(f"Sending all {len(all_images)} remaining images...")
        selected_images = all_images
        random.shuffle(selected_images)
        
    return selected_images

def predict_log_switch(selected_image_paths):
    """
    Function that takes several image paths as input and classifies the
    corresponding images, logs the results in csv form and optionally
    in mlflow, and performs the switch between challenger and champion
    when needed.
    
    Parameters
    ----------
    selected_images: list of Path objects
        List of image paths returned by the get_image_paths() function. 
    
    Returns
    -------
    None
    """
    # set tracking uri for mlflow
    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # set model name
    model_name = "Xray_classifier"
    
    # load all three models and save the outputs for later use
    models = []
    input_shapes = []
    input_types = []
    aliases = ["champion", "challenger", "baseline"]
    
    for alias in aliases:
        # get model and signature
        model, input_shape, input_type  = load_model_from_registry(model_name = model_name, alias = alias)
        # store the outputs 
        models.append(model)
        input_shapes.append(input_shape)
        input_types.append(input_type)
        
    for i, image_file in enumerate(selected_image_paths):
        
        # get class from parent folder name
        data_class = image_file.parent.name
        label = 0 if data_class == "NORMAL" else 1
        
        api_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # make predictions and logging for each of the three models
        for alias, model, input_shape, input_type in zip(aliases, models, input_shapes, input_types):
            
            # get model version and tag for logging
            model_version, model_tag = get_modelversion_and_tag(model_name=model_name, model_alias=alias)
            
            # open image and resize it according to model signature
            with Image.open(image_file, "r") as img:
                formatted_image = resize_image(image=img, signature_shape = input_shape, signature_dtype=input_type)
            
            # make prediction
            y_pred = make_prediction(model, image_as_array=formatted_image)
            accuracy_pred = int(label == np.around(y_pred))
            
            # logging and precalculations in csv-file
            logged_csv_data = save_performance_data_csv(alias = alias, 
                                                       timestamp = api_timestamp, 
                                                       y_true = label, 
                                                       y_pred = y_pred, 
                                                       accuracy=accuracy_pred, 
                                                       file_name=image_file.name, 
                                                       model_version=model_version, 
                                                       model_tag=model_tag)

            # switch off mlflow tracking (if needed)
            mlflow_tracking = False 
            if mlflow_tracking:
                # logging in mlflow performance runs, if switched on
                save_performance_data_mlflow(log_counter = logged_csv_data["log_counter"], 
                                                alias = alias, 
                                                timestamp = logged_csv_data["timestamp"], 
                                                y_true = label, 
                                                y_pred = y_pred, 
                                                accuracy = accuracy_pred, 
                                                file_name = logged_csv_data["filename"], 
                                                model_version = model_version, 
                                                model_tag = model_tag)

        # check if switch should be made
        if check_challenger_takeover(last_n_predictions = 20, window = 50):
            switch_champion_and_challenger()
            # swap the challenger and champion aliases
            # instead of loading the models again
            aliases[0], aliases[1] = aliases[1], aliases[0]
        
        print(f"Prediction no. {i+1} of {len(selected_image_paths)} with class {data_class} done.")
        
' ##############################################################################################'
' ######################### logging of prediction data #########################################'

def save_performance_data_csv(alias, timestamp, y_true, y_pred, accuracy, file_name, model_version, model_tag):
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
        Accuracy of prediction
    file_name: string
        Name of image file used for prediction
    model_version : int
        Version number of mlflow registry model version
    model_tag : string
        Tag of mlflow registry model version

    Returns
    -------
    data : dictionary
        Dictionary of data to be logged into csv-file.
    """ 

    # get absolute path of the project dir to later find required csv-files
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # get path of folder for performance tracking (where csv files are located)
    tracking_path = os.path.join(project_folder ,f"unified_experiment/performance_tracking")
    
    # make folder, if not existing yet
    os.makedirs(tracking_path, exist_ok=True)
    file_path = os.path.join(tracking_path, f'performance_data_{alias}.csv')
    
    # initializing standard values for cumulative and global values
    log_counter = 1
    
    # Calculate consecutive values from last row's values and current values
    if os.path.exists(file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            if rows:
                last_row = rows[-1]
                # get values of last row to determine cumulations and global accuracy
                log_counter = int(last_row['log_counter']) + 1

    # prepare data for output (formatting)
    data = {
        'log_counter': log_counter,
        'timestamp': timestamp,
        'y_true': y_true,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'filename': file_name,
        'model_version': model_version,
        'model_tag': model_tag,
        "model_alias": alias,
        "model_switch": False
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

    # print runtime and execution confirmation
    print(f"Data has been saved in {file_path}.")

    return data

def save_performance_data_mlflow(log_counter, alias, timestamp, y_true, y_pred, accuracy, file_name, model_version, model_tag):
    """
    For a given alias, it stores the received logging data from model predicitons (runs) in a unique mlflow run of the corresponding experiment.
    
    Parameters
    ----------
    log_counter: int
        Run number of (csv-logged) run that is to be stored (note: unique mlflow run number usually differs!)
    alias : string
        Alias of mlflow registry model version
    timestamp : string
        Contains time of API call.
    y_true : integer (0 or 1)
        True label of image
    y_pred : float (0 <= y_pred <=1)
        Predicted label of image
    accuracy : int (0 or 1)
        Accuracy of prediction
    file_name: string
        Name of image file used for prediction
    model_version : int
        Version number of mlflow registry model version
    model_tag : string
        Tag of mlflow registry model version

    Returns
    -------
    None
    """ 
    
    # set experiment name for model (logging performance for each model in separate experiment)
    mlflow.set_experiment(f"performance {alias}")

    # logging of metrics
    with mlflow.start_run():
        
        # log the metrics
        metrics_dict = {
            'log counter': log_counter,
            "y_true": y_true,
            "y_pred": y_pred,
            "accuracy": accuracy,
            }
        mlflow.log_metrics(metrics_dict)

        # log model version and tag
        params = {
            'timestamp': timestamp,
            "model version": model_version,
            "model tag": model_tag,
            'image file name': file_name,
            }
        mlflow.log_params(params)



' ##############################################################################################'
' ######################### performance reporting and plotting functions #######################'

def get_performance_indicators_mlflow(num_steps_short_term):
    '''
    Function that fetches data from the mlflow client and 
    returns a dictionary summarizing the to-date performance 
    of the three pneumonia x-ray classification (aliased) models.
    
    Parameters
    ----------
    num_steps_short_term : positive int
        Size of the window used for calculating the sliding average
        of accuracy.  
        
    Returns
    -------
    performance_dictionary: dict 
        Dictionary with three keys corresponding to the three tracked
        ML classifiers. The corresponding values are also dictionaries
        with the following keys: total number of predictions, 
        average accuracy for the last {num_steps_short_term} predictions,
        pneumonia true positives, pneumonia true negatives, pneumonia 
        false positives, and pneumonia false negatives.
    
    '''

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
        # sorting according to the timestamps
        values_array = values_array[:, values_array[0].argsort()]
        # get rid of the timestamps row
        values_array = values_array[1:]
        # restrict array to latest {num_steps_short_term} runs
        values_array_short_term = values_array[:,-num_steps_short_term:]
        print(values_array_short_term.shape)

        print(f"{exp_name}: calc confusion matrix")
        # calculate confusion matrix elements
        true_positives = np.sum((values_array_short_term[0] == 1) & (values_array_short_term[1] == 1))
        true_negatives = np.sum((values_array_short_term[0] == 1) & (values_array_short_term[1] == 0))
        false_negatives = np.sum((values_array_short_term[0] == 0) & (values_array_short_term[1] == 1))
        false_positives = np.sum((values_array_short_term[0] == 0) & (values_array_short_term[1] == 0))
        
        # save the experiment information in a dictionary
        exp_dictionary ={
            'total number of predictions': str(len(accuracies)),
            f'average accuracy for the last {num_steps_short_term} predictions': str(np.mean(values_array_short_term[0])),
            'pneumonia true positives': str(true_positives),
            'pneumonia true negatives': str(true_negatives),
            'pneumonia false positives': str(false_positives), 
            'pneumonia false negatives': str(false_negatives),   
        }
        
        # update the dictionary containing the information from the other experiments
        performance_dictionary.update({exp_name: exp_dictionary})
          
    return performance_dictionary

def get_performance_indicators_csv(alias, last_n_predictions = 100):
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

    # initialize confusion matrix
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    # convert to numpy and get accuracies and true labels, restricted to last_n_predictions
    y_true = np.array([float(row['y_true']) for row in rows[-last_n_predictions:]])
    accuracy = np.array([float(row['accuracy']) for row in rows[-last_n_predictions:]])

    # calc confusion matrix
    true_positives = np.sum((y_true == 1) & (accuracy == 1))
    true_negatives = np.sum((y_true == 0) & (accuracy == 1))
    false_positives = np.sum((y_true == 0) & (accuracy == 0))
    false_negatives = np.sum((y_true == 1) & (accuracy == 0))

    # calc avg of last n predictions
    avg_last_n_predictions = np.mean(accuracy)

    # generate result dict
    summary = {
        f"performance csv {alias}": {
            "total number of predictions": str(total_predictions),
            f"average accuracy last {min(total_predictions, last_n_predictions)} predictions": f"{round(avg_last_n_predictions, 4)}",
            "pneumonia true positives": str(true_positives),
            "pneumonia true negatives": str(true_negatives),
            "pneumonia false positives": str(false_positives),
            "pneumonia false negatives": str(false_negatives)
        }
    }

    return summary

def generate_model_comparison_plot(window = 50, scaling =  "log_counter"):
    '''
    Function that generates a plot comparing the performance of
    models over time. The upper part of the plot shows the accuracy 
    of the champion and challenger models, averaged over a window
    whose lenght is specified by the {window} parameter. The lower
    part of the plot shows which of the trained ML models is the 
    challenger and the champion at a given time.
    
    Parameters
    ----------
    window : positive int
        Size of the window (= number of consecutive runs) used for 
        calculating the sliding average of the accuracy.
    scaling : "log_counter" or "timestamp"
        Controls what is shown on the x-axis of the plot. If
        scaling = "timestamp", then the x-axis shows the timestamps
        at which the api was used. Otherwise the x-axis shows the 
        run number (= number of times the api was used). 
        
    Returns
    -------
    fig: figure object 
        Figure comparing the performance of models.
    
    '''
    
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

    # get switching points from challenger csv aka. df_challenger dataframe. 
    # Result will be a pandas series containing the log_counters of the switches. 
    # The resetted index enumerates the switches.
    switch_points_log_counter = df_challenger[df_challenger["model_switch"]==True].reset_index(drop=True)["log_counter"]
    
    # define the figure and its subplots
    fig, axs = plt.subplots(2, 1, sharex=True, figsize = (16,8), height_ratios= [3,1])
    # remove horizontal space between axes
    fig.subplots_adjust(hspace=0)

    # generate plot lines
    moving_avg_challenger = moving_average_column(df_challenger["accuracy"], window = window)
    moving_avg_champion = moving_average_column(df_champion["accuracy"], window = window)
    axs[0].plot(df_champion[scaling], moving_avg_champion, label='Champion', color='blue', linestyle='-', linewidth=3)
    axs[0].plot(df_challenger[scaling], moving_avg_challenger, label='Challenger', color='orange', linestyle='--', linewidth=3)


    # vertical lines showing when the automated switches happened
    for log_counter in switch_points_log_counter:
        axs[0].axvline(
            x=log_counter,
            color='black',
            linestyle='-',
            linewidth=2,
            # generate label (legend) only for first element to avoid redundancy in legend
            label='automated model switch' if log_counter == switch_points_log_counter[0] else None
        )

    # set common axis labels and titles
    axs[0].set_ylabel(f"moving avg accuracy for the last {window} predictions", fontsize=12)
    axs[0].set_title(f'Model comparison over time', fontsize=20)

    # legend
    axs[0].legend(fontsize=15)


    # add grid
    axs[0].grid(True, linestyle='--', alpha=0.7)



    # organize all the models in a set
    models_champion = list(df_champion["model_tag"].unique())
    models_challenger = list(df_challenger["model_tag"].unique())
    models = set(models_champion + models_challenger)

    # dict for creating a new column in the df's
    # makes the y-axis ticks of lower plot easier to code
    model_mapping = {model: idx + 1 for idx, model in enumerate(models)}

    # generate plot lines
    for color, df in zip(["blue", "orange"], [df_champion, df_challenger]):
        df["plot"] = df["model_tag"].map(model_mapping)
        axs[1].plot(df["log_counter"], 
                    df["plot"], 
                    marker = "|",
                    markersize = 10, 
                    linestyle = '', 
                    color = color,
                    )

    # vertical lines showing when the automated switches happened
    for log_counter in switch_points_log_counter:
        axs[1].axvline(
            x=log_counter,
            color='black',
            linestyle='-',
            linewidth=2,
        )


    # set custom axis and title formatting according to scaling
    if scaling == "timestamp":
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        axs[1].xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
        axs[1].set_xlabel("Time of run", fontsize=12)

    else:
        axs[1].xaxis.set_major_formatter(plt.ScalarFormatter())
        axs[1].xaxis.set_major_locator(plt.AutoLocator())
        axs[1].set_xlabel("Run number", fontsize=12)

    # set limits on y-axis
    axs[1].set_ylim(bottom=0.5, top=max(model_mapping.values()) + 0.5)
    # set ticks on y-axis
    axs[1].set_yticks(tuple(model_mapping.values()), labels = models)
    # make the ticks disappear from the y-axis
    axs[1].tick_params(axis='y', which='both', length=0)

    return fig    

def generate_confusion_matrix_plot(last_n_predictions = 10):
    '''
    Function that generates plot of confusion matrix of current champion.
    
    Parameters
    ----------
    last_n_predictions : positive int
        Timeframe (number of last predictions) used for confusion matrix.
        
    Returns
    -------
    fig: figure object 
        Figure of confusion matrix.
    
    '''
    # get data from csv performance report
    data = get_performance_indicators_csv(alias = "champion", last_n_predictions=last_n_predictions)

    # extract only nested dictionary of champion
    data_champion = data["performance csv champion"]

    # restructuring input for plot
    conf_matrix = np.array([
            [
                int(data_champion["pneumonia true positives"]), 
                int(data_champion["pneumonia false positives"])
                ],
            [
                int(data_champion["pneumonia false negatives"]), 
                int(data_champion["pneumonia true negatives"])
            ]
            ])
    
    # setting up plot
    fig = plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Predicted Positive', 'Predicted Negative'],
        yticklabels=['Actual Positive', 'Actual Negative']
    )
    
    # get displayed predictions (get_performance_indicators_csv returns total number of available predictions)
    available_predictions = min(last_n_predictions, int(data_champion["total number of predictions"]))
    print(available_predictions)
    # now configure plot
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    accuracy = data_champion[f"average accuracy last {available_predictions} predictions"]
    main_title = f'Confusion Matrix (champion) last {available_predictions} predictions'
    plt.title(f"{main_title}\nAccuracy last {available_predictions} predictions: {accuracy}", pad=20)
    plt.tight_layout()
    
    return fig

' ##############################################################################################'
' ######################## model comparison and takeover (switch) functions ####################'

def moving_average_column(column, window):
    """"
    For a given input column, the moving average of column values is calculated. 
    The window parameter (input) controls, how many consecutive values are uses for the moving average calculation.
    I.e.: 
    - For a column element's index, the last {window} predecessor values are used to calculate the moving average.
    - If there are not enough predecessors available (e.g. at low column indexes), only the avalílable values are taken.
    - This means: For low column indexes (lower than {window}), the window is shortened by definition!

    Parameters
    ----------
    column : array-like
        array-like integer containing numeric values
    window: int
        Controls how many predecessing values of a column element are used to calculate the moving average at that column element's index.
        
    Returns
    -------
    np.array: numpy array
        Numpy array containing moving averages for the input column.
    """
    column = np.array(column)
    averaged_col = [np.sum(column[max(0,i-window):i])/min(i, window) for i in range(1,len(column)+1)]
    
    return np.array(averaged_col)

def check_challenger_takeover(last_n_predictions = 20, window=50):
    """"
    Fetches data from performance logs (csv-files) of challenger and champion registry model versions. 
    Checks if the challenger's moving average accuracy has been better than the champion's moving average accuracy during last_n_predictions.
    Check is done by using accuracy column from csv-files. Moving average calculation is done with {window}.
    
    Parameters
    ----------
    last_n_predictions : integer
        Input for takeover condition (model switch). 
        Challenger has to have better moving average accuracy than champion during {last_n_predictions} to take over. 
    window: int
        Window parameter for moving average calculation. Will be passed to helper function moving_average_column.
        
    Returns
    -------
    check_if_chall_is_better: boolean
        True if challenger satisfies takeover condition (model switch).
    """

    # get relevant paths
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tracking_path = os.path.join(project_folder, "unified_experiment/performance_tracking")

    file_path_champ = os.path.join(tracking_path, f'performance_data_champion.csv')
    file_path_chall = os.path.join(tracking_path, f'performance_data_challenger.csv')

    # read csv file champion
    with open(file_path_champ, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        
    # Breaking condition nr. 1: check if there are at least {last_n_predictions + window} runs. If so, break
    if len(rows) < last_n_predictions + window:
        print(f"Initial protection phase (less than {last_n_predictions + window} runs available). No switch allowed yet.")
        return False
    
    
    # # Breaking condition nr. 2: check if switch was done in the previous {last_n_predictions} runs. If so, break
    # organize the last {last_n_predictions} model tag in a list, get unique tags (set)
    last_model_tags_unique = set([row['model_tag'] for row in rows[-(last_n_predictions+window):]])
    # check is switch was performed, i.e. more than one model tags in history
    switch_done = len(last_model_tags_unique) > 1
    # quit the function if the switch was done in the last {last_n_predictions + window} runs
    if switch_done:
        print(f"A switch happend during the last {last_n_predictions+window} runs. No switch allowed yet.")
        return False
        
    # If continuing here, start model comparison.
    # get last last_n_predictions, extract accuracy as integers
    last_rows_champ = rows[-(last_n_predictions + window):]
    last_acc_values_champ = [int(row['accuracy']) for row in last_rows_champ]
    # get moving average. Careful: correct calculation by extended window and capping! 
    # moving_average_column cuts window at the lower end of the column, thus the lower end has to be extended!
    moving_averages_champ = moving_average_column(last_acc_values_champ, window)[-last_n_predictions:]

    # read csv file challenger
    with open(file_path_chall, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    # get last last_n_predictions, extract accuracy as integers
    last_rows_chall = rows[-(last_n_predictions + window):]
    last_acc_values_chall = [int(row['accuracy']) for row in last_rows_chall]
    # get moving average. Careful: correct calculation by extended window and capping!
    # moving_average_column cuts window at the lower end of the column, thus the lower end has to be extended!
    moving_averages_chall = moving_average_column(last_acc_values_chall, window)[-last_n_predictions:]
    
    # compare by calculating difference
    diff = moving_averages_champ - moving_averages_chall
    
    # check if all entries negative
    check_if_chall_is_better = np.all(diff <= 0)
    print(f"Performance comparison between challenger and champion has been made. Challenger's moving average better during last {last_n_predictions} runs: ", check_if_chall_is_better)
    
    return check_if_chall_is_better

def switch_champion_and_challenger():
    """"
    Swaps mlflow registry model versions that are associated with champion and 
    challenger model aliases. Swap is achieved by swapping content of alias files. 
    After swapping, function updates the "model_switch" column of the last run 
    in the champion's and challenger's csv files (new column value = "True") 
    
    Parameters
    ----------
    No parameters
        
    Returns
    -------
    No returns
    """

    # get paths of alias files
    project_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    unif_exp_path = os.path.join(project_folder, r"unified_experiment")
    path_challenger_alias = os.path.join(unif_exp_path, r"mlruns/models/Xray_classifier/aliases/challenger")
    path_champion_alias = os.path.join(unif_exp_path, r"mlruns/models/Xray_classifier/aliases/champion")
    path_challenger_csv = os.path.join(unif_exp_path, r"performance_tracking/performance_data_challenger.csv")
    path_champion_csv = os.path.join(unif_exp_path, r"performance_tracking/performance_data_champion.csv")

    # read alias files 
    with open(path_champion_alias, 'r') as file:
        version_number_champion = file.read()
    with open(path_challenger_alias, 'r') as file:
        version_number_challenger = file.read()

    # swap content (i.e. version numbers)
    with open(path_champion_alias, 'w') as file:
        file.write(version_number_challenger)
    with open(path_challenger_alias, 'w') as file:
        file.write(version_number_champion)
        
    # update challenger csv-files of predictions: mark model_switch
    with open(path_challenger_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows_chall = list(reader)
        rows_chall[-1]["model_switch"]="True"
    with open(path_challenger_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows_chall[0].keys())
        writer.writeheader()
        writer.writerows(rows_chall)
    # update champion csv-files of predictions: mark model switch
    with open(path_champion_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows_champ = list(reader)
        rows_champ[-1]["model_switch"]="True"
    with open(path_champion_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows_champ[0].keys())
        writer.writeheader()
        writer.writerows(rows_champ)
    print("challenger and champion have been switched")



# if run locally (for tests)
if __name__ == "__main__":
    pass
    generate_confusion_matrix_plot(last_n_predictions = 5)
    # # modell laden
    # model_name_test = "Xray_classifier"  # Small_CNN, MobileNet_transfer_learning, MobileNet_transfer_learning_finetuned
    # model_alias = "baseline"
    
    # ' ###################### test: extract version via alias ########### '
    # print(get_modelversion_and_tag(model_name=model_name_test, model_alias=model_alias))

    # ' #################### test: extract tag of model version #############'
    
    # # Bild laden
    # img = Image.open(r"C:\Users\pprie\OneDrive\Dokumente\Python_Projekte\95_Xray\data\test\NORMAL\IM-0001-0001.jpeg")

    # # mlflow setting
    # "    mlflow server --host 127.0.0.1 --port 8080     "
    # mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # model, input_shape, input_type = load_model_from_registry(model_name = model_name_test, alias=model_alias)

    # formatted_image = resize_image(image=img, signature_shape = input_shape, signature_dtype=input_type)

    # y_pred = make_prediction(model, image_as_array=formatted_image)

    # print(y_pred)
    
    # print(get_performance_indicators())
    # check_challenger_takeover(last_n_predictions = 74)
    # generate_model_comparison_plot()
    # print(moving_average_column(np.array([1,2,3,4,5]), 10))
    # print(moving_average_column([1,2,3,4,5], 10))

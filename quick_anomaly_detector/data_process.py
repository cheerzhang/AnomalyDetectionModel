import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, roc_curve
import pandas as pd
import numpy as np
import torch

#########################################
#             Histogram graph           #
#########################################
def graph_multiple_histograms(df, columns, layout=(2, 2), bin_numbers=None):
    """
    Plot multiple histograms of specified columns from a DataFrame.

    :param df: The pandas DataFrame containing the data.
    :type df: pandas.DataFrame
        
    :param columns: List of column names to be plotted.
    :type columns: list
        
    :param layout: Tuple specifying the layout dimensions of subplots. Default is (2, 2).
    :type layout: tuple, optional
        
    :param bin_numbers: List of integers specifying the number of bins for each column.
    :type bin_numbers: list, optional

    :return: The generated matplotlib Figure object containing the histograms.
    :rtype: matplotlib.figure.Figure
    
    Example:
    
    .. code-block:: python

        from quick_anomaly_detector.data_process import graph_multiple_histograms

        columns_to_plot = ['a', 'b', 'c', 'd']
        bin_numbers = [10, 20, 15, 10]  # Example list of bin numbers corresponding to each column
        fig = graph_multiple_histograms(df, columns_to_plot, layout=(2, 2), bin_numbers=bin_numbers)
        plt.show()
    
    """
    num_plots = len(columns)
    num_rows, num_cols = layout
    total_plots = num_rows * num_cols

    if num_plots > total_plots:
        raise ValueError("Number of columns exceeds the available space in the layout.")

    if bin_numbers is None:
        bin_numbers = [10] * num_plots
    elif len(bin_numbers) != num_plots:
        raise ValueError("Length of bin_numbers must match the number of columns.")

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 10))
    axes = axes.ravel()

    for i, (column, bins) in enumerate(zip(columns, bin_numbers)):
        ax = axes[i]
        ax.hist(df[column], bins=bins)
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of {column}")

    # Hide empty subplots
    for j in range(num_plots, total_plots):
        axes[j].axis('off')

    plt.tight_layout()
    return fig

def category_hist_graph(df, category_column):
    """
    This function is for plot the histograms of category feature
    """
    category_counts = df[category_column].value_counts()
    fig, ax = plt.subplots()
    category_counts.plot(kind='bar')
    ax.set_xlabel(category_column)
    ax.set_ylabel('Frequency')
    ax.set_title(f'Histogram of {category_column}')
    return fig, category_counts


##################################################
#                Scatter graph                   #
##################################################
def graph_scatter(df, x_column, y_column, color_column):
    """
    Create a scatter plot with color mapping based on a column of a DataFrame.

    :param df: The pandas DataFrame containing the data.
    :type df: pandas.DataFrame

    :return: The generated scatter plot figure.
    :rtype: matplotlib.figure.Figure

    :raises ValueError: If one or more specified columns do not exist in the DataFrame.

    Example: 

    .. code-block:: python

        from quick_anomaly_detector.data_process import graph_scatter

        df = pd.DataFrame({'x_column': [1, 2, 3], 'y_column': [4, 5, 6], 'color_column': ['red', 'blue', 'green']})
        fig = graph_scatter(df, 'x_column', 'y_column', 'color_column')
        plt.show()
    
    .. note:: 
    
        Ensure that the DataFrame contains the required columns for plotting.
    """
    # Validate input parameters
    if not all(col in df.columns for col in [x_column, y_column, color_column]):
        raise ValueError("One or more specified columns do not exist in the DataFrame.")

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df[x_column], df[y_column], c=df[color_column], cmap='viridis', alpha=0.7)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"Scatter Chart (Color by {color_column})")
    cbar = plt.colorbar(ScalarMappable(norm=None, cmap='viridis'), ax=ax, label=color_column)
    return fig



##################################################
#        Check Wrongly-Predicted Data           #
##################################################
def check_wrong(df, predict='predict', label='label'):
    """
    `check_wrong` is a function to calculate the predicted wrongly df

    :param df: The pandas DataFrame containing the data.
    :type df: pandas.DataFrame

    :param predict: the column name of predict 
    :type predict: string

    :param label: the column name of label 
    :type label: string

    :return: df_tn, df_fn, df_tp, df_fp dataframe
    :rtype: [pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]

    """
    df_tn = df[(df[predict] == 0) & (df[label] == 0)]
    df_fp = df[(df[predict] == 1) & (df[label] == 0)]
    df_fn = df[(df[predict] == 0) & (df[label] == 1)]
    df_tp = df[(df[predict] == 1) & (df[label] == 1)]
    return df_tn, df_fn, df_tp, df_fp



##################################################
#         Log Sqaure feature transform           #
##################################################
def apply_transformations(df, column_name):
    """
    Apply logarithm and square transformations to a column in a DataFrame.

    :param df: The pandas DataFrame containing the data.
    :type df: pandas.DataFrame
        
    :param column_name: The name of the column to transform.
    :type column_name: str

    :return: A list of column names including the original column and the transformed columns.
    :rtype: list
        
    Example

    .. code-block:: python

        from quick_anomaly_detector.data_process import apply_transformations

        columns_to_plot = log_sqaure(df, 'column_name')
    """
    # Apply transformations
    df[f'log_{column_name}'] = df[column_name].apply(lambda x: np.log(x + 0.001))
    df[f'square2_{column_name}'] = df[column_name].apply(lambda x: x ** 2)
    df[f'square0.5_{column_name}'] = df[column_name].apply(lambda x: x ** 0.5)
    
    # Define columns to plot
    transformed_columns = [column_name, f'log_{column_name}', f'square2_{column_name}', f'square0.5_{column_name}']
    
    return transformed_columns



##################################################
#         check valid tensor                     #
##################################################
def check_valid_tensor_data(input_tensor):
    """
    Perform checks on the input tensor.
    
    :param input_tensor: Input tensor to be checked.
    :type input_tensor: torch.Tensor
    
    :return: A tuple containing a boolean indicating whether the input passes all checks and a message indicating the result of the checks.
    :rtype: tuple[bool, str]

    
    Example: 

    .. code-block:: python

        from quick_anomaly_detector.data_process import check_valid_tensor_data

        input_tensor = torch.tensor([1.0, 2.0, float('nan'), 4.0])  # Example tensor with NaN
        valid, message = check_valid_tensor_data(input_tensor)
        print(valid, message)
    """
    # Check if input_tensor is a torch.Tensor
    if not isinstance(input_tensor, torch.Tensor):
        return False, "Input is not a torch.Tensor"
    
    # Check if input_tensor contains NaN or infinite values
    if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
        return False, "Input contains NaN or infinite values"
    
    # Check if input_tensor is of floating-point data type
    if input_tensor.dtype not in [torch.float32, torch.float64]:
        return False, "Input is not of floating-point data type"
    
    # Check if input_tensor has a valid shape (not empty)
    if input_tensor.numel() == 0:
        return False, "Input tensor has an empty shape"
    
    return True, "Input passes all checks"



#########################################
#              Eval Metics              #
#########################################
def calculate_metrics(actual_labels, predicted_labels):
    """
    Calculate various evaluation metrics for binary classification.

    :param actual_labels: Ground truth labels.
    :type actual_labels: array-like

    :param predicted_labels: Predicted labels.
    :type predicted_labels: array-like

    :return: Dictionary containing the following evaluation metrics:
        - precision (float): Precision score.
        - recall (float): Recall score.
        - label_pass_rate (float): Proportion of samples labeled as negative class in the ground truth.
        - predict_pass_rate (float): Proportion of samples predicted as negative class.
        - ks (float): Kolmogorov-Smirnov statistic.
        - gini (float): Gini coefficient.
        - f1 (float): F1 score.
        - auc_roc (float): Area under the ROC curve.
        - accuracy (float): Accuracy score.
    :rtype: dict
    """
    # Precision
    precision = precision_score(actual_labels, predicted_labels)
    # Recall
    recall = recall_score(actual_labels, predicted_labels)
    # Pass Rate
    label_pass_rate = np.mean(actual_labels == 0)
    predict_pass_rate = np.mean(predicted_labels == 0)
    # KS statistic
    fpr, tpr, _ = roc_curve(actual_labels, predicted_labels)
    ks_statistic = max(tpr - fpr)
    # F1 Score
    f1 = f1_score(actual_labels, predicted_labels)
    # AUC-ROC
    auc_roc = roc_auc_score(actual_labels, predicted_labels)
    # Gini coefficient
    gini = 2 * auc_roc - 1
    # Accuracy
    accuracy = accuracy_score(actual_labels, predicted_labels)
    metrics = {
      'precision': precision, 'recall': recall, 
      'label_pass_rate': label_pass_rate, 'predict_pass_rate': predict_pass_rate,
      'ks': ks_statistic, 'gini': gini, 'f1': f1, 'auc_roc': auc_roc, 'accuracy': accuracy
    }
    return metrics



#########################################
#              Location API             #
#########################################
address_json = {
    "amsterdam": {"lat": 52.3676, "lon": 4.9041},
    "rotterdam": {"lat": 51.9244, "lon": 4.4777},
    "'s-gravenhage": {"lat": 52.0705, "lon": 4.3007},
    "almere": {"lat": 52.3508, "lon": 5.2647},
    "eindhoven": {"lat": 51.4231, "lon": 5.4623},
    "utrecht": {"lat": 52.0907, "lon": 5.1214},
    "enschede": {"lat": 52.2215, "lon": 6.8937},
    "tilburg": {"lat": 51.5606, "lon": 5.0919},
    "arnhem": {"lat": 51.9851, "lon": 5.8987},
    "breda": {"lat": 51.5719, "lon": 4.7683},
    "den haag": {"lat": 52.0705, "lon": 4.3007},
    "groningen": {"lat": 53.2194, "lon": 6.5665}, 
    "dordrecht": {"lat": 51.8133, "lon": 4.6901}, 
    "nijmegen": {"lat": 51.8433, "lon": 5.8609}, 
    "apeldoorn": {"lat": 52.2112, "lon": 5.9699}, 
    "zoetermeer": {"lat": 52.0607, "lon": 4.4940}, 
    "amersfoort": {"lat": 52.1561, "lon": 5.3878}, 
    "almelo": {"lat": 52.3670, "lon": 6.6685}, 
    "zwolle": {"lat": 52.5168, "lon": 6.0830}, 
    "'s-hertogenbosch": {"lat": 51.6978, "lon": 5.3037},
    "leeuwarden": {"lat": 53.2012, "lon": 5.7999}, 
    "haarlem": {"lat": 52.3874, "lon": 4.6462}, 
    "leiden": {"lat": 52.1636, "lon": 4.4802}, 
    "schiedam": {"lat": 51.9170, "lon": 4.3988}, 
    "delft": {"lat": 52.0116, "lon": 4.3571}, 
    "lelystad": {"lat": 52.5185, "lon": 5.4714}, 
    "heerlen": {"lat": 50.8860, "lon": 5.9804}, 
    "helmond": {"lat": 51.4793, "lon": 5.6570}, 
    "alkmaar": {"lat": 52.6324, "lon": 4.7534}, 
    "roosendaal": {"lat": 51.5358, "lon": 4.4653}, 
    "purmerend": {"lat": 52.5144, "lon": 4.9641}, 
    "spijkenisse": {"lat": 51.8562, "lon": 4.2972}, 
    "hengelo": {"lat": 52.2574, "lon": 6.7928}, 
    "maastricht": {"lat": 50.8514, "lon": 5.6910}, 
    "deventer": {"lat": 52.2661, "lon": 6.1552}, 
    "venlo": {"lat": 51.3704, "lon": 6.1724}, 
    "hoofddorp": {"lat": 52.3061, "lon": 4.6907}, 
    "gouda": {"lat": 52.0115, "lon": 4.7105}, 
    "emmen": {"lat": 52.7862, "lon": 6.8917}, 
    "nieuwegein": {"lat": 52.0248, "lon": 5.0918}, 
    "etten-leur": {"lat": 51.5869, "lon": 4.6671}, 
    "assen": {"lat": 52.9928, "lon": 6.5642}, 
    "oss": {"lat": 51.7612, "lon": 5.5140}, 
    "zaandam": {"lat": 52.4420, "lon": 4.8292},
    "hoorn": {"lat": 52.6424, "lon": 5.0602},
    "oosterhout": {"lat": 51.6410, "lon": 4.8617},
    "hilversum": {"lat": 52.2292, "lon": 5.1669},
    "doetinchem": {"lat": 51.9647, "lon": 6.2938},
    "zwijndrecht": {"lat": 51.8106, "lon": 4.6273},
    "amstelveen": {"lat": 52.3114, "lon": 4.8701},
    "tiel": {"lat": 51.8876, "lon": 5.4279},
    "roermond": {"lat": 51.1913, "lon": 5.9878},
    "alphen aan den rijn": {"lat": 52.1294, "lon": 4.6578},
    "landgraaf": {"lat": 50.9068, "lon": 6.0255},
    "gorinchem": {"lat": 51.8372, "lon": 4.9758},
    "sittard": {"lat": 51.0006, "lon": 5.8865},
    "bergen op zoom": {"lat": 51.4946, "lon": 4.2872},
    "hoogeveen": {"lat": 52.7286, "lon": 6.4901},
    "vlaardingen": {"lat": 51.9121, "lon": 4.3494},
    "ede": {"lat": 52.0402, "lon": 5.6649},
    "beverwijk": {"lat": 52.4870, "lon": 4.6574},
    "geleen": {"lat": 50.9672, "lon": 5.8277},
    "capelle aan den ijssel": {"lat": 51.9302, "lon": 4.5777},
    "houten": {"lat": 52.0278, "lon": 5.1630},
    "veghel": {"lat": 51.6158, "lon": 5.5392},
    "ridderkerk": {"lat": 51.8703, "lon": 4.6022},
    "harderwijk": {"lat": 52.3422, "lon": 5.6367},
    "huizen": {"lat": 52.2995, "lon": 5.2434},
    "wijchen": {"lat": 51.8137, "lon": 5.7529},
}
def location_json(city='', street=''):
    lat, lon = [None, None]
    if city != '':
        if city in address_json.keys():
            lat, lon = address_json[city]['lat'], address_json[city]['lon']
    return lat, lon
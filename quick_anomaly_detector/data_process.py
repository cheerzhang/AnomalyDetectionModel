import matplotlib.pyplot as plt
import pandas as pd


#########################################
#             Histogram graph           #
#########################################
def graph_histogram(df, x_column, bin_number=10):
    """
    Plot a histogram of a specified column from a DataFrame.

        :param df: The pandas DataFrame containing the data.
        :type df: pandas.DataFrame
        :param x_column: The name of the column to be plotted.
        :type x_column: str
        :param bin_number: The number of bins to use for the histogram. Default is 10.
        :type bin_number: int, optional
        :return: The matplotlib Figure object containing the histogram plot.
        :rtype: matplotlib.figure.Figure
    
    :Example:

    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> df = pd.DataFrame({'age': [25, 30, 35, 40, 45, 50]})
    >>> fig = graph_histogram(df, 'age', bin_number=5)
    >>> plt.show()
    """
    fig, ax = plt.subplots()
    ax.hist(df[x_column], bins=bin_number)
    ax.set_xlabel(x_column)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {x_column}")
    return fig
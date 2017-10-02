# Import regressor
from numpy_regressor.regressor import Regression

# Import data_processing
from numpy_regressor.data_processing import DataProcessing

# Import Pandas
import pandas as pd

# Import Bokeh
import bokeh
from bokeh.plotting import figure, show
from bokeh.palettes import d3

# Getting the data from uci data repo
airfoil_df = pd.read_table(
    filepath_or_buffer = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
    names = ["Frequency", "Angle of attack", "Chord length", "Free-stream velocity",
             "Suction side displacement", "Scaled sound pressure"])

# Creating class objects
airfoil_regressor = Regression()
airfoil_data_process = DataProcessing()

# Splitting up train and test set
airfoil_df_train, airfoil_df_test = airfoil_data_process.train_test_split(airfoil_df)

# Calling tbe regression function to get the prediction
prediction = airfoil_regressor.my_regression(airfoil_df_train, airfoil_df_test.iloc[:, 0:-1], 1)

# Calculating lmse value
lmse = airfoil_regressor.lmse(airfoil_df_test.iloc[:, -1].values.reshape(len(airfoil_df_test), 1), prediction)
print(lmse)

# Custom style attribute function
# Best practive for bokeh users
def model_param_analysis(data, title, width = 800, height = 600, xlab = "X-axis", ylab = "Y-axis", line_width = 4):
    # Creating the variables for the color prop
    colors_list = [d3['Category20b'][17][i] for i in [2, 6]]
    # Create a parameter dataframe from the regressor object
    df = pd.DataFrame(data)
    # Structuring the x and y axis values
    basis_function = df.basis_function.unique()
    xs = [df.lamda.unique()] * len(basis_function)
    ys = [df.loc[df["basis_function"] == x]["lmse"] for x in basis_function]
    # Create the figure object
    p = figure(width = width, height = height, title = title, x_axis_label = xlab, y_axis_label = ylab)
    # Iterate through each degree and draw the lines
    for (col_name, colr, x, y) in zip(basis_function, colors_list, xs, ys):
        current_plot = p.line(x, y, color = colr, legend = col_name.upper(), line_width = line_width)
        p.legend.location = "center"
    # Add plot style attributes
    p.title.text_font_size = "20pt"
    p.title.align = "center"
    p.xaxis.axis_label_text_font_size = "20pt"
    p.yaxis.axis_label_text_font_size = "20pt"
    return show(p)

# Show the plot
model_param_analysis(data = airfoil_regressor.parameters_list, title = "Airfoil Dataset", width = 800, height = 600,
                         xlab = "Lamda", ylab = "Mean Square Error", line_width = 4)

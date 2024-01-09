from dash import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from googletrans import Translator
import webbrowser
import plotly.graph_objects as go
from datetime import datetime
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import folium
from folium.plugins import MarkerCluster
from branca.colormap import linear
import warnings

warnings.filterwarnings("ignore")
import statsmodels
from plotly.subplots import make_subplots
from matplotlib import dates
import io
import json
import base64
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.io as pio
from scipy.stats import kstest

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Create the Dash app
Final_Term_Project_app = dash.Dash("Information Visualization Final Term Project app",
                                   external_stylesheets=external_stylesheets)

#############################
df = pd.read_csv('esas.csv')
df = df.head(120000)
df = df.rename(
    columns={'Unnamed: 0': 'ID', 'satish_kodu': 'Receipt_Number', 'mehsul_kodu': 'Product_Code', 'mehsul_ad': 'Product',
             'mehsul_kateqoriya': 'Category', 'mehsul_qiymet': 'Price', 'satish_tarixi': 'Date',
             'endirim_kompaniya': 'Discount', 'bonus_kart': 'Bonus_cart', 'magaza_ad': 'Store_Branch',
             'magaza_lat': 'Store_Lat', 'magaza_long': 'Store_Long'})
clean_df = df.dropna()
clean_df = clean_df.drop('ID', axis=1)
clean_df = clean_df.reset_index(drop=True)
duplicated_rows = clean_df.duplicated()
clean_df = clean_df.drop_duplicates()
clean_df = clean_df.copy()  # Create a copy of the DataFrame
clean_df['Product'] = clean_df['Product'].astype('category')
clean_df.loc[:, 'Date'] = pd.to_datetime(clean_df['Date'], dayfirst=True)
clean_df['Category'] = clean_df['Category'].astype('category')
clean_df['Discount'] = clean_df['Discount'].astype('category')
clean_df['Store_Branch'] = clean_df['Store_Branch'].astype('category')
replace_dict = {'Sərin Yay günləri': 'Summer Season', 'S?rf?li Yaz': 'Fifth of May Season',
                'Bərəkətli Novruz': 'Happy Nowruz Season', 'Yeni il fürsətləri': 'New Year Season',
                'Payız endirimləri': 'Autumn Season'}
clean_df['Discount'] = clean_df['Discount'].replace(replace_dict)

translator = Translator()  # Increase the timeout value (in seconds)

clean_df['Category'] = clean_df['Category'].apply(lambda x: translator.translate(x).text)
clean_df['Store_Branch'] = clean_df['Store_Branch'].apply(lambda x: translator.translate(x).text)

product_counts = clean_df['Product'].value_counts()
products_to_replace = product_counts[product_counts < 12].index
clean_df['Product'] = clean_df['Product'].replace(products_to_replace, 'Others')

clean_df['Date'] = pd.to_datetime(clean_df['Date'], format='%d/%m/%Y %H:%M')
clean_df['Month'] = clean_df['Date'].dt.month
clean_df['Year'] = clean_df['Date'].dt.year

product_counts = clean_df['Product'].value_counts()
if 'Others' not in product_counts.index:
    product_counts['Others'] = 0
products_to_replace = product_counts[product_counts < 12].index
clean_df['Product'] = clean_df['Product'].replace(products_to_replace, 'Others')

new_df = clean_df[['Date', 'Price']].copy()
new_df['Date'] = pd.to_datetime(new_df['Date']).dt.date
unique_dates_count = new_df['Date'].nunique()
new_df.sort_values(by='Date', inplace=True)
#########################


server = Final_Term_Project_app.server

Final_Term_Project_app.layout = html.Div(
    [
        html.H3('Information Visualization Final Term Project app',
                style={"text-align": 'center', "color": 'blue', "font-size": '36px'}),
        dcc.Tabs(id='tabs',
                 value="q1",
                 children=[
                     dcc.Tab(label='About', value="about"),
                     dcc.Tab(label="Line plot", value="p1"),
                     dcc.Tab(label="Bar plot", value="p2"),
                     dcc.Tab(label="Count plot", value="p3"),
                     dcc.Tab(label="Pie plot", value="p4"),
                     dcc.Tab(label="3D plot", value="p5"),
                     dcc.Tab(label="Area plot", value="p6"),
                     dcc.Tab(label="PCA and other plots", value="p7"),

                 ]),
        html.Div(id="tabs-content")
    ]
)
##################################################

# Define the layout of the web application
about_layout = html.Div(
    [
        html.H1("Data Analysis of 21 Supermarkets in Baku City"),
        html.H6('About the Dataset'),
        html.P(
            id='dataset-info',
            children=[
                "The dataset used in this application provides a comprehensive information about various supermarket transactions in the year 2019 in the city of Baku, Azerbaijan.",
                " It encompasses a vast array of data, including details on a total of 438,826 products that were purchased from these supermarkets.",
                " These products were bought by a customer base of around 80,000 individuals, highlighting the breadth of consumer engagement.",
                " The transactions were distributed across 21 branches of the supermarket, capturing the geographical spread and operational diversity of the business during the specified year of 2019."
            ]
        ),

        html.P("The geo-spatial map of these supermarkets along with the Total revenue is shown below in the map:"),

        # Add an iframe to display the contents of folium_map.html
        html.Iframe(srcDoc=open('assets/folium_map.html', 'r').read(), width='90%', height='400px'),

        html.Br(),

        html.P(id='dataset-small-info', children=[
            "The dataset used in this application provides a comprehensive view of supermarket transactions in Baku, Azerbaijan in 2019.",
            html.Br(),
            html.Strong("Key Features:"),
            html.Ul([
                html.Li("Total Products / Rows : 438,826"),
                html.Li("Total Columns : 12"),
                html.Li("Customers: 80,000"),
                html.Li("Branches: 21"),
            ]),

            html.Strong("Steps used to preprocess the dataset:"),
            html.Ul([
                html.Li(" Dropped all the Null (NA) values from the dataset."),
                html.Li(" Removed 'ID' column from the dataset."),
                html.Li(" Removed any duplicate rows from the dataset."),
                html.Li(" Replaced the column name of the dataset from Azerbaijani to English."),
                html.Li(
                    " Translated the values of columns 'Category' and Store_Branch' from Azerbaijani to English using 'googleTranslator' library."),
            ]),
        ]),
        html.Br(),
        html.Strong("The information of the cleaned or preprocessed dataset is shown below:"),

        html.Div([
            html.Img(src='assets/clean_df_info.png', style={'width': '150%'}),
        ], style={'width': '30%', 'vertical-align': 'middle'}),

        html.Button(
            'Visit Dataset',
            id='visit-website-button',
            n_clicks=0,
            style={'margin-top': '10px'}
        ),

        dcc.Location(id='url', refresh=False),
        html.A(id='external-link', target='_blank', style={'display': 'none'}),

        html.Br(),

        html.H6("A Project By Shubham Laxmikant Deshmukh"),

    ]
)


# Define callback to update the external link
@Final_Term_Project_app.callback(
    Output('external-link', 'href'),
    [Input('visit-website-button', 'n_clicks')]
)
def update_external_link(n_clicks):
    if n_clicks is not None and n_clicks > 0:
        # Open the link in a new tab or window
        webbrowser.open_new_tab("https://www.kaggle.com/datasets/mexwell/supermarket-dataset")

    # Return an empty string to prevent the page from refreshing
    return ''


#############################################
# Store branch colors
branch_colors = {
    'N.Narimanov': '#1f77b4',
    'Khatai': '#ff7f0e',
    'Machine market': '#2ca02c',
    'Ahmadli': '#d62728',
    'Hypermarket': '#9467bd',
    'M. Ajami': '#8c564b',
    'Ayna Sultanova': '#e377c2',
    'Narimanov-2': '#7f7f7f',
    'C. Mammadguluzade': '#bcbd22',
    'Almond': '#17becf',
    "People\'s friendship": '#aec7e8',
    'SEK 8 million': '#ffbb78',
    'Hazi Aslanov-1': '#98df8a',
    'Zabrat': '#c5b0d5',
    'Hazi Aslanov-2': '#c49c94',
    'Yasamal': '#f7b6d2',
    'Small church': '#dbdb8d',
    'Radiozavod': '#9edae5',
    '28-May': '#d62728',
    'New Yasamal': '#1f77b4',
    '20th January': '#2ca02c'
}

# Define the layout of the app
plot1_layout = html.Div(
    [
        html.H2('Various Line plots from the Dataset'),
        html.Label('Select Store Branches:'),
        dcc.Dropdown(
            id='store-branches-dropdown',
            options=[{'label': branch, 'value': branch} for branch in clean_df['Store_Branch'].unique()],
            value=['N.Narimanov', 'Khatai', 'Machine market', 'Ahmadli', 'Hypermarket', 'M. Ajami', 'Ayna Sultanova',
                   'Narimanov-2', 'C. Mammadguluzade', 'Almond', "People's friendship", 'SEK 8 million',
                   'Hazi Aslanov-1', 'Zabrat', 'Hazi Aslanov-2', 'Yasamal', 'Small church', 'Radiozavod', '28-May',
                   'New Yasamal', '20th January'],
            multi=True
        ),
        dcc.Graph(id='revenue-plot'),
        html.Br(),  # Add a horizontal line to separate the two graphs
        dcc.Graph(id='total-revenue-plot')  # New graph for total revenue
    ]
)


# Callback to update the individual store branches revenue plot
@Final_Term_Project_app.callback(
    Output('revenue-plot', 'figure'),
    [Input('store-branches-dropdown', 'value')]
)
def update_revenue_plot(selected_branches):
    # Filter data for selected store branches
    filtered_data = clean_df[clean_df['Store_Branch'].isin(selected_branches)]

    # Group by 'Year', 'Month', and 'Store_Branch' and calculate the sum of prices
    monthly_prices = filtered_data.groupby(['Year', 'Month', 'Store_Branch'])['Price'].sum().reset_index()

    # Create a line plot using plotly express
    fig = px.line(monthly_prices, x='Month', y='Price', color='Store_Branch',
                  labels={'x': 'Month', 'y': 'Total Revenue'}, line_shape='linear',
                  color_discrete_map=branch_colors)

    fig.update_layout(title='Total Revenue Over Months for Each Store Branch',
                      xaxis_title='Month', yaxis_title='Total Revenue ($)',
                      title_x=0.5, title_y=0.95)  # Adjust title_x and title_y for positioning

    # Format y-axis labels as currency
    fig.update_layout(yaxis=dict(tickformat='$,.2f'))

    # Update x-axis labels to display month names
    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September"]
    fig.update_layout(xaxis=dict(tickvals=list(range(1, 10)), ticktext=month_names))
    fig.update_xaxes(range=[1, 9])
    return fig


# Callback to update the total revenue plot
@Final_Term_Project_app.callback(
    Output('total-revenue-plot', 'figure'),
    [Input('store-branches-dropdown', 'value')]
)
def update_total_revenue_plot(selected_branches):
    # Filter data for selected store branches
    filtered_data = clean_df[clean_df['Store_Branch'].isin(selected_branches)]

    # Group by 'Year', 'Month', and calculate the total revenue for all branches
    total_revenue = filtered_data.groupby(['Year', 'Month'])['Price'].sum().reset_index()

    # Create a line plot for total revenue
    total_fig = px.line(total_revenue, x='Month', y='Price', labels={'x': 'Month', 'y': 'Total Revenue'},
                        line_shape='linear', color_discrete_sequence=['black'])

    # Adding labels and title
    total_fig.update_layout(title='Total Revenue Over Months for All Store Branches',
                            xaxis_title='Month', yaxis_title='Total Revenue ($)',
                            title_x=0.5, title_y=0.95)  # Adjust title_x and title_y for positioning

    # Format y-axis labels as currency
    total_fig.update_layout(yaxis=dict(tickformat='$,.2f'))

    # Update x-axis labels to display month names
    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September"]
    total_fig.update_layout(xaxis=dict(tickvals=list(range(1, 10)), ticktext=month_names))

    total_fig.update_xaxes(range=[1, 9])
    return total_fig


###############################
# Define the layout of the app

# Group by 'Discount' and 'Category', and calculate the count for each combination
seasonal_category_counts = clean_df.groupby(['Discount', 'Category']).size().reset_index(name='Count')

# List of unique seasons (discounts)
seasons = seasonal_category_counts['Discount'].unique()

# Calculate average prices by category
average_prices_by_category = clean_df.groupby('Category')['Price'].mean().sort_values(ascending=False).reset_index()

# Group by 'Discount' and 'Product', and calculate the count for each combination
seasonal_product_counts = clean_df.groupby(['Discount', 'Product']).size().reset_index(name='Count')

# List of unique seasons (discounts)
seasons = seasonal_product_counts['Discount'].unique()

# Create a dictionary to map season names to numerical indices
season_indices = {season: index for index, season in enumerate(seasons)}

# Define the layout of the app
plot2_layout = html.Div([
    html.H2('Various Bar plots from the Dataset'),
    html.H6("Select the season of Discount to get the category with maximum sale:"),
    dcc.RadioItems(
        id='season-radio',
        options=[{'label': season, 'value': season} for season in seasons],
        value=seasons[0],
        labelStyle={'display': 'block'}
    ),
    dcc.Graph(id='seasonal-category-plot'),
    html.Hr(),
    html.H2("Average Prices by Category"),

    dcc.Dropdown(
        id='sort-dropdown',
        options=[
            {'label': 'Ascending', 'value': 'asc'},
            {'label': 'Descending', 'value': 'desc'}
        ],
        value='desc',
        style={'width': '50%'}
    ),

    dcc.Graph(
        id='average-price-bar-plot',
    ),

    html.Div(id='top-categories-output'),

])


@Final_Term_Project_app.callback(
    dash.dependencies.Output('seasonal-category-plot', 'figure'),
    [dash.dependencies.Input('season-radio', 'value')]
)
def update_plot(selected_season):
    # Filter data for the selected season
    season_data = seasonal_category_counts[seasonal_category_counts['Discount'] == selected_season]

    # Sort the data for the selected season in descending order
    season_data = season_data.sort_values(by='Count', ascending=False)

    # Find the category with the maximum count in the selected season
    max_category = season_data.loc[season_data['Count'].idxmax(), 'Category']

    # Plotting with Plotly Express
    fig = px.bar(
        season_data,
        x='Count',
        y='Category',
        color='Category',
        text='Count',
        title=f'Category Sold the Most in {selected_season}',
        labels={'Count': 'Category Count'},
        template='plotly',
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    # Highlight the bar for the category with the maximum count
    fig.update_traces(marker=dict(color='red'), selector=dict(name=max_category))

    # Adding labels and title
    fig.update_layout(title_x=0.5, title_y=0.95)  # Adjust title_x and title_y for positioning

    return fig


# Define callback to update the graph and print top 5 categories
@Final_Term_Project_app.callback(
    [Output('average-price-bar-plot', 'figure'),
     Output('top-categories-output', 'children')],
    [Input('sort-dropdown', 'value')]
)
def update_graph_sort(sort_order):
    sorted_df = average_prices_by_category.sort_values(by='Price', ascending=(sort_order == 'desc'))

    # Get top 5 categories and their counts
    top_categories = sorted_df.tail(5)
    top_categories_output = f"Top 5 Categories ({sort_order}): \n \n" + top_categories.to_string(index=False)

    fig = px.bar(
        sorted_df,
        x='Price',
        y='Category',
        orientation='h',
        labels={'Price': 'Average Price ($)', 'Category': 'Category'},
        title='Average Prices by Category',
        color='Price',
        color_continuous_scale='viridis'
    )

    # Format x-axis labels as currency
    fig.update_layout(xaxis=dict(tickformat='$,.2f'))

    # Adding labels and title
    fig.update_layout(title_x=0.5, title_y=0.95)  # Adjust title_x and title_y for positioning

    return fig, top_categories_output


###################################################
# Find the costliest product and its count as compared to others
costliest_product = clean_df.groupby('Product')['Price'].mean().idxmax()

# Filter the DataFrame for the costliest product
costliest_product_data = clean_df[clean_df['Product'] == costliest_product]

# Define the layout of the app
plot3_layout = html.Div([
    html.H1(f"Count of Products in the Supermarket"),

    html.H6(f"Choose the threshold for count of products from  which you want the countplot of Products: "),

    dcc.Slider(
        id='count-threshold-slider',
        min=10,
        max=50,
        step=1,
        value=10,
        marks={i: str(i) for i in range(10, 51)},
        tooltip={'placement': 'bottom', 'always_visible': True}
    ),

    # Add a loading component
    dcc.Loading(
        id="loading-costliest-product",
        type="default",
        children=[
            dcc.Graph(id='count-plot'),
        ]
    ),

    html.Div([
        html.Pre(id='costliest-product-info')
    ]),

    # Add a download button
    html.Button("Download the clean_df csv here", id="download-button"),
    dcc.Download(id="download-costliest-plot")
])


# Define callback to update the graph and print costliest product info
@Final_Term_Project_app.callback(
    [Output('count-plot', 'figure'),
     Output('costliest-product-info', 'children'),
     Output("loading-costliest-product", "children")],
    [Input('count-plot', 'relayoutData'),
     Input('count-threshold-slider', 'value')]
)
def update_graph(relayout_data, count_threshold):
    # Filter the DataFrame based on the count threshold
    filtered_df = clean_df.groupby('Product').filter(lambda x: len(x) >= count_threshold)

    # Create a bar plot using Plotly Express
    fig = px.bar(
        filtered_df,
        x='Product',
        title=f'Count of all Products with count: {count_threshold} or more ',
        labels={'Product': 'Products', 'count': 'Count', 'Price': 'Price ($)'},
        category_orders={'Product': filtered_df['Product'].value_counts().index[::-1]},
        color='Price',
        color_continuous_scale='viridis',  # Optional: Set color scale
        color_discrete_map={col: f"${col}" for col in filtered_df['Price'].unique()}  # Add currency symbol to legend
    )

    # Limit y-axis for better readability
    fig.update_yaxes(range=[0, 50])

    # Adding labels and title
    fig.update_layout(title_x=0.5)

    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=90, tickmode='array')

    fig.update_layout(height=1000, width=1200)

    # Extracting the relevant information for display in Dash
    costliest_product_info = f"Costliest Product among the Dataset: {costliest_product}\nCount: {len(costliest_product_data)}\nAverage Price: ${costliest_product_data['Price'].mean():.2f}"

    return fig, costliest_product_info, [dcc.Graph(id='count-plot', figure=fig)]


# Callback to download the clean_df as CSV
@Final_Term_Project_app.callback(
    Output("download-costliest-plot", "data"),
    [Input("download-button", "n_clicks")],
    prevent_initial_call=True
)
def download_costliest_plot(n_clicks):
    return dcc.send_data_frame(clean_df.to_csv, "clean_df.csv")


##################################################

# Convert 'Date' column to datetime
clean_df['Date'] = pd.to_datetime(clean_df['Date'])

# Calculate the counts for each product
product_counts = clean_df['Product'].value_counts()

# Identify products with counts less than 12
products_to_replace = product_counts[product_counts < 12].index

# Replace those products with 'Others'
clean_df['Product'] = clean_df['Product'].replace(products_to_replace, 'Others')

# Filter out 'Others' from the counts and labels
product_counts = product_counts[product_counts.index != 'Others']
product_labels = product_counts.index

# Calculate the counts for each category
category_counts = clean_df['Category'].value_counts()

# Set a threshold for percentage labels
percentage_threshold = 1.0  # Display labels only for slices with percentage greater than or equal to this threshold

# Identify slices with percentage below the threshold and combine them into 'Others' slice
below_threshold = category_counts[category_counts / category_counts.sum() * 100 < percentage_threshold]
category_counts['Others'] = below_threshold.sum()
category_counts = category_counts[category_counts / category_counts.sum() * 100 >= percentage_threshold]

# Create a DataFrame with counts of products by discount category
discount_counts = clean_df['Discount'].value_counts()

# Define app layout
plot4_layout = html.Div(children=[
    html.H2('Various Pie plots from the Dataset'),

    # Dropdown for selecting pie plot
    dcc.Dropdown(
        id='pie-plot-dropdown',
        options=[
            {'label': 'Product Distribution', 'value': 'product'},
            {'label': 'Category Distribution', 'value': 'category'},
            {'label': 'Discount Distribution', 'value': 'discount'},
        ],
        value='product',  # Default value
        style={'width': '50%'}
    ),

    html.Hr(),

    # Placeholder for the selected pie plot
    dcc.Graph(id='selected-pie-plot'),

])


# Callback to update the selected pie plot based on the dropdown value
@Final_Term_Project_app.callback(
    Output('selected-pie-plot', 'figure'),
    [Input('pie-plot-dropdown', 'value')]
)
def update_pie_plot(selected_plot):
    if selected_plot == 'product':
        # Pie plot for 'Product' distribution
        fig = px.pie(clean_df[clean_df['Product'] != 'Others'], names='Product',
                     title="Distribution of Products (Counts > 11, excluding 'Others')")
    elif selected_plot == 'category':
        # Pie chart for 'Category' distribution
        fig = go.Figure(data=[go.Pie(labels=category_counts.index, values=category_counts,
                                     hoverinfo='label+percent', textinfo='label+percent', textfont_size=16,
                                     hole=0.3)])
        fig.update_layout(title=dict(
            text='Pie Chart of Category Distribution',
            x=0.5,
            y=0.95
        ))
    elif selected_plot == 'discount':
        # Pie chart for 'Discount' distribution
        fig = go.Figure(data=[go.Pie(labels=discount_counts.index, values=discount_counts,
                                     hoverinfo='label+percent', textinfo='label+percent', textfont_size=16,
                                     hole=0.3)])
        fig.update_layout(title=dict(
            text='Pie Chart of Discount Distribution',
            x=0.5,
            y=0.95
        ))
    else:
        fig = go.Figure()

    fig.update_layout(height=1000, width=1400, title_x=0.5)
    return fig


##########################################
# Assuming 'Date' is the column name in your DataFrame
new_df['Date'] = pd.to_datetime(new_df['Date'], dayfirst=True)

# Calculate mean prices for each unique date
mean_prices = new_df.groupby('Date')['Price'].mean().reset_index()

# Create a 3D scatter plot
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

scatter = go.Scatter3d(
    x=dates.date2num(mean_prices['Date']),
    y=mean_prices['Price'],
    z=mean_prices['Price'],
    mode='markers',
    marker=dict(
        size=20,
        color=mean_prices['Price'],
        colorscale='Viridis',
        opacity=0.8
    ),
    text=mean_prices['Date'].dt.strftime('%Y-%m-%d'),
    name='Mean Prices'
)

fig.add_trace(scatter)

# Adding labels and title
fig.update_layout(
    scene=dict(
        xaxis_title='Date',
        yaxis_title='Mean Price ($)',
        zaxis_title='Mean Price ($)'
    ),
    title=dict(
        text='3D Scatter Plot of Mean Prices Over Time',
        x=0.5,  # Set x to 0.5 to center the title horizontally
        xanchor='center'  # Center the title horizontally
    )
)

# Format y-axis labels as currency
fig.update_layout(scene=dict(yaxis=dict(tickformat='$,.2f')))

# Improve the formatting of date labels
fig.update_xaxes(
    tickvals=dates.date2num(mean_prices['Date']),
    ticktext=mean_prices['Date'].dt.strftime('%Y-%m-%d'),
    tickangle=45
)

fig.update_layout(height=800, width=1200)

# App layout
plot5_layout = html.Div(children=[

    # Output for displaying the plot based on user input
    html.Div(
        id='output-plot',
        children=[
            # 3D Scatter plot
            html.H3('3D plot from the Dataset'),
            dcc.Graph(
                id='scatter-3d-plot',
                figure=fig
            ),
        ]
    ),

    # Input for user to choose plot type
    dcc.Input(
        id='plot-type-input',
        type='text',
        placeholder='Enter plot type (e.g., 3D)',
        value=''
    ),
])


# Callback to update the visibility of the 3D plot based on user input
@Final_Term_Project_app.callback(
    Output('scatter-3d-plot', 'style'),
    [Input('plot-type-input', 'value')]
)
def update_plot_visibility(plot_type):
    # Display the 3D plot if user input is '3D', otherwise hide it
    return {'display': 'block'} if plot_type.lower() == '3d' else {'display': 'none'}


#########################################
# Assuming 'Date' is the column name in your DataFrame
clean_df['Date'] = pd.to_datetime(clean_df['Date'], dayfirst=True)

# Define the layout
plot6_layout = html.Div([
    html.H1('Area Plot of Various Categories Over Time'),

    html.H6("Choose various categories for the area plot: "),

    # Checklist for selecting categories
    dcc.Checklist(
        id='category-checklist',
        options=[
            {'label': 'Sweets', 'value': 'Sweets'},
            {'label': 'Household products', 'value': 'Household products'},
            {'label': 'Vodka', 'value': 'Vodka'},
            {'label': 'Fruit juices', 'value': 'Fruit juices'},
            {'label': 'Dishes', 'value': 'Dishes'},
            {'label': 'Spices', 'value': 'Spices'},
            {'label': 'Tea', 'value': 'Tea'}
        ],
        value=['Fruit juices'],  # Set default value to Fruit juices
        labelStyle={'display': 'block'}
    ),
    html.Br(),
    html.H6("Choose from which month to what month you want to see the Area plot over time:"),
    # Range slider for months
    dcc.RangeSlider(
        id='month-slider',
        marks={i: month for i, month in
               enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])},
        min=0,
        max=11,
        step=1,
        value=[0, 2]  # Set default months to Jan to Mar
    ),
    html.Br(),

    # Area plot
    dcc.Graph(id='area-plot')
])


# Define callback to update the area plot based on selected categories and months
@Final_Term_Project_app.callback(
    Output('area-plot', 'figure'),
    [Input('category-checklist', 'value'),
     Input('month-slider', 'value')]
)
def update_area_plot(selected_categories, selected_months):
    # Filter data based on selected categories and months
    filtered_data = clean_df[clean_df['Category'].isin(selected_categories)]
    filtered_data = filtered_data[(filtered_data['Date'].dt.month >= selected_months[0] + 1) & (
                filtered_data['Date'].dt.month <= selected_months[1] + 1)]

    # Group by 'Date' and 'Category', and calculate the sum of prices
    grouped_data = filtered_data.groupby(['Date', 'Category'])['Price'].sum().reset_index()

    # Create traces for each category
    traces = []
    for category in selected_categories:
        category_data = grouped_data[grouped_data['Category'] == category]
        trace = go.Scatter(
            x=category_data['Date'],
            y=category_data['Price'],
            mode='lines',
            stackgroup='one',
            name=category
        )
        traces.append(trace)

    # Layout settings
    layout = go.Layout(
        title='Area Plot of Various Categories Over Time',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Total Price ($)', tickformat='$,.2f', range=[0, 17]),
        # Format y-axis labels as currency and set the range
        showlegend=True
    )

    # Create the figure
    figure = go.Figure(data=traces, layout=layout)

    return figure


########################################


# Reading the dataset from the csv file
sub_df = pd.read_csv("esas.csv")
# Renaming column Names
sub_df = sub_df.rename(
    columns={'Unnamed: 0': 'ID', 'satish_kodu': 'Receipt_Number', 'mehsul_kodu': 'Product_Code', 'mehsul_ad': 'Product',
             'mehsul_kateqoriya': 'Category', 'mehsul_qiymet': 'Price', 'satish_tarixi': 'Date',
             'endirim_kompaniya': 'Discount', 'bonus_kart': 'Bonus_cart', 'magaza_ad': 'Store_Branch',
             'magaza_lat': 'Store_Lat', 'magaza_long': 'Store_Long'})
# droping the na values in df and checking the if they are gone
sub_df = sub_df.dropna()
# dropping unescessay columns
sub_df = sub_df.drop('ID', axis=1)
# reseting index
sub_df = sub_df.reset_index(drop=True)
# checking for duplicate rows and dropping them
duplicated_rows = clean_df.duplicated()
clean_df = clean_df.drop_duplicates()
# Coverting type of some coulmns
sub_df = sub_df.copy()  # Create a copy of the DataFrame
sub_df['Receipt_Number'] = sub_df['Receipt_Number'].astype('int')
sub_df['Product'] = sub_df['Product'].astype('category')
sub_df.loc[:, 'Date'] = pd.to_datetime(sub_df['Date'], dayfirst=True)
sub_df['Category'] = sub_df['Category'].astype('category')
sub_df['Discount'] = sub_df['Discount'].astype('category')
sub_df['Store_Branch'] = sub_df['Store_Branch'].astype('category')
# Replace multiple values in the 'Discount' column
replace_dict = {'Sərin Yay günləri': 'Summer Season', 'S?rf?li Yaz': 'Fifth of May Season',
                'Bərəkətli Novruz': 'Happy Nowruz Season', 'Yeni il fürsətləri': 'New Year Season',
                'Payız endirimləri': 'Autumn Season'}
sub_df['Discount'] = sub_df['Discount'].replace(replace_dict)
# Translate values in the 'Category' and 'Store Branch' column
from googletrans import Translator

translator = Translator()
sub_df['Category'] = sub_df['Category'].apply(lambda x: translator.translate(x).text)
sub_df['Store_Branch'] = sub_df['Store_Branch'].apply(lambda x: translator.translate(x).text)

from sklearn.preprocessing import StandardScaler

# Assuming 'clean_df' is your DataFrame
scaler = StandardScaler()
numerical_data = sub_df.select_dtypes(include=[np.number])  # Select numerical columns only
standardized_data = scaler.fit_transform(numerical_data)
correlation_matrix = pd.DataFrame(standardized_data).corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f")

from sklearn.decomposition import PCA

pca = PCA(svd_solver="full", n_components=0.95, random_state=5764)
pca_data = pca.fit_transform(standardized_data)
exp_var_pca = pca.explained_variance_ratio_
pca_data_df = pd.DataFrame(pca_data, columns=[f'PC_{i}' for i in range(1, pca_data.shape[1] + 1)])
cum_sum_eigenvalues = np.cumsum(exp_var_pca) * 100
index_95_variance = np.argmax(cum_sum_eigenvalues >= 95)

Q1_price = clean_df['Price'].quantile(0.25)
Q3_price = clean_df['Price'].quantile(0.75)
IQR_price = Q3_price - Q1_price
# Identify outliers for 'Price'
lower_bound_price = Q1_price - 1.5 * IQR_price
upper_bound_price = Q3_price + 1.5 * IQR_price
# Remove outliers from the dataset
cleaned_df_price = clean_df[(clean_df['Price'] >= lower_bound_price) & (clean_df['Price'] <= upper_bound_price)]

from scipy.stats import kstest


def ks_test(x, title):
    mean = np.mean(x)
    std = np.std(x)
    dist = np.random.normal(mean, std, len(x))
    stats, p = kstest(x, dist)
    return stats, p


from scipy.stats import shapiro


def shapiro_test(x):
    stats, p = shapiro(x)
    return stats, p


scaler = StandardScaler()
standard_Price = scaler.fit_transform(clean_df[['Price']])

from scipy.stats import normaltest


def da_k_squared_test(x):
    stats, p = normaltest(x)
    return stats, p


# Define app layout
plot7_layout = html.Div(children=[
    html.H2('Analysis Dashboard'),

    html.H6("Choose what kind of plot you want to see: "),

    # Dropdown for selecting analysis type
    dcc.Dropdown(
        id='analysis-dropdown',
        options=[
            {'label': 'Correlation Heatmap', 'value': 'correlation'},
            {'label': 'PCA Analysis', 'value': 'pca'},
            {'label': 'Boxplot of Price', 'value': 'boxplot_price'},
            {'label': 'Boxplot of Cleaned Price', 'value': 'boxplot_cleaned_price'},
            {'label': 'K-S Test for Price', 'value': 'ks_test_price'},
            {'label': 'Shapiro-Wilk Test for Price', 'value': 'shapiro_test_price'},
            {'label': 'D Agostino\'s K^2 test for Price', 'value': 'da_k_squared_test_price'},
        ],
        value='correlation',  # Default value
        style={'width': '50%'}
    ),

    # Placeholder for the selected analysis plot
    dcc.Graph(id='selected-analysis-plot'),

])


# PCA Analysis
def generate_pca_analysis():
    # Assuming 'cum_sum_eigenvalues' and 'index_95_variance' are available
    x_values = list(range(1, len(cum_sum_eigenvalues) + 1))

    # Create the plot
    fig = go.Figure()

    # Cumulative Explained Variance
    fig.add_trace(go.Scatter(
        x=x_values,
        y=cum_sum_eigenvalues,
        mode='lines+markers',
        name='Cumulative Explained Variance',
        line=dict(color='blue', width=2)
    ))

    # 95% Explained Variance Line
    fig.add_shape(
        go.layout.Shape(
            type='line',
            x0=min(x_values),
            x1=max(x_values),
            y0=95,
            y1=95,
            line=dict(color='black', dash='dash'),
            name='95% Explained Variance'
        )
    )

    # Optimal Features Line
    fig.add_shape(
        go.layout.Shape(
            type='line',
            x0=index_95_variance + 1,
            x1=index_95_variance + 1,
            y0=min(cum_sum_eigenvalues),
            y1=max(cum_sum_eigenvalues),
            line=dict(color='red', dash='dash'),
            name=f'Optimal Features: {index_95_variance + 1}'
        )
    )

    # Layout settings
    fig.update_layout(
        title='Cumulative Explained Variance vs. Number of Components',
        xaxis=dict(title='Number of Components'),
        yaxis=dict(title='Cumulative Explained Variance (%)'),
        legend=dict(x=0.7, y=0.9),
        height=600,
        width=1000
    )

    return fig


# Boxplot of 'Price'
def generate_boxplot_price():
    fig = px.box(sub_df['Price'], x='Price', orientation='h', title='Boxplot of Price with Outliers')
    return fig


# Correlation Heatmap
def generate_correlation_heatmap():
    # Round the correlation matrix values to 2 decimal points
    correlation_matrix_rounded = correlation_matrix.round(2)

    # Create a list of lists to display values in annotations
    z_text = [['' if np.isnan(value) else f'{value:.2f}' for value in row] for row in correlation_matrix_rounded.values]

    # Create the heatmap with custom colorscale
    fig = px.imshow(correlation_matrix_rounded, labels=dict(x="Features", y="Features", color="Correlation"),
                    color_continuous_scale='Viridis')

    # Add text annotations
    fig.update_layout(annotations=[
        dict(x=i, y=j, text=z_text[i][j], showarrow=False) for i in range(len(correlation_matrix_rounded.columns))
        for j in range(len(correlation_matrix_rounded.columns))
    ])

    # Set layout title
    fig.update_layout(title='Correlation Coefficient Matrix Heatmap')

    return fig


# K-S Test for 'Price'
def generate_ks_test_price():
    # Assuming 'sub_df' is your DataFrame
    stats, p = ks_test(sub_df['Price'], 'Raw')

    # Display K-S test results for 'Price'
    result_text = "Not Normal" if p < 0.01 else "Normal"

    # Create a table with results
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Statistic', 'P-Value', 'Result']),
        cells=dict(values=[stats.round(2), p.round(2), result_text])
    )])

    # Set layout title
    fig.update_layout(title='K-S Test for Price')

    return fig


# Shapiro-Wilk Test for 'Price'
def generate_shapiro_test_price():
    # Assuming 'cleaned_df_price' is your cleaned DataFrame for 'Price'
    stat_price, p_value_price = shapiro_test(sub_df['Price'])

    # Convert 'stat_price' and 'p_value_price' to rounded strings
    stat_price_str = f"{stat_price:.2f}" if isinstance(stat_price, (int, float)) else ""
    p_value_price_str = f"{p_value_price:.2f}" if isinstance(p_value_price, (int, float)) else ""

    # Interpret the Shapiro-Wilk test result for 'Price'
    result_price = "Not Normal" if p_value_price < 0.01 else "Normal"

    # Create a table with results
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Statistic', 'P-Value', 'Result']),
        cells=dict(values=[stat_price_str, p_value_price_str, result_price])
    )])

    # Set layout title
    fig.update_layout(title='Shapiro-Wilk Test for Price')

    return fig


def da_k_squared_test_price():
    stats, p = da_k_squared_test(sub_df['Price'])

    # Interpret the Shapiro-Wilk test result for standardized 'Price'
    result_std_price = "Not Normal" if p < 0.01 else "Normal"

    # Create a table with results
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Statistic', 'P-Value', 'Result']),
        cells=dict(values=[stats.round(2), p.round(2), result_std_price])
    )])

    # Set layout title
    fig.update_layout(title=' D Agostino\'s K^2 test for Price')

    return fig


def generate_boxplot_cleaned_price():
    fig = px.box(cleaned_df_price, x='Price', orientation='h', title='Boxplot of Cleaned Price')
    return fig


# Callback to update the selected analysis plot based on the dropdown value
@Final_Term_Project_app.callback(
    Output('selected-analysis-plot', 'figure'),
    [Input('analysis-dropdown', 'value')]
)
def update_analysis_plot(selected_analysis):
    if selected_analysis == 'correlation':
        return generate_correlation_heatmap()
    elif selected_analysis == 'pca':
        return generate_pca_analysis()
    elif selected_analysis == 'ks_test_price':
        return generate_ks_test_price()
    elif selected_analysis == 'boxplot_price':
        return generate_boxplot_price()
    elif selected_analysis == 'boxplot_cleaned_price':
        return generate_boxplot_cleaned_price()
    elif selected_analysis == 'shapiro_test_price':
        return generate_shapiro_test_price()
    elif selected_analysis == 'da_k_squared_test_price':
        return da_k_squared_test_price()
    else:
        return go.Figure()


########################################
# Main callback
@Final_Term_Project_app.callback(
    Output("tabs-content", "children"),
    [Input("tabs", "value")]
)
def render_content(tab):
    if tab == "about":
        return about_layout
    elif tab == "p1":
        return plot1_layout
    elif tab == "p2":
        return plot2_layout
    elif tab == "p3":
        return plot3_layout
    elif tab == "p4":
        return plot4_layout
    elif tab == "p5":
        return plot5_layout
    elif tab == "p6":
        return plot6_layout
    elif tab == "p7":
        return plot7_layout


if __name__ == '__main__':
    Final_Term_Project_app.run_server(debug=False, host='0.0.0.0', port=8051)
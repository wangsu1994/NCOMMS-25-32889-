import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import netCDF4 as nc
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import numpy.ma as ma
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from shutil import copyfile
from datetime import datetime
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import pdb
from sklearn.metrics import silhouette_score

nc_file = nc.Dataset('ERA5.nc', 'r')
output_folder = './Categorized_blue_201803_wea_pm_gmm_4type2023_deletp_2non_obs+era5+2aqi'
nc_file_path = "./RGB.nc"
excel_file_path = r'AQI.csv'
n_clusters =4
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# Function to create a histogram of category proportions per month
def create_category_histogram(classification_labels, date_sel, output_folder):
    # Convert date strings to datetime objects
    date_sel = [datetime.strptime(date, '%Y%m%d') for date in date_sel]
    
    months = [date.month for date in date_sel]
    unique_labels = np.unique(classification_labels)
    plt.figure(figsize=(12, 6))
    cluster_name=["Deep","Medium","Light","Non"]
    for label in unique_labels:
        label_count = [classification_labels[i] for i in range(len(classification_labels)) if classification_labels[i] == label]
        label_months = [months[i] for i in range(len(classification_labels)) if classification_labels[i] == label]
        histogram_data = [label_months.count(month) / months.count(month) for month in range(1, 13)]
        plt.plot(range(1, 13), histogram_data, label=cluster_name)

    plt.xlabel('Month')
    plt.ylabel('Proportion')
    plt.title('Category Proportions Per Month')
    plt.xticks(range(1, 13))
    plt.legend()
    plt.grid(True)

    histogram_path = os.path.join(output_folder, 'category_histogram.png')
    plt.savefig(histogram_path)

lcc = nc_file.variables['lcc'][:]
tcc = nc_file.variables['tcc'][:]
# tp[tp > 0.] = 32766
u10 = nc_file.variables['u10'][:]
v10 = nc_file.variables['v10'][:]
blh = nc_file.variables['blh'][:]

# Extract variables with two dimensions (time, level)
w1 = nc_file.variables['w'][:,0]
w2 = nc_file.variables['w'][:,1]
w3 = nc_file.variables['w'][:,2]
r1 = nc_file.variables['r'][:,0]
r2 = nc_file.variables['r'][:,1]
r3 = nc_file.variables['r'][:,2]
d1 = nc_file.variables['d'][:,0]
d2 = nc_file.variables['d'][:,1]
d3 = nc_file.variables['d'][:,2]

nc_file = r'Meteo_Obs.csv'
# Read the CSV file into a DataFrame
meteo_sta = pd.read_csv(nc_file, index_col=0, parse_dates=True)

# Extract variables with one dimension (time)
tp = np.array(meteo_sta['PRE_1h'])
t2m = np.array(meteo_sta['tem'])
wd= np.array(meteo_sta['WD'])
ws = np.array(meteo_sta['WS'])
vis = np.array(meteo_sta['vis']*10e-3)

tpo=tp
# r1[r1 > 100.] = 100
tp[tp < 0.] = np.nan
tp[tp < 0.] = tp.mean()
print(max(tp))

# tp[tp > 1] = 1
# tp[tp <= 1] = 0
variables_c = [lcc, tcc, t2m, wd,ws,tp,d1, d2,d3,r1, r2,r3, w1,w2,w3,u10, v10]
variable_names_c = ["lcc", "tcc", "t2m", "WD","WS",'tp', "d1",  "d2", "d3", "rh1", "rh2", "rh3",'w1','w2','w3',"u10", "v10"]
variables_c = [np.where(var <= -32760, var.mean(), var) for var in variables_c]
variables_c = [np.where(var > 32760, var.mean(), var) for var in variables_c]
for i in range(len(variables_c)):
    print(variable_names_c[i],max(variables_c[i].flatten()),min(variables_c[i].flatten()))

# Convert the list of variables to a 2D NumPy array
data = np.array(variables_c).T  # Transpose to have variables as columns

# Use StandardScaler for normalization
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

# Transpose the normalized data back to the original shape
normalized_variables_c = normalized_data.T

# Create a DataFrame with the 'variables' list as columns for correlation_c
correlation_c = pd.DataFrame()

for i in range(len(variables_c)):
    if isinstance(variables_c[i], np.ndarray):
        correlation_c[variable_names_c[i]] = normalized_variables_c[i].flatten()
        # print('c= ',max(normalized_variables_c[i]))
    else:
        correlation_c[variable_names_c[i]] = normalized_variables_c[i]
        # print('c= ',max(normalized_variables_c[i]))



# Read the Excel file into a DataFrame, skipping any non-numeric rows (e.g., the picture)
filled_daily_average_extended = pd.read_csv(excel_file_path)
pm25_column = filled_daily_average_extended['pm25']
pm10_column = filled_daily_average_extended['pm10']
o3_column = filled_daily_average_extended['o3']
o3_column = filled_daily_average_extended['o3']
so2_column = filled_daily_average_extended['so2']
no2_column = filled_daily_average_extended['no2']
co_column = filled_daily_average_extended['co']
date_range = pd.date_range(start='2013-01-01 00:00:00', periods=pm25_column.shape[0], freq='D')
print(date_range)


pm25_column.loc[pm25_column == -999] = np.nan
pm10_column.loc[pm10_column == -999] = np.nan
o3_column.loc[o3_column == -999] = np.nan

pm25o = pm25_column.copy()
pm10o = pm10_column.copy()
o3o = o3_column.copy()
so2o = so2_column.copy()
no2o = no2_column.copy()
coo = co_column.copy()

pm25o.loc[pm25o == -999] = pm25o.mean()
pm10o.loc[pm10o == -999] = pm10o.mean()
o3o.loc[o3o == -999] = o3o.mean()
pm25o.index = date_range
pm10o.index = date_range
o3o.index = date_range
pm25o = np.array(pm25o)
pm10o = np.array(pm10o)
o3o = np.array(o3o)
so2o = np.array(so2o)
no2o = np.array(no2o)
coo = np.array(coo)

pm25= scaler.fit_transform(pm25_column.values.reshape(-1, 1))
pm10= scaler.fit_transform(pm10_column.values.reshape(-1, 1))
o3= scaler.fit_transform(o3_column.values.reshape(-1, 1))
so2= scaler.fit_transform(so2_column.values.reshape(-1, 1))
no2= scaler.fit_transform(no2_column.values.reshape(-1, 1))
co= scaler.fit_transform(co_column.values.reshape(-1, 1))

variables_p = [pm25, pm10]
variable_names_p = ["pm25", "pm10"]
# variables_p = [pm25, pm10,o3]
# variable_names_p = ["pm25", "pm10","o3"]
variables_p = [np.where(np.isnan(var), np.nanmean(var), var) for var in variables_p]
# Replace negative values with the mean for each variable
variables_p = [np.where(var < 0, np.nanmean(var), var) for var in variables_p]
# Create a DataFrame with the 'variables' list as columns for correlation_p
correlation_p = pd.DataFrame()

for i in range(len(variables_p)):
    if isinstance(variables_p[i], np.ndarray):
        correlation_p[variable_names_p[i]] = variables_p[i].flatten()
        # print('c= ',max(variables_p[i]))
    else:
        correlation_p[variable_names_p[i]] = variables_p[i]
        # print('p= ',max(variables_p[i]))

# Load data from the NetCDF file------------------------------======================RGB===============
ds = xr.open_dataset(nc_file_path)
# Extract the "blue" variable and replace 0s with a default value
blue = ds["blue"].values
blue[blue <= 0] = np.nan
blueo=blue
blue_normalized= scaler.fit_transform(blue.reshape(-1, 1))
print(blue)

#333333333333333333333333333-----------------------------------------------------------------------------
# Split the "blue" variable into two parts
split_index = 1885

#Normalize the first 1885 data points inwidually
blue_first_part = blue[:split_index]
scaler_first_part = MinMaxScaler()
blue_first_normalized = scaler_first_part.fit_transform(blue_first_part.reshape(-1, 1))

# Normalize the remaining data points inwidually
blue_second_part = blue[split_index:]
scaler_second_part = MinMaxScaler()
blue_second_normalized = scaler_second_part.fit_transform(blue_second_part.reshape(-1, 1))
# Concatenate the two normalized parts to obtain the final "blue_normalized"
blue_normalized1 = np.concatenate((blue_first_normalized, blue_second_normalized))

blue_normalized = pd.DataFrame(data=blue_normalized1, index=range(len(blue_normalized)), columns=['blue'])


blue_not_nan_mask = ~np.isnan(blue_normalized['blue'].values) & ~np.isnan(so2.flatten())
print(correlation_c.shape)
print(correlation_p.shape)
print(blue_normalized.shape)

sumc = correlation_c.loc[blue_not_nan_mask].mean(axis=1,numeric_only=True)
sump = correlation_p.loc[blue_not_nan_mask].mean(axis=1,numeric_only=True)
date_sel = date_range[blue_not_nan_mask]
# Initialize a dictionary to store the SEL values for each category
sel_values = pd.DataFrame()
 
blue=blue_normalized['blue'][blue_not_nan_mask]
# print('blue===========')

blue_not = np.isnan(blue_normalized['blue'].values)
blued=date_range[blue_not]
# print(sumc.size ,blue.size)

###########################################################################333
datao = sumc+sump+blue#+np.array(tp).flatten()[blue_not_nan_mask]
# Create a mask to identify rows with NaN values in sumc, sump, or blue
###########################################################################################3
# Create a mask to identify rows with tp < 0.1
tpp=tp[blue_not_nan_mask]
tp_mask = (np.array(tpp) <= 0.1).flatten()
print(len(tp_mask))

# Convert the boolean mask to boolean type
nan_mask1 = np.isnan(sumc) | np.isnan(sump) | np.isnan(blue)
# Convert the boolean mask to boolean type
nan_mask = np.isnan(sumc) | np.isnan(sump) | np.isnan(blue) | ~tp_mask
# Apply the mask to data to remove rows with NaN values in any of the variables
data = datao[~nan_mask]
data2 = datao[~nan_mask1]
data_not = np.isnan(data)
date_sel1 = date_sel[~nan_mask1].strftime('%Y%m%d')
date_sel = date_sel[~nan_mask].strftime('%Y%m%d')


print(date_sel1)

data = np.array(data).reshape(-1, 1)

bluee=blue[~nan_mask]
# Path to your image folder and the output folder
# print(bluee)

image_folder = './Cut'
# pdb.set_trace() # 设置追踪断点
import shutil
image_paths = [os.path.join(image_folder, f'{date}.jpg') for date in date_sel1]

# List all image files in the folder
image_files = [f for f in  image_paths if f.endswith('.jpg')]


# Fit a Gaussian Mixture Model to the blue channel intensity data
gmm = GaussianMixture(n_components=n_clusters, random_state=0)
gmm.fit(data)

# Predict labels using the GMM model
classification_labels = gmm.predict(data)

silhouette_avg = silhouette_score(data, classification_labels)
print(f"Silhouette Score: {silhouette_avg}")
# Define the file path
file_path = output_folder+'silhouette_avg.txt'

# Write the silhouette_avg to the file
with open(file_path, 'w') as f:
    f.write(f"Silhouette Score: {silhouette_avg}\n")
    
plt.figure(figsize=(8, 6))
plt.hist(classification_labels, bins=n_clusters, color='skyblue', edgecolor='black')
plt.xlabel('Cluster Label')
plt.ylabel('Count')
plt.title('Distribution of Clusters')
# plt.show()

bluee=bluee.values
# pdb.set_trace() 
# Ca#lculate the average blue channel value for each category
category_blue_averages = []
for i in range(n_clusters):
    # Extract blue channel values for the current category
    blue_values = [bluee[j] for j in range(len(bluee)) if classification_labels[j] == i]
    # Calculate the average blue channel value for this category
    category_average = np.mean(blue_values)
    category_blue_averages.append((i, category_average))

# Sort the categories based on the average blue channel values
sorted_categories = sorted(category_blue_averages, key=lambda x: x[1])

# Create a dictionary to map old labels to new ranks
category_rank_mapping = {category[0]: rank  for rank, category in enumerate(sorted_categories)}

# Replace the classification_labels with the new ranks
classification_labels = [category_rank_mapping[label] for label in classification_labels]
 

# Add this part of the code to create classification_labels2
classification_labels2 = np.zeros(len(date_sel1), dtype=int)

# Loop through the dates where tp_mask is True and set classification_labels2 to 4
for i, date in enumerate(date_sel1):
    if date in date_sel:  # Check if the date is present in date_sel
        index_in_date_sel = np.where(date_sel == date)[0][0]
        classification_labels2[i] = classification_labels[index_in_date_sel]
    else:
        classification_labels2[i] = n_clusters-1  # Set to 3 for Category_4
classification_labelso=classification_labels
classification_labels=classification_labels2
 
# Create folders for each category if they don't exist
for classification_type in range(n_clusters):
    cluster_folder = os.path.join(output_folder, f'Category_{classification_type + 1}')
    if os.path.exists(cluster_folder):
        shutil.rmtree(cluster_folder)#删除再建立
    os.makedirs(cluster_folder, exist_ok=True)

# Move images to their respective folders based on the GMM classification
for i, image_files in enumerate(date_sel1):
    # (filepath, image_file) = os.path.split(image_files)
    image_file=image_files+".jpg"
    # print(image_file)
    classification_type = classification_labels[i]  
    cluster_folder = os.path.join(output_folder, f'Category_{classification_type+1}')
    output_path = os.path.join(cluster_folder, image_file)

    # Ensure the output folder exists before moving the image
    os.makedirs(cluster_folder, exist_ok=True)

    copyfile(os.path.join(image_folder, image_file), output_path)
print("well=========")
# # Create a scatter plot to visualize the results
# plt.figure(figsize=(10, 6))
# colors = ['yellow','red', 'green', 'blue', 'purple']


variables_sel = [("LCC", lcc), ("TCC", tcc), ("PM25", pm25o), ("PM10", pm10o), ("vis", vis), ("B value", blueo)]
box_data_df = {}
# print(variables_sel)
for i, (var_name, variable) in enumerate(variables_sel):
    variable = variable.flatten().reshape(-1, 1)  # Flatten 并 reshape 成 2D 以适应 MinMaxScaler
    normalized_data = scaler.fit_transform(variable)
    # print('max=',max(normalized_data))
    # print('min=',min(normalized_data))
    var_data = pd.DataFrame({var_name: normalized_data.flatten()}, index=date_range)
    blue_channel_values = var_data.loc[date_sel1, var_name].values.flatten()
    
    # 将每个变量的数据和类别标签创建为DataFrame
    box_data_df[var_name] = pd.DataFrame({'Values': blue_channel_values, 'Category': classification_labels})

# 确认变量名和实际数据一致
lccd = box_data_df['LCC']['Values']
tccd= box_data_df['TCC']['Values']
pm25d = box_data_df['PM25']['Values']
pm10d = box_data_df['PM10']['Values']
visd = box_data_df['vis']['Values']

def calculate_medians(values, categories):
    df = pd.DataFrame({'Values': values, 'Category': categories})
    medians = df.groupby('Category').agg({'Values': 'median'})
    return medians

# 计算变量中位数
medians_lcc = calculate_medians(lccd, classification_labels)
medians_tcc = calculate_medians(tccd, classification_labels)
medians_pm25 = calculate_medians(pm25d, classification_labels)
medians_pm10 = calculate_medians(pm10d, classification_labels)
medians_vis = calculate_medians(visd, classification_labels)


import numpy as np
import pandas as pd
from scipy.stats import pearsonr, f_oneway
from sklearn.preprocessing import MinMaxScaler


# --- 1. 趋势偏离分析得分调整 ---
def calculate_trend_deviation(medians, trend='increasing'):
    values = medians.sort_index().values.flatten()
    diffs = np.diff(values)
    if trend == 'increasing':
        deviations = np.where(diffs >= 0, 0, -diffs)
    else:
        deviations = np.where(diffs <= 0, 0, diffs)
    return np.sum(deviations)

def trend_deviation_score(medians_lcc, medians_tcc, medians_vis, medians_pm25, medians_pm10):
    lcc_dev = calculate_trend_deviation(medians_lcc, trend='increasing')
    tcc_dev = calculate_trend_deviation(medians_tcc, trend='increasing')
    vis_dev = calculate_trend_deviation(medians_vis, trend='decreasing')
    pm25_dev = calculate_trend_deviation(medians_pm25, trend='increasing')
    pm10_dev = calculate_trend_deviation(medians_pm10, trend='increasing')
    
    total_deviation = lcc_dev + tcc_dev + vis_dev + pm25_dev + pm10_dev

    # 为了将趋势偏离得分标准化到 [0,1]，并且越高越好，我们使用指数衰减函数
    trend_score = np.exp(-total_deviation)
    return trend_score  # 趋势偏离越小，trend_score 越接近1

# --- 2. 时序相关性得分调整 ---
def correlation_score(lccd, tccd, pm25d, pm10d, visd, classification_labels):
    # 计算相关系数
    corr_lcc, _ = pearsonr(lccd, classification_labels)
    corr_tcc, _ = pearsonr(tccd, classification_labels)
    corr_pm25, _ = pearsonr(pm25d, classification_labels)
    corr_pm10, _ = pearsonr(pm10d, classification_labels)
    corr_vis, _ = pearsonr(visd, classification_labels)
    
    # 调整相关系数方向，使得期望的方向为正
    corr_vis = -corr_vis  # 因为能见度与分类等级期望负相关
    
    # 将相关系数标准化到 [0,1]
    corr_lcc_score = (corr_lcc + 1) / 2
    corr_tcc_score = (corr_tcc + 1) / 2
    corr_pm25_score = (corr_pm25 + 1) / 2
    corr_pm10_score = (corr_pm10 + 1) / 2
    corr_vis_score = (corr_vis + 1) / 2

    # 确保相关性为负的变量得分不为负数
    corr_scores = [max(0, score) for score in [corr_lcc_score, corr_tcc_score, corr_pm25_score, corr_pm10_score, corr_vis_score]]
    
    # 计算平均相关性得分
    total_correlation_score = np.mean(corr_scores)
    return total_correlation_score  # 越高越好，范围在 [0,1]

# --- 3. 分类评估的统计检验得分调整 ---
def statistical_test_score(lccd, tccd, pm25d, pm10d, visd, classification_labels):
    # 对5个变量分别进行ANOVA
    _, p_value_lcc = f_oneway(lccd[classification_labels == 0], lccd[classification_labels == 1],
                              lccd[classification_labels == 2], lccd[classification_labels == 3])
    
    _, p_value_tcc = f_oneway(tccd[classification_labels == 0], tccd[classification_labels == 1],
                              tccd[classification_labels == 2], tccd[classification_labels == 3])
    
    _, p_value_pm25 = f_oneway(pm25d[classification_labels == 0], pm25d[classification_labels == 1],
                               pm25d[classification_labels == 2], pm25d[classification_labels == 3])
    
    _, p_value_pm10 = f_oneway(pm10d[classification_labels == 0], pm10d[classification_labels == 1],
                               pm10d[classification_labels == 2], pm10d[classification_labels == 3])
    
    _, p_value_vis = f_oneway(visd[classification_labels == 0], visd[classification_labels == 1],
                              visd[classification_labels == 2], visd[classification_labels == 3])
    
    # 将p值转换为得分，使用 1 - (p_value / alpha)，并限制在 [0,1]
    alpha = 0.05  # 显著性水平
    def p_value_to_score(p_value):
        score = 1 - (p_value / alpha)
        return max(0, min(score, 1))  # 限制在 [0,1]
    
    stat_score_lcc = p_value_to_score(p_value_lcc)
    stat_score_tcc = p_value_to_score(p_value_tcc)
    stat_score_pm25 = p_value_to_score(p_value_pm25)
    stat_score_pm10 = p_value_to_score(p_value_pm10)
    stat_score_vis = p_value_to_score(p_value_vis)
    
    # 计算平均统计检验得分
    total_statistical_score = np.mean([stat_score_lcc, stat_score_tcc, stat_score_pm25, stat_score_pm10, stat_score_vis])
    return total_statistical_score  # 越高越好，范围在 [0,1]

# --- 4. 综合评估 ---
def evaluate_classification(medians_lcc, medians_tcc, medians_vis, medians_pm25, medians_pm10,
                            lccd, tccd, pm25d, pm10d, visd, classification_labels):
    
    # 1. 趋势偏离得分
    trend_score = trend_deviation_score(medians_lcc, medians_tcc, medians_vis, medians_pm25, medians_pm10)
    print(f"Trend Deviation Score: {trend_score:.4f}")
    
    # 2. 时序相关性得分
    correlation_score_value = correlation_score(lccd, tccd, pm25d, pm10d, visd, classification_labels)
    print(f"Correlation Score: {correlation_score_value:.4f}")
    
    # 3. 统计检验得分
    statistical_score = statistical_test_score(lccd, tccd, pm25d, pm10d, visd, classification_labels)
    print(f"Statistical Test Score: {statistical_score:.4f}")

    # 将三部分得分按 1:1:1 比例加权平均
    final_score = (trend_score + correlation_score_value + statistical_score) / 3
    
    return final_score

# 调用评估函数计算最终评分
final_classification_score = evaluate_classification(medians_lcc, medians_tcc, medians_vis, medians_pm25, medians_pm10,
                                                    lccd, tccd, pm25d, pm10d, visd, classification_labels)

print(f"Final Classification Score: {final_classification_score:.4f}")

# 保存结果到文件
file_path = os.path.join(output_folder, 'final_classification_score.txt')
with open(file_path, 'w') as f:
    f.write(f"Final Classification Score: {final_classification_score:.4f}\n")

    
num_rows = 3
num_columns = 8

# Create subplots to display all the probability density distributions together in a 3x3 grid
fig, axes = plt.subplots(num_rows, num_columns, figsize=(20, 22))
gs = gridspec.GridSpec(num_rows, num_columns, figure=fig)
cluster_name=["Deep","Medium","Light","Non"]
variables =[("LCC", lcc), ("TCC", tcc),("t2m", t2m),("BLH", blh),("Precipitaion", tp), ("U10", u10),("V10", v10),("WD", wd),("WS", ws),
            ("Div500", d1),("Div850", d2),("Div975", d3), ("w500", w1),("w850", w2),("w975", w3),
            ("RH", r3), 
            ("PM25", pm25o),("PM10", pm10o),("O3", o3o),("vis", vis),("B value",blueo)]

colors =['#07689F','#40A8C4','#A2D5F2','#9CB0C3']#['#F47F72', '#8DD2C5', '#BFBCDA', '#7FB2D5']



specific_colors =colors# ['lightcoral',  'navajowhite', 'paleturquoise','darkgray']
all_var_means = []


# Create a new figure for the box plots of all 22 variables
fig_box, axes_box = plt.subplots(num_rows, num_columns, figsize=(30,15))
gs = gridspec.GridSpec(num_rows, num_columns, figure=fig_box)
for i, (var_name, variable) in enumerate(variables):
    ax_box = plt.subplot(gs[i // num_columns, i % num_columns])
    # print(type(variable))
    # print(var_name)
    var_data = pd.DataFrame({var_name: variable.flatten()}, index=date_range)
    blue_channel_values = var_data.loc[date_sel1, var_name].values.flatten()
    
    # Create a DataFrame with the data and category labels
    # 
    # violin_data_df = pd.DataFrame({'Values': blue_channel_values, 'Category': classification_labels})

    # violinplot = sns.violinplot(x='Category', y='Values', data=violin_data_df, ax=ax_box, palette=specific_colors, saturation=1.2)
    # Create a DataFrame with the data and category labels
    box_data_df = pd.DataFrame({'Values': blue_channel_values, 'Category': classification_labels})

    # Plot the box without outliers and with the specified color and saturation
    boxplot = sns.boxplot(x='Category', y='Values', data=box_data_df, ax=ax_box, palette=specific_colors, saturation=1.2, boxprops=dict(edgecolor="None"))

        # Set the color of the median line to match the box color
    # for line in boxplot.get_lines():
    #     line.set_color(line.get_color())
    average_values = box_data_df.groupby('Category')['Values'].mean()
    box_positions = np.arange(len(average_values))  # x-axis positions for each box

    if len(average_values) == len(specific_colors):
        ax_box.plot(box_positions, average_values.values, marker='v', linestyle='', markersize=8, color='black')


    # Calculate and plot average values
    # average_values = box_data_df.groupby('Category')['Values'].mean()
    # if len(average_values) == len(specific_colors):
    #     ax_box.hlines(average_values.values, range(len(average_values)), np.array(range(len(average_values))) + 0.8, colors=specific_colors, linewidth=2)

    ax_box.set_xlabel('')
    ax_box.set_ylabel(var_name)
    ax_box.set_xticklabels(cluster_name)
    ax_box.ticklabel_format(style='sci', scilimits=(0,1), axis='y')

    # ax_box.set_title(f'{var_name}')
    ax_box.set_xticks([])  # Remove X-axis ticks
    ax_box.set_ylabel(var_name, fontsize=20)
    # ax_box.set_xticklabels(cluster_name)
    ax_box.ticklabel_format(style='sci', scilimits=(0,1), axis='y')
    ax_box.yaxis.get_offset_text().set_fontsize(15)
    ax_box.tick_params(axis='both', labelsize=20)  # Set tick label size
    ax_box.tick_params(axis='y', labelsize=15)

    # # Find values exceeding upper and lower limits for each variable
    # upper_limit = np.percentile(blue_channel_values, 75) + 1.5 * (
    #         np.percentile(blue_channel_values, 75) - np.percentile(blue_channel_values, 25))
    # lower_limit = np.percentile(blue_channel_values, 25) - 1.5 * (
    #         np.percentile(blue_channel_values, 75) - np.percentile(blue_channel_values, 25))
x_labels = cluster_name#[f'Category {j + 1}' for j in range(n_clusters)]
i=i+1 
ax2 = plt.subplot(gs[i // num_columns, i % num_columns])
variables = [np.array(data2).flatten()]
variable_names = ['Q Index']
# Create box diagrams for data and blueo based on categories
for i, (variable, var_name) in enumerate(zip(variables, variable_names)):
    sns.boxplot(x=classification_labels, y=variable, ax=ax2,palette=specific_colors,saturation=1.2,boxprops=dict(edgecolor="None"))
    box_data_df = pd.DataFrame({'Values': variable, 'Category': classification_labels})
    average_values = box_data_df.groupby('Category')['Values'].mean()
    box_positions = np.arange(len(average_values))  # x-axis positions for each box

    if len(average_values) == len(specific_colors):
        ax2.plot(box_positions, average_values.values, marker='v', linestyle='', markersize=8, color='black')
    # axes3[i].set_title(f'Box Diagram for {var_name}')
    ax2.set_xlabel('')
    # ax2.set_xticklabels(cluster_name)
    ax2.set_xticklabels([])  # Remove X-axis labels
    ax2.set_ylabel(f'{var_name}', fontsize=20)
    # ax2.legend()
        # Format y-axis using scientific notation
    # ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.tick_params(axis='both', labelsize=20)  # Set tick label size
    ax2.yaxis.get_offset_text().set_fontsize(15)#设置1e6的大小与位置

# fig_box.delaxes(axes_box[-1, -1])
# Center the 5 graphs in the last row
ax_legend = plt.subplot(gs[-1, -1])
ax_legend.axis('off')  # Turn off the axis for the legend subplot
legend_labels = [f'{label}' for label in x_labels]
legend_rects = [plt.Rectangle((0, 0), 1, 1, color=color) for color in specific_colors[:n_clusters]]
ax_legend.legend(legend_rects, legend_labels, title='Type Names', title_fontsize='17', loc='center', ncol=1,frameon=False,fontsize='17')

ax_legend2 = plt.subplot(gs[-1, -2])
ax_legend2.axis('off') 
fig_box.subplots_adjust(wspace=.55,hspace=0.1) 
box_variables_path = os.path.join(output_folder, 'box_variables.png')

plt.savefig(box_variables_path)

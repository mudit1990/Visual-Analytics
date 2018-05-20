import numpy as np
import pandas as pd
import calendar
import editdistance
import pickle

from sklearn.cluster import DBSCAN
from collections import defaultdict
from itertools import groupby
from scipy import misc
from scipy.stats import itemfreq

from bokeh.io import curdoc
from bokeh.models import Legend
from bokeh.models.markers import Circle
from bokeh.models.widgets import PreText
from bokeh.models.widgets import Button,Slider
from bokeh.models import ColumnDataSource,LabelSet
from bokeh.models.tools import HoverTool,CrosshairTool
from bokeh.palettes import viridis
from bokeh.plotting import figure
from bokeh.layouts import column,row,widgetbox

####### Some important function definitions

def saveLocations(source,dc):
    x = source.data['x'];
    y = source.data['y'];
    text = source.data['text'];
    for i in range(0,len(text)):
        dc[text[i]] = str(x[i])+'_'+str(y[i]);

def equalList(l1,l2):
    if l1[0] == l2[0] and l1[1] == l2[1] and l1[2] == l2[2]:
        return True;
    return False;

def parseImage(image):
    red = [255,0,0];
    green = [76,255,0];
    blue = [0,255,255];
    black = [0,0,0];
    white = [255,255,255];
    orange = [255,106,0];
    yellow = [255,216,0];
    pink = [255,0,220]

    parsedImage = defaultdict(list);

    for i in range(0,200):
        for j in range(0,200):
            l = image[200-i-1][j];
            if equalList(l,red):
                parsedImage['gate'].append((j,i));
            elif equalList(l,green):
                parsedImage['entrance'].append((j,i));
            elif equalList(l,blue):
                parsedImage['general-gate'].append((j,i));
            elif equalList(l,white):
                parsedImage['road'].append((j,i));
            elif equalList(l,orange):
                parsedImage['camping'].append((j,i));
            elif equalList(l,yellow):
                parsedImage['ranger-stop'].append((j,i));
            elif equalList(l,pink):
                parsedImage['ranger-base'].append((j,i));
            else:
                parsedImage['blank'].append((j,i));

    return parsedImage;

def plotMap(parsedImage):
    mp = figure(title="Park Map", plot_width=750, plot_height=650);

    # plot all the blank spaces
    mp.rect(x = [x[0] for x in parsedImage['blank']],y = [x[1] for x in parsedImage['blank']],
    width=1, height=1, color='#FFFFFF');

    # plot the road
    mp.rect(x = [x[0] for x in parsedImage['road']],y = [x[1] for x in parsedImage['road']],
    width=1, height=1, color='#000000',alpha=0.5);

    # plot the gates
    gates_source = ColumnDataSource(data=dict(x=[x[0] for x in parsedImage['gate']],
    y=[x[1] for x in parsedImage['gate']],text=['gate8','gate7','gate6','gate5','gate4','gate3','gate2','gate1','gate0']));
    mp.rect(x='x',y='y', width=1, height=1, color='#FF0000', source=gates_source);
    mp.add_layout(LabelSet(x='x', y='y', text='text', source=gates_source, text_font_size='8pt',
    x_offset=0, y_offset=5, text_color='#FF0000',level='glyph',render_mode='canvas'));
    #saveLocations(gates_source,locDict);

    # plot the entrance
    entrance_source = ColumnDataSource(data=dict(x=[x[0] for x in parsedImage['entrance']],
    y=[x[1] for x in parsedImage['entrance']],text=['entrance4','entrance3','entrance2','entrance1','entrance0']));
    mp.rect(x='x',y='y', width=1, height=1, color='#00FF00', source=entrance_source);
    mp.add_layout(LabelSet(x='x', y='y', text='text', source=entrance_source, text_font_size='8pt',
    x_offset=0, y_offset=5, text_color='#00FF00',level='glyph',render_mode='canvas'));
    #saveLocations(entrance_source,locDict);

    # plot the general-gates
    general_gates_source = ColumnDataSource(data=dict(x=[x[0] for x in parsedImage['general-gate']],
    y=[x[1] for x in parsedImage['general-gate']],
    text=['general-gate7','general-gate6','general-gate5','general-gate4','general-gate3','general-gate2','general-gate1','general-gate0']));
    mp.rect(x='x',y='y', width=1, height=1, color='#0000FF', source=general_gates_source);
    mp.add_layout(LabelSet(x='x', y='y', text='text', source=general_gates_source, text_font_size='8pt',
    x_offset=0, y_offset=5, text_color='#0000FF',level='glyph',render_mode='canvas'));
    #saveLocations(general_gates_source,locDict);

    # plot the camping
    camping_source = ColumnDataSource(data=dict(x=[x[0] for x in parsedImage['camping']],
    y=[x[1] for x in parsedImage['camping']],
    text=['camping6','camping7','camping5','camping4','camping3','camping2','camping1','camping8','camping0']));
    mp.rect(x='x',y='y', width=1, height=1, color='#FF7F50', source=camping_source);
    mp.add_layout(LabelSet(x='x', y='y', text='text', source=camping_source, text_font_size='8pt',
    x_offset=0, y_offset=5, text_color='#FF7F50',level='glyph',render_mode='canvas'));
    #saveLocations(camping_source,locDict);

    # plot ranger-stops
    rangerstops_source = ColumnDataSource(data=dict(x=[x[0] for x in parsedImage['ranger-stop']],
    y=[x[1] for x in parsedImage['ranger-stop']],
    text=['ranger-stop7','ranger-stop6','ranger-stop5','ranger-stop4','ranger-stop3','ranger-stop2','ranger-stop1','ranger-stop0']));
    mp.rect(x='x',y='y', width=1, height=1, color='#FFFF00', source=rangerstops_source);
    mp.add_layout(LabelSet(x='x', y='y', text='text', source=rangerstops_source, text_font_size='8pt',
    x_offset=0, y_offset=5, text_color='#CBCE0C',level='glyph',render_mode='canvas'));
    #saveLocations(rangerstops_source,locDict);

    # plot ranger-base
    rangerbase_source = ColumnDataSource(data=dict(x=[x[0] for x in parsedImage['ranger-base']],
    y=[x[1] for x in parsedImage['ranger-base']],text=['ranger-base']));
    mp.rect(x='x',y='y', width=1, height=1, color='#FF00FF', source=rangerbase_source);
    mp.add_layout(LabelSet(x='x', y='y', text='text', source=rangerbase_source, text_font_size='8pt',
    x_offset=0, y_offset=5, text_color='#FF00FF',level='glyph',render_mode='canvas'));
    #saveLocations(rangerbase_source,locDict);

    # with open('sensor_locations.pickle', 'wb') as handle:
    #     pickle.dump(locDict, handle);
    return mp;

def encodeColors(numList, colorList):
    numColList = [];
    scale = (max(numList)+1)/10;
    for num in numList:
        if num == 0:
            numColList.append('#FFFFFF');
        else:
            numColList.append(colorList[int(num/scale)]);
    return numColList;

def getHeatMapData(hh, sensors, data):
    hhList = [];
    sensorList = [];
    normalList = [];
    specialList = [];
    for h in hh:
        for sensor in sensors:
            data_sub = data[(data['Timestamp'].dt.hour == h) & (data['gate-name'] == sensor)];
            hhList.append(str(h));
            sensorList.append(sensor);
            special_data_len = len(data_sub[data_sub['car-type'] == '2P']);
            normal_data_len = len(data_sub[data_sub['car-type'] != '2P']);
            normalList.append(normal_data_len);
            specialList.append(special_data_len);
    blueColorList = ['#EBF5FB','#D6EAF8','#AED6F1','#85C1E9','#5DADE2','#3498DB','#2E86C1','#2874A6','#21618C','#1B4F72'];
    greenColorList = ['#E8F8F5','#D1F2EB','#A3E4D7','#76D7C4','#48C9B0','#1ABC9C','#17A589','#148F77','#117864','#0E6251'];
    normalColors = encodeColors(normalList, blueColorList);
    specialColors = encodeColors(specialList, greenColorList);
    normalData = ColumnDataSource(data=dict(hh=hhList,sensors=sensorList,value=normalList,colors=normalColors));
    specialData = ColumnDataSource(data=dict(hh=hhList,sensors=sensorList,value=specialList,colors=specialColors));
    return normalData,specialData;

##### Read data

df = pd.read_csv('Lekagul Sensor Data.csv')

## Add time information
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['year'] = df.Timestamp.dt.year
df['month'] = df.Timestamp.dt.month
df['weekday'] = df.Timestamp.dt.weekday
df['day'] = df.Timestamp.dt.day
df['hour'] = df.Timestamp.dt.hour

car_types = np.unique(df['car-type'])
numCarTypes = car_types.shape[0]
car_type_colors = [viridis(numCarTypes)[x] for x in np.arange(numCarTypes)]

month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']



##### DATA FOR MONTHLY PLOT

MonthData = []

for car in car_types:
	mini_df = df.loc[df['car-type'] == car]
	tmp = []
	for mth in np.unique(df['month']):
		check_ins = mini_df.loc[mini_df['month'] == mth]
		tmp.append(check_ins.shape[0])
	MonthData.append(tmp)

monthLineData = {
'x'       : [list(np.unique(df['month'])) for i in np.arange(numCarTypes)],
'y'       : MonthData,
'cartype' : car_types,
'colors'  : car_type_colors,
'alpha'   : [0.9] * 7}

monthLineCDS = ColumnDataSource(data=monthLineData)

monthCircleData={
'x'       : list(np.unique(df['month'])) * numCarTypes,
'y'       : [item for data in MonthData for item in data],
'cartype' : [tmp for tmp in car_types for i in np.arange(12)],
'month'   : month_names * numCarTypes,
'colors'  : [tmp for tmp in car_type_colors for i in np.arange(12)],
'alpha'   : [0.9] * 84}

monthCircleCDS = ColumnDataSource(data=monthCircleData)



##### DATA FOR WEEKLY PLOT

WeekData = {}

for month in np.unique(df['month']):
	tmp = []
	for car in car_types:
		mini_df = df.loc[(df['month'] == month) & (df['car-type'] == car)]
		tmp1 = []
		for day in np.unique(df['weekday']):
			check_ins = mini_df.loc[df['weekday'] == day]
			tmp1.append(check_ins.shape[0])
		tmp.append(tmp1)
	WeekData[month] = tmp

weekLineData={
'x'       : [list(np.unique(df['weekday'])) for i in car_types],
'y'       : WeekData[1],
'cartype' : car_types,
'colors'  : car_type_colors,
'alpha'   : [0.9] * 7}

weekLineCDS = ColumnDataSource(data=weekLineData)

weekCircleData={
'x'       : list(np.unique(df['weekday'])) * numCarTypes,
'y'       : [item for data in WeekData[1] for item in data],
'weekday' : weekday_names * numCarTypes,
'cartype' : [tmp for tmp in car_types for i in np.arange(7)],
'colors'  : [tmp for tmp in car_type_colors for i in np.arange(7)],
'alpha'   : [0.9] * 49}

weekCircleCDS = ColumnDataSource(data=weekCircleData)



##### DATA FOR ILLEGAL GATE CHECK INS

illegal_stops = ['ranger-stop3', 'ranger-stop6']

### FIND THE CHECK INS BY NON-2P VEHICLES BETWEEN 2:00 AM and 5:00 AM IN ILLEGAL STOPS

illegal_check_ins = df.loc[(df['car-type'] != '2P') & (df['hour'].isin([2,3,4,5])) & (df['gate-name'].isin(illegal_stops))]

illegal_cars = np.unique(illegal_check_ins['car-id'])

illegal_paths = []

for car in illegal_cars:
	mini_df = df.loc[df['car-id'] == car]
	# print(mini_df.loc[:, ['Timestamp', 'car-id', 'car-type', 'gate-name']], "\n\n")
	illegal_paths.append(list(mini_df['gate-name']))

##### SET CALLBACKS #######################################################

def circle_callback(attr, old, new):
	pts = new['1d']['indices']
	if len(pts) > 0:
		month = monthCircleData['x'][pts[0]]
		### change data of 2nd plot to reflect month that was tapped upon
		weekLineCDS.data['y'] = WeekData[month]
		weekCircleCDS.data['y'] = [item for data in WeekData[month] for item in data]
		weekView.title.text = "Plot of Traffic in " + month_names[month - 1] + " by Weekday and Car Type"

		### change transparency of circles
		monthCircleCDS.data['alpha'] = [0.3] * 84
		weekCircleCDS.data['alpha'] = [0.3] * 49

def line_callback(attr, old, new):
	print("Line tapped!")
	line = new['1d']['indices']
	if len(line) > 0:
		clicked_line = line[0]
		### change transparency in both plots
		new_alpha = [0.3] * 7
		new_alpha[clicked_line] = 0.9

		monthCircleCDS.data['alpha'] = [0.3] * 84
		weekCircleCDS.data['alpha'] = [0.3] * 49

		monthLineCDS.data['alpha'] = new_alpha
		weekLineCDS.data['alpha'] = new_alpha

		print("monthLineCDS.data['alpha'] = ", new_alpha)
	else:
		### reset transparency
		monthCircleCDS.data['alpha'] = [0.9] * 84
		weekCircleCDS.data['alpha'] = [0.9] * 49

		monthLineCDS.data['alpha'] = [0.9] * 7
		weekLineCDS.data['alpha'] = [0.9] * 7


pre_intro = PreText(text="""

Students:
Mudit Bhargava (mbhargava@umass.edu)
Priyadarshi Rath (priyadarshir@umass.edu)

We have chosen to attempt VAST 2017. For Mini-Challenge 1, given traffic sensor data in the Lekagul Preserve, we are required to find
(a) daily patterns of life,
(b) extended patterns of life
(c) unusual patterns.

The first visualisation attempts to find patterns over the year. It shows two plots. The one on the left is the primary plot, showing traffic
activity in the park for each car type, along all months. The month is plotted from January to December. On the right, is the secondary, which
displays the traffic activity for weekdays of a particular month. The weekdays are plotted from Monday to Sunday The month can be changed by
interacting with the primary plot.

Interacting with this visualisation:
* When a line on the primary plot is clicked, it gets highlighted in both plots, enhancing the activity for the particular type desired.
* When a circle is clicked, the corresponding month is displayed in the right-hand-side plot.

Interesting points to note:
* The primary immediately tells us that the period between June-September is the busiest time of the year for the preserve. Sensors get
  triggered with high frequency, and the traffic consists mostly of motorcycles(type 1) and light cars(type 2). Heavier vehicles are relatively
  sparse in comparison.
* In the off-seasons, most of the traffic consists of ranger vehicles(type 2P).
* Traffic generally drops during the middle of the week, and is higher during the weekends.
""",
width=1200, height=420)
curdoc().add_root(pre_intro)

##### PLOT FOR EACH MONTH #####################################################

monthView = figure(x_range=[0,13], tools=['wheel_zoom', 'pan', 'tap', 'reset'], title="Monthly Plot of Traffic by Car Type")
monthHover = HoverTool(tooltips=[('Car Type', '@cartype'), ('Count', '$y{00000}'), ('Month', '@month')])
monthView.add_tools(monthHover)
monthLines = monthView.multi_line(xs="x", ys="y", line_color="colors", line_width=3, alpha="alpha", legend="cartype", source=monthLineCDS)
monthCircles = monthView.circle(x="x", y="y", size=10, fill_color="colors", line_color="colors", alpha="alpha", source=monthCircleCDS)
monthView.xaxis.axis_label = "Month"
monthView.yaxis.axis_label = "Traffic activity"
monthView.xaxis.ticker = []
monthView.min_border_bottom=100;
monthCircles.selection_glyph = Circle(fill_alpha=0.8, line_alpha=0.8)
monthCircles.nonselection_glyph = Circle(fill_alpha=0.3, line_alpha=0.3)

monthLines.data_source.on_change('selected', line_callback)
monthCircles.data_source.on_change('selected', circle_callback)

##### PLOT FOR A WEEK #########################################################

weekView = figure(x_range=[-1,7], tools=['wheel_zoom', 'pan', 'tap', 'reset'], title="Plot of Traffic in January by Weekday and Car Type")
weekHover = HoverTool(tooltips=[('Car Type', '@cartype'), ('Count', '$y{0000}'), ('Day', '@weekday')])
weekView.add_tools(weekHover)
weekView.min_border_bottom=100;
weekLines = weekView.multi_line(xs='x', ys='y', line_color='colors', line_width=3, alpha="alpha", source=weekLineCDS)
weekCircles = weekView.circle(x="x", y="y", size=6, fill_color="colors", line_color="colors", alpha="alpha", source=weekCircleCDS)
weekView.xaxis.axis_label = "Week"
weekView.yaxis.axis_label = "Traffic activity"
weekView.xaxis.ticker = []

##### ADD PLOTS TO LAYOUT #################################################

layout = row(monthView, weekView)
curdoc().add_root(layout)

###############################################################################
# EXPLORATORY HEATMAP

def heatmap_button_callback():
    if heatmap_button.label == 'Show Ranger Traffic':
        exp_heatmap.title.text = "Activity inside the reserve: Ranger Traffic";
        exp_heatmap.title.text_color = '#1D8348';
        heatmap_button.label = 'Show Normal Traffic';
        heatmap_button.button_type = 'primary';
        heatMapData.data = heatmapSpecialData.data;
    else:
        exp_heatmap.title.text = "Activity inside the reserve: Normal Traffic";
        exp_heatmap.title.text_color = '#21618C';
        heatmap_button.label = 'Show Ranger Traffic';
        heatmap_button.button_type = 'success';
        heatMapData.data = heatmapNormalData.data;

def outlier_button_callback():
    numList = heatmapNormalData.data['value'];
    colList = heatmapNormalData.data['colors'];
    hh = heatmapNormalData.data['hh'];
    sensors = heatmapNormalData.data['sensors'];

    restricted_zones = ['ranger-stop1','ranger-stop3','ranger-stop5','ranger-stop6','ranger-stop7'];
    newColList = [];
    for i in range(0,len(sensors)):
        if sensors[i] in restricted_zones and numList[i] > 0:
            newColList.append('firebrick');
        else:
            newColList.append('#F2F3F4');
    heatMapData.data = dict(hh=hh,sensors=sensors,value=numList,colors=newColList);
    exp_heatmap.title.text = "Activity inside the reserve: Outliers in Normal Traffic";
    exp_heatmap.title.text_color = 'firebrick';

def edit_distance_metric(docList):
    noDocs = len(docList);
    matrix = np.ndarray(shape=(noDocs,noDocs));
    for i in range(0,noDocs):
        for j in range(0,noDocs):
            if(i > j):
                matrix[i][j] = matrix[j][i];
            else:
                matrix[i][j] = editdistance.eval(docList[i],docList[j]);
    return matrix;

def removeConsecutiveDuplicates(l):
    return [x[0] for x in groupby(l)];

def getCarIdsAndDocsByMonth(data, month):
    #subset the data for the particular month
    #get the unique car-id and prepare the document for every unique car-id
    data_month = pd.DataFrame.copy(data[data['Timestamp'].dt.month == month]);
    data_month['car-id-byday'] = data_month['car-id'] + '_' + data_month['Timestamp'].dt.day.astype(str);
    carIdByDay = pd.unique(data_month['car-id-byday']);
    carIdList = [];
    sensorDocList = [];
    for carId in carIdByDay:
        carIdData = data_month[data_month['car-id-byday'] == carId];
        sensorList = removeConsecutiveDuplicates(carIdData['gate-name'].values);
        sensorDoc = ','.join(sensorList);
        carIdList.append(carId);
        sensorDocList.append(sensorDoc);
    return carIdList, sensorDocList;

def saveEditDistanceForAllMonths(data):
    monthList = [1,2,3,4,5,6,7,8,9,10,11,12];
    for month in monthList:
        print('computing for month...',month);
        carIdList, sensorDocList = getCarIdsAndDocsByMonth(data, month);
        np.save('carIdList'+str(month)+'.npy', carIdList);
        editdist_matrix = edit_distance_metric(sensorDocList);
        np.save('editDistanceMatrix'+str(month)+'.npy', editdist_matrix);

def loadEditDistMatrix(month):
    #carIdList = np.load('carIdList'+str(month)+'.npy');
    editDistMatrix = np.load('editDistanceMatrix'+str(month)+'.npy');
    return editDistMatrix;

def top3ClustersPaths(data):
    topPatternsDict={};
    for month in range(1,13):
        editDistMatrix = loadEditDistMatrix(month);
        print("dbscan starts now ...");
        dbscn = DBSCAN(eps=0.5,metric='precomputed',min_samples=1);
        dbscn_clusters = dbscn.fit_predict(X=editDistMatrix);
        cluster_freq = itemfreq(dbscn_clusters);
        sorted_cluster_freq = sorted(cluster_freq.tolist(),key=lambda x: x[1],reverse=True);
        top3Clusters = [x[0] for x in sorted_cluster_freq][0:3];
        carIdList, sensorDocList = getCarIdsAndDocsByMonth(data,month);
        topPathsDict={};
        for i in range(0,len(dbscn_clusters)):
            if dbscn_clusters[i] in top3Clusters:
                topPathsDict[dbscn_clusters[i]] = sensorDocList[i];
        topPatternsDict[month] = list(topPathsDict.values());

    with open('topPatterns.pickle', 'wb') as handle:
        pickle.dump(topPatternsDict, handle);

def getMonthPatterns(month):
    topPatterns = topPatternsDict[month];
    xs = [];
    ys = [];
    for pattern in topPatterns:
        patternList = pattern.split(',');
        x=[];
        y=[];
        for p in patternList:
            x_y = sensorLocDict[p].split('_');
            x.append(int(x_y[0]));
            y.append(int(x_y[1]));
        xs.append(x);
        ys.append(y);
    return xs,ys;

def topPatternsUpdate(attr,new,old):
    month = int(topPatternsSlider.value);
    parkMapPlotForPatterns.title.text='Park patterns for '+calendar.month_name[month];
    xs,ys = getMonthPatterns(month);
    data1 = dict(x=xs[0],y=ys[0]);
    data2 = dict(x=xs[1],y=ys[1]);
    data3 = dict(x=xs[2],y=ys[2]);
    pattern1Source.data = data1;
    pattern2Source.data = data2;
    pattern3Source.data = data3;

def dummy_callback(attr,new,old):
    print('i got clicked');
    return;

reserve_data = pd.read_csv("Lekagul Sensor Data.csv");
reserve_data['Timestamp'] = pd.to_datetime(reserve_data['Timestamp']);
#top3ClustersPaths(reserve_data);
with open('topPatterns.pickle', 'rb') as handle:
    topPatternsDict = pickle.load(handle);

with open('sensor_locations.pickle', 'rb') as handle:
    sensorLocDict = pickle.load(handle);

#saveEditDistanceForAllMonths(reserve_data);

hover_heatmap = HoverTool(tooltips=[('Time', '@hh'),('Sensor','@sensors'),('Freq','@value')]);
heatmap_hh = pd.unique(reserve_data['Timestamp'].dt.hour);
heatmap_sensors = pd.unique(reserve_data['gate-name']);
heatmapNormalData, heatmapSpecialData = getHeatMapData(heatmap_hh, heatmap_sensors, reserve_data);
heatMapData = ColumnDataSource(data=heatmapNormalData.data);
exp_heatmap = figure(title="Activity inside the reserve",x_range=[str(h) for h in heatmap_hh],y_range=list(heatmap_sensors), plot_height=550, plot_width=750);
exp_heatmap.title.text_font_size='13pt';
exp_heatmap.rect(x='hh', y='sensors', color='colors', width = 0.9, height = 0.9,source=heatMapData);
exp_heatmap.add_tools(hover_heatmap);
exp_heatmap.add_tools(CrosshairTool());
exp_heatmap.yaxis.axis_label="Sensor Locations";
exp_heatmap.xaxis.axis_label="Time (24h)";
exp_heatmap.min_border_top = 50;

heatmap_button = Button(label="Show Ranger Traffic", width=150, button_type='success');
heatmap_button.on_click(heatmap_button_callback);
outlier_button = Button(label="Find the Outliers", width=150, button_type='warning');
outlier_button.on_click(outlier_button_callback);

heatmap_pretext = PreText(text="""
The Heat Map on the left shows the sensor readings by sensor location and
time for the entire dataset. The ‘Show Ranger Traffic/Show Normal Traffic’
button shifts the Heat Map between ranger and traffic readings.
The ‘Outlier’ button shows interesting outliers in the dataset. We define
an Outlier as any normal traffic reading in restricted zones (i.e zones that
can only be accessed by rangers). For the dataset, restricted zones were
ranger-stop1, ranger-stop3, ranger-stop5, ranger-stop6, ranger-stop7.
Quite a few interesting patterns were found by this Heat Map.

* Click on the ‘Outlier’ button, 2 main outliers show up. First, at
  ranger-stop1 between 10am and 4pm. This may have happened possibly by
  accident. But the second outlier shows up at ranger-stop6 and ranger-stop3
  between 2am and 5am and the number of readings is consistently greater
  than 10. This according to us is suspicious and needs to be investigated
  by the rangers.

* Click on ‘Show Ranger Traffic/Show Normal Traffic’. Interestingly, the lower
  half of the heat map is much darker than the upper half, giving a high level
  idea of flow of traffic through the reserve. For e.g general-gate1,
  ranger-stop2, ranger-stop1, general-gate2 route is one of the most used
  routes in the reserve. Camping1 is overall the least used camping site.

* Click on ‘Show Ranger Traffic/Show Normal Traffic’. The heat map is much more
  distributed than the Normal Traffic Heat Map which says that the rangers cover
  almost the entire reserves during inspection. Apart from the ranger-base,
  ranger-stop6 shows unusually high ranger traffic as compared to other
  ranger stops. This may be because of some ongoing work or they may have
  discovered some problems at that place.
""", width=650, height=500)
curdoc().add_root(row(exp_heatmap,column(heatmap_button,outlier_button,heatmap_pretext)));

###############################################################################
# Illegal Path plot

pre_illegal = PreText(text="""
The plot on the left shows suspicious activity within the park. From the heatmap
visualisation, some traffic activity can be seen as non-ranger vehicles
triggering ranger stops and restricted gates between 2:00 AM and 5:00 AM.
Analysis of the related vehicle IDs reveals that they are all
"4 axle (and above) truck" category vehicles(type 4). Provided here is a
partial list of dates on which these trucks visit the park:

Date        ID
2015-05-05  20150505020522-625
2015-05-21  20151521021518-235
2015-06-02  20154702044723-914
2015-06-16  20150416040441-902
2015-06-25  20152925022919-735

From this, one can see that on average, one such suspicious truck visits
the preserve once every two weeks. All of these vehicles have distinct IDs,
and they all enter the park during the times mentioned above. Their path can
be seen in the map. They follow a fixed path, entering through entrance 3,
making their way upto ranger stop 3, back-tracking their path and leaving the
same way they entered. They all spend roughly one hour in the reserve, and of
that one hour, nearly 10 minutes are spent at ranger stop 6 itself, suggesting
that the driver(s) stop the vehicle there and get off to do something which
takes some time. This suggests some strange activity, and should be investigated.
""",
width=900, height=250)

illegalLineData = {
'x' : [114, 115, 122, 130, 123, 148, 147, 147, 148, 123, 130, 122, 115, 114],
'y' : [ 31,  48,  52,  53,  88, 139, 154, 154, 139,  88,  53,  52,  48,  31],
'index' : ['entrance3', 'gate6', 'ranger-stop6', 'gate5', 'general-gate5', 'gate3', 'ranger-stop3', \
           'ranger-stop3', 'gate3', 'general-gate5', 'gate5', 'ranger-stop6', 'gate6', 'entrance3']}

illegalLineCDS = ColumnDataSource(data=illegalLineData)
image = misc.imread('Lekagul Roadways.bmp')
imageDict = parseImage(image)
illegalPlot = plotMap(imageDict)
illegalPlot.min_border_bottom=100;
illegalLine = illegalPlot.line(x='x', y='y', line_width=7, line_color="red", line_alpha=0.7, source=illegalLineCDS)
curdoc().add_root(row(illegalPlot, pre_illegal))

###############################################################################
# Top patterns per month
parkMapPlotForPatterns = plotMap(imageDict);

xs,ys = getMonthPatterns(1);
parkMapPlotForPatterns.title.text='Park patterns for '+calendar.month_name[1];
parkMapPlotForPatterns.title.text_font_size='11pt';
pattern1Source = ColumnDataSource(data=dict(x=xs[0],y=ys[0]));
pattern2Source = ColumnDataSource(data=dict(x=xs[1],y=ys[1]));
pattern3Source = ColumnDataSource(data=dict(x=xs[2],y=ys[2]));
patternLine1 = parkMapPlotForPatterns.line(x='x',y='y',color='#3634D4',line_width=8,line_alpha=0.5,source=pattern1Source,legend='Pattern1');
patternLine2 = parkMapPlotForPatterns.line(x='x',y='y',color='#17A589',line_width=8,line_alpha=0.5,source=pattern2Source,legend='Pattern2');
patternLine3 = parkMapPlotForPatterns.line(x='x',y='y',color='#AA1414',line_width=8,line_alpha=0.5,source=pattern3Source,legend='Pattern3');
parkMapPlotForPatterns.legend.click_policy='hide';
topPatternsSlider = Slider(title="Month of the Year", value=1, start=1, end=12);
topPatternsSlider.on_change('value',topPatternsUpdate);
pre_patterns = PreText(text="""
We now try to find some movement patterns of people in the reserve park across the
months. For this we represent the movement of every car-id on each day as a separate
document. The documents are clustered with DBSCAN using levenshtein distance (edit distance)
as the metric. The edit distance tolerance is kept very low at 0.5 and the clustering
is performed seperately for each month. The intuition behind the low edit distance
is that we want to find the most common movement patterns in the reserve. The
clustering was done every month because based on the above analysis, we believe
different months will give very different patterns. The map plot on the left shows
the top 3 most frequent movement patterns by month. The month can be changed with
the help of the slider. One can click on the legend to view any specific pattern
and hide other patterns.

Interesting observations
* Through the months of Jan-May, the most common patterns are short patterns going
from entrance to entrance. Some even go directly from one entrance to the other
without any other sensor readings. The next most common patterns are the ranger
patterns. This shows that the during these off peak months the most common patterns
are people just passing through the reserve or very short trip through the reserve
* Through the months from June-September, there is change in the most frequent patterns.
Most of the patterns are from camping4/5/8 to entrance gates, suggesting a lot of
camping activites during the peak season. These patterns also show that when coming
into the park, people take very different paths to reach their final camping spots.
But while exiting the park from the camping spots, people tend to take the shortest
path to the desired entrance. Also, the consistent ranger patterns have disappeared
due to the large influx of general public
* The months through October to December show very similar patterns to Jan-May.
Hence october basically marks the start of offseason, where the most frequent patterns
are mostly entrance to entrance and ranger movements.
""",
width=900, height=250)
curdoc().add_root(column(widgetbox(topPatternsSlider,width=500),row(parkMapPlotForPatterns,pre_patterns)));

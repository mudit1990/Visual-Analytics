import numpy as np;
import pandas as pd;
from collections import defaultdict;
import calendar
import pickle

from bokeh.io import curdoc;
from bokeh.models import ColumnDataSource,LabelSet,Label,FactorRange,Arrow,OpenHead,NormalHead,VeeHead;
from bokeh.plotting import figure;
from bokeh.layouts import gridplot,column,widgetbox,row;
from bokeh.models.widgets import Slider,PreText,Button;
from bokeh.models.tools import TapTool;
from bokeh.palettes import viridis
from bokeh.models.glyphs import Rect
from bokeh.models.tools import HoverTool
from bokeh.models.widgets import Panel, Tabs
from bokeh.models.widgets import RadioButtonGroup

# hr should be a numpy array
# returns the angle in degrees given the hour of the day
def getAngle(hr):
	return (hr*360.0)/24.0;

# r and th should be numpy arrays
# returns the x,y points given the length of the segment
# and the angle it makes with the positive y axis
def getAngularPoints(r,th):
	x = r*np.sin((2*np.pi*th)/360.0);
	y = r*np.cos((2*np.pi*th)/360.0);
	return x,y;

# draws plot of a circle that represents time
def getTimeCircle(circle_radius):
	time_ang = getAngle(timeArr);
	radial_fig = figure(title="Radial Plot", plot_width=500, plot_height=500,x_range=(-20,20),y_range=(-20,20));
	radial_fig.circle(0,0,radius=circle_radius,fill_color='#F4F6F6',line_color='black');
	# add the smaller circles
	x_cir_ang,y_cir_ang = getAngularPoints(np.array(np.zeros(len(time_ang),))+circle_radius,time_ang);
	radial_fig.circle(x=x_cir_ang,y=y_cir_ang,size=4,fill_color='black');
	# add time annotations
	disp_time = np.array([0,4,8,12,16,20]);
	disp_time_ang = getAngle(disp_time);
	x_cir_label_ang,y_cir_label_ang = getAngularPoints(np.array(np.zeros(6,))+circle_radius+1,disp_time_ang);
	cir_label_source = ColumnDataSource(data=dict(x=x_cir_label_ang,y=y_cir_label_ang,text=disp_time));
	circle_labels = LabelSet(x='x', y='y', text='text', level='glyph',
	x_offset=0, y_offset=0, render_mode='canvas',text_font_size='10pt',source=cir_label_source);
	circle_title = Label(x=-3,y=-1,text='Time',render_mode='canvas',text_font_size='20pt',text_font='helvetica',text_font_style='bold');
	radial_fig.add_layout(circle_labels);
	radial_fig.add_layout(circle_title);
	radial_fig.axis.visible = False;
	return radial_fig;

def getWindCircle(circle_radius, dirList):
	radial_fig = figure(title="Wind Direction", plot_width=500, plot_height=500,x_range=(-40,40),y_range=(-40,40));
	radial_fig.circle(0,0,radius=circle_radius,fill_color='#F4F6F6',line_color='black');
	# add the smaller circles
	x_cir_ang,y_cir_ang = getAngularPoints(np.array(np.zeros(len(dirList),))+circle_radius,dirList);
	radial_fig.circle(x=x_cir_ang,y=y_cir_ang,size=4,fill_color='black');
	# add wind annotations
	disp_wind = np.array([0,60,120,180,240,300]);
	x_cir_label_ang,y_cir_label_ang = getAngularPoints(np.array(np.zeros(len(disp_wind),))+circle_radius+1,disp_wind);
	cir_label_source = ColumnDataSource(data=dict(x=x_cir_label_ang,y=y_cir_label_ang,text=disp_wind));
	circle_labels = LabelSet(x='x', y='y', text='text', level='glyph',
	x_offset=0, y_offset=0, render_mode='canvas',text_font_size='10pt',source=cir_label_source);
	circle_title = Label(x=-6,y=-1,text='Wind Direction',render_mode='canvas',text_font_size='10pt',text_font='helvetica',text_font_style='bold');
	radial_fig.add_layout(circle_labels);
	true_north = Label(x=-0.5,y=8,text='N',render_mode='canvas',text_font_size='10pt',text_font='helvetica',text_font_style='bold');
	radial_fig.add_layout(Arrow(end=OpenHead(line_color="navy", line_width=2,size=10),
	x_start=0, y_start=-7.5, x_end=0, y_end=7.5,line_color='navy',line_width=2));
	radial_fig.add_layout(true_north);
	radial_fig.add_layout(circle_title);
	radial_fig.axis.visible = False;
	return radial_fig;


def getAvgReadingByTime(data, chemical):
	reading = [];
	sub_data = data[data['Chemical'] == chemical];
	for tm in timeArr:
		avg_val = sub_data[sub_data['DateTime'].dt.hour == tm]['Reading'].mean();
		reading.append(avg_val);
	return np.array(reading);

def createChemicalPatternPlot(data_monitor_marginalized, chemical, color):
	chemical_reading = getAvgReadingByTime(data_monitor_marginalized, chemical);
	circle_radius = 10.0;
	radial_fig = getTimeCircle(circle_radius);
	time_ang = getAngle(timeArr);
	x_ang,y_ang = getAngularPoints(chemical_reading+circle_radius,time_ang);
	radial_fig.line(x=x_ang,y=y_ang,line_width=6, line_alpha=0.6, line_color = color);
	radial_fig.circle(x=x_ang,y=y_ang,size=4, fill_color='black');
	radial_fig.title.text=chemical;
	radial_fig.title.text_font='helvetica';
	radial_fig.title.text_font_size='12pt';
	return radial_fig;

def getSensorDataWithMaxReadings(sensorData):
	idx = sensorData.groupby(['Chemical','DateTime'])['Reading'].transform(max) == sensorData['Reading'];
	return pd.DataFrame.copy(sensorData[idx]);

def getCustomTupleKeyValues(df):
	dt = df[['Monitor','Chemical']].values.astype(str);
	keys = [tuple(x) for x in dt];
	values = df['DateTime'].values;
	return keys, values;

def getSensorChemicalDistributionByMonth(month):
	monthData = sensorDataMaxReadings[sensorDataMaxReadings['DateTime'].dt.month == month];
	sensorChemicalDist = monthData.groupby(['Chemical','Monitor']).count().reset_index();
	sensorBarKeys,sensorBarValues = getCustomTupleKeyValues(sensorChemicalDist);
	return sensorBarKeys, sensorBarValues;

def getWindDirectionByMonth(month):
	dirData = metData[metData['Date'].dt.month == month];
	dirData = dirData['Wind Direction'];
	dirDict = {};
	for d in np.arange(0,360,winddir_split):
		dirDict[d]=0;
	for d in dirData:
		if not np.isnan(d):
			dirDict[int(d/winddir_split)*winddir_split]+=1;
	sorted_list = sorted(dirDict.items());
	dirList = np.array([t[0] for t in sorted_list]);
	dirList = np.append(dirList,0);
	counts = np.array([t[1] for t in sorted_list]);
	counts = np.append(counts,dirDict[0]);
	counts_norm = (counts*8)/counts.max();
	x_ang,y_ang = getAngularPoints(counts_norm+wind_circle_radius,dirList);
	return x_ang,y_ang;

def windAndDistUpdate(attr,new,old):
	month = int(monthSlider.value);
	x_ang,y_ang = getWindDirectionByMonth(month);
	winddir_source.data = dict(x=x_ang,y=y_ang);
	windDirPlot.title.text = 'Wind Direction in '+calendar.month_name[month];

	sensorBarKeys, sensorBarValues = getSensorChemicalDistributionByMonth(month);
	sensorBarPlot.x_range = FactorRange(*sensorBarKeys);
	sensorChemicalDistSource.data = dict(x=sensorBarKeys,y=sensorBarValues);

def plotSensors(plt,rel_x,rel_y,circle_size,color):
	rel_loc_x = sensor_loc_x - rel_x;
	rel_loc_y = sensor_loc_y - rel_y;
	plt.circle(x=rel_loc_x,y=rel_loc_y,fill_color=color,size=circle_size);
	sensor_text_disp = np.array([1,2,3,4,5,6,7,8,9]);
	sensor_label_source = ColumnDataSource(data=dict(x=rel_loc_x,y=rel_loc_y,text=sensor_text_disp));
	sensor_labels = LabelSet(x='x', y='y', text='text', level='glyph',
	x_offset=-15, y_offset=0, render_mode='canvas',text_font_size='10pt',source=sensor_label_source);
	plt.add_layout(sensor_labels);

def getFactorySensorLines(fact_x,fact_y):
	xs = [];
	ys = [];
	for i in range(len(sensor_loc_x)):
		sx = sensor_loc_x[i];
		sy = sensor_loc_y[i];
		xs.append([fact_x,sx]);
		ys.append([fact_y,sy]);
	return xs,ys;

def selected_factory(attr,old,new):
	pts = new['1d']['indices'];
	if len(pts) == 0:
		factorySensorLinesSource.data = dict(xs=[],ys=[]);
	else:
		fact_x = factory_source.data['x'][pts];
		fact_y = factory_source.data['y'][pts];
		xs,ys = getFactorySensorLines(fact_x,fact_y);
		factorySensorLinesSource.data = dict(xs=xs,ys=ys);

def joinSensorAndMetData(sensorData,metData):
	M = pd.DataFrame.copy(metData[['Date','Wind Direction']]);
	M = M.dropna(how='all');
	M['hr'] = M['Date'].dt.hour.astype(int);
	M['day'] = M['Date'].dt.day.astype(int);
	M['month'] = M['Date'].dt.month.astype(int);

	S = pd.DataFrame.copy(sensorData);
	S['hr'] = (S['DateTime'].dt.hour/3).astype(int)*3;
	S['day'] = S['DateTime'].dt.day.astype(int);
	S['month'] = S['DateTime'].dt.month.astype(int);

	SandM = pd.merge(left=S,right=M,on=['hr','day','month']);
	return SandM;

def normalizeReadingsByPercentile(readings,percentile):
	normalizer = np.percentile(readings,99.9);
	readings = readings/normalizer;
	# squish all readings greater than normalizer
	readings[readings>1] = 1;
	factorScale=5;
	distScale=0;
	return readings*factorScale + distScale;

def scaleReadingsBySensorPos(data):
	for i in range(len(sensor_loc_x)):
		sx = sensor_loc_x[i];
		sy = sensor_loc_y[i];
		data.loc[data.Monitor == (i+1),'ang_x'] += sx;
		data.loc[data.Monitor == (i+1),'ang_y'] += sy;
	return data;

def prepareAngularCoordinates(data):
	# reverse the wind direction, since we will be plotting
	# backwards from the sensors
	#data['Wind Direction'] = (data['Wind Direction']+180)%360;
	wind_dir = np.array((data['Wind Direction']+180)%360);
	readings = np.array(data['Reading']);
	readings = normalizeReadingsByPercentile(readings,95);
	ang_x, ang_y = getAngularPoints(readings,wind_dir);
	data['ang_x'] = ang_x;
	data['ang_y'] = ang_y;
	data = scaleReadingsBySensorPos(data);
	return data;

def getReadingsByWindDirectionAndChemical(dirStart,dirEnd,chemical):
	subdata = sensorAndMetData[(sensorAndMetData['Chemical'] == chemical) &
	(sensorAndMetData['Wind Direction'] >= dirStart) &
	(sensorAndMetData['Wind Direction'] <= dirEnd)];
	return np.array(subdata['ang_x']),np.array(subdata['ang_y']);

def callback_meth_button():
	x,y = getReadingsByWindDirectionAndChemical(0,360,'Methylosmolene');
	color = ['#E74C3C']*len(x);
	readingsByWindAndChemicalSource.data = dict(x=x,y=y,color=color);
	chemicalFactoryPlot.title.text = "Methylosmolene Responsibility";
	chemicalFactoryPlot.title.text_color = '#E74C3C';

def callback_chloro_button():
	x,y = getReadingsByWindDirectionAndChemical(0,360,'Chlorodinine');
	color = ['#DC7633']*len(x);
	readingsByWindAndChemicalSource.data = dict(x=x,y=y,color=color);
	chemicalFactoryPlot.title.text = "Chlorodinine Responsibility";
	chemicalFactoryPlot.title.text_color = '#DC7633';

def callback_agoc_button():
	x,y = getReadingsByWindDirectionAndChemical(0,360,'AGOC-3A');
	color = ['#2ECC71']*len(x);
	readingsByWindAndChemicalSource.data = dict(x=x,y=y,color=color);
	chemicalFactoryPlot.title.text = "AGOC-3A Responsibility";
	chemicalFactoryPlot.title.text_color = '#2ECC71';

def callback_appl_button():
	x,y = getReadingsByWindDirectionAndChemical(0,360,'Appluimonia');
	color = ['#3498DB']*len(x);
	readingsByWindAndChemicalSource.data = dict(x=x,y=y,color=color);
	chemicalFactoryPlot.title.text = "Appluimonia Responsibility";
	chemicalFactoryPlot.title.text_color = '#3498DB';

def get_data():

	Readings = []
	Counts = []

	for sensor in sensors:
		tmp1 = []
		tmp2 = []
		for chem in chems:
			mini_df = snsr_df.loc[(snsr_df['Monitor'] == sensor) & (snsr_df['Chemical'] == chem)]
			Rd = []
			Cts = []
			for day in days:
				ct_df = mini_df.loc[mini_df['Day'] == day]
				Cts.append(ct_df.shape[0])
				for hr in hours:
					tmp = mini_df.loc[(mini_df['Day'] == day) & (mini_df['hour'] == hr)]
					if tmp.shape[0] == 0:
						Rd.append(-1)
					else:
						Rd.append(tmp.iloc[0]['Reading'])
			tmp1.append(Rd)
			tmp2.append(Cts)
		Readings.append(tmp1)
		Counts.append(tmp2)

	return Readings, Counts

def HMCallback(attr, old, new):
	if len(new['1d']['indices']) > 0:
		clicked_pt = new['1d']['indices'][0]
		alphas = [0.3] * len(HMCDS.data['alpha'])
		alphas[clicked_pt] = 0.9
		HMCDS.data['alpha'] = alphas

		# Update Line plot
		dayIndexToShow = np.int(np.mod(clicked_pt, 92))
		dayToShow = days[dayIndexToShow]
		sensorToShow = 1 + np.int(clicked_pt/92)

		LinePlot.title.text = "Readings for Sensor " + str(sensorToShow) + " on " + str(dates[dayIndexToShow])


		AGOC = snsr_df.loc[(snsr_df['Monitor'] == sensorToShow) & (snsr_df['Day'] == dayToShow) & (snsr_df['Chemical'] == chems[0])]
		AGOC_reading = list(AGOC['Reading'])
		AGOC_times = list(AGOC['hour'])

		Appl = snsr_df.loc[(snsr_df['Monitor'] == sensorToShow) & (snsr_df['Day'] == dayToShow) & (snsr_df['Chemical'] == chems[1])]
		Appl_reading = list(Appl['Reading'])
		Appl_times = list(Appl['hour'])

		Chlr = snsr_df.loc[(snsr_df['Monitor'] == sensorToShow) & (snsr_df['Day'] == dayToShow) & (snsr_df['Chemical'] == chems[2])]
		Chlr_reading = list(Chlr['Reading'])
		Chlr_times = list(Chlr['hour'])

		Meth = snsr_df.loc[(snsr_df['Monitor'] == sensorToShow) & (snsr_df['Day'] == dayToShow) & (snsr_df['Chemical'] == chems[3])]
		Meth_reading = list(Meth['Reading'])
		Meth_times = list(Meth['hour'])

		LCDS.data['y'] = [AGOC_reading, Appl_reading, Chlr_reading, Meth_reading]
		LCDS.data['x'] = [AGOC_times, Appl_times, Chlr_times, Meth_times]
		LinePlot.title.text = "Readings for Sensor " + sensorToShow
	else:
		HMCDS.data['alpha'] = [0.9] * len(HMCDS.data['alpha'])

# SET CALLBACK

def chem_callback(new):
	for CDS in CDSs:
		CDS.data['y'] = CDS.data[chems[new]]

# ------------------------------------------------------------------------------
# ------------- actual code using these functions starts from here -------------
# ------------------------------------------------------------------------------

# data used for the first part MC2
snsr_df = pd.read_excel('Sensor Data.xlsx')
metr_df = pd.read_excel('Meteorological Data.xlsx')
# data used for second part of MC2
sensorData = pd.DataFrame.copy(snsr_df);
metData = pd.DataFrame.copy(metr_df);

snsr_df.rename(columns={'Date Time ': 'Timestamp'}, inplace=True)
snsr_df['Timestamp'] = pd.to_datetime(snsr_df['Timestamp'])
snsr_df['Day'] = snsr_df.Timestamp.dt.dayofyear
snsr_df['hour'] = snsr_df.Timestamp.dt.hour
snsr_df['Date'] = snsr_df.Timestamp.dt.date

days = pd.unique(snsr_df['Day'])
dates = pd.unique(snsr_df['Date'])
hours = sorted(pd.unique(snsr_df['hour']))
chems = sorted(pd.unique(snsr_df['Chemical']))
sensors = sorted(pd.unique(snsr_df['Monitor']))

# Readings, Counts = get_data()
# pickle.dump(Readings, open("Readings.pickle", "wb"))
# pickle.dump(Counts, open("Counts.pickle", "wb"))

Readings = pickle.load(open('Readings.pickle', 'rb'))
Counts = pickle.load(open('Counts.pickle', 'rb'))

# ------------------------------------------------------------------------------
# ------------- Create the gridplot containing 9 sensors scatter plot ----------
# ------------------------------------------------------------------------------

CDSs = []
for i in range(len(sensors)):
	data = {'x':range(2208), 'y':Readings[i][0]}
	for j in range(len(chems)):
		data[chems[j]] = Readings[i][j]
	CDS = ColumnDataSource(data=data)
	CDSs.append(CDS)

# Create 4 panels, each with 9 figures, one for each sensor

chem_select = RadioButtonGroup(labels=chems, active=0)
chem_select.on_click(chem_callback)

figures = []

for j in range(len(sensors)):
	fig = figure(plot_width=300, plot_height=300, y_range=[0,10], tools=['wheel_zoom', 'pan', 'reset'], title="Sensor " + str(j+1))
	dots = fig.circle(x='x', y='y', source = CDSs[j])
	fig.xaxis.ticker = []
	fig.xaxis.axis_label = "Time"
	fig.yaxis.axis_label = "Sensor Reading"
	figures.append(fig)

row1 = row(figures[0], figures[1], figures[2])
row2 = row(figures[3], figures[4], figures[5])
row3 = row(figures[6], figures[7], figures[8])
grid = column(row1, row2, row3)

pre_intro = PreText(text="""
Mudit Bhargava (mbhargava@umass.edu)
Priyadarshi Rath (priyadarshir@umass.edu)

VAST Challenge 2017 - Mini Challenge 2
----------------------------------------


Data Exploration: The Sensors
-------------------------------
The first step in understanding the behaviour of the sensors, is to plot their readings. The plot below shows a 3x3 grid, containing
the sensor reading for some particular chemical. The choice of chemical can be made by the user, and there are a few interesting
points to be observed.

* Sensor 3 and Sensor 7 give consistently noisy outputs.

* Sensor 4 performs worse as time passes, as can be seen from a shift in its base-line measurements.

* Sensor 5 exhibits linearly increasing noise as time passes. The noise is not as abrupt as Sensor 4, but it is a visible pattern.

* Sensor 9 probably experiences a malfunction during the last month(December). This is shown by a noticeable increase in noise
  during the final one-third portion of the plot.
""",
width=1200, height=400)


curdoc().add_root(pre_intro)
curdoc().add_root(chem_select)
curdoc().add_root(grid)


# ------------------------------------------------------------------------------
# ------------------------- Create the heatmap plot ----------------------------
# ------------------------------------------------------------------------------

x = []
y = []
date = []
colors = []
c1 = []
c2 = []
c3 = []
c4 = []

for i in range(len(sensors)):
	chemCts = Counts[i]
	for k in range(92): # 92 days
		chem1ct = chemCts[0][k]
		chem2ct = chemCts[1][k]
		chem3ct = chemCts[2][k]
		chem4ct = chemCts[3][k]
		totalct = chem1ct + chem2ct + chem3ct + chem4ct
		if totalct < 24*4:
			colors.append('red')
		elif ((chem1ct != 24) or (chem1ct != 24) or (chem1ct != 24) or (chem1ct != 24)):
			colors.append('green')
		elif ((chem1ct == 24) and (chem1ct == 24) and (chem1ct == 24) and (chem1ct == 24)):
			colors.append('blue')
		x.append(str(days[k]))
		y.append(str(sensors[i]))
		date.append(str(dates[k]))
		c1.append(chem1ct)
		c2.append(chem2ct)
		c3.append(chem3ct)
		c4.append(chem4ct)

HMData = {
'x' : x,
'y' : y,
'date' : date,
'colors' : colors,
'c1' : c1,
'c2' : c2,
'c3' : c3,
'c4' : c4,
'alpha': [0.9]*len(x)}

HMCDS = ColumnDataSource(data=HMData)

HM = figure(title="HeatMap of sensor reading counts", x_range = [str(i) for i in days], y_range = [str(i) for i in sensors],
			width=1200, height=500, tools=['wheel_zoom', 'pan', 'tap', 'reset'])
HM.xaxis.ticker = []
HM.xaxis.axis_label = "Day"
HM.yaxis.axis_label = "Sensor ID"
HMHover = HoverTool(tooltips=[('AGOC-3A', '@c1'), ('Appluimonia', '@c2'), ('Chlorodinine', '@c3'),
							  ('Methylosmolene', '@c4'), ('Date', '@date')])
HM.add_tools(HMHover)
HMRect = HM.rect(x='x', y='y', color='colors', alpha='alpha', width=0.85, height=0.9, source=HMCDS)
HMRect.data_source.on_change('selected', HMCallback)

pre_HM = PreText(text="""
Another way to understand sensor behavior is a heatmap visualisation of sensor readings. The color code for the plot is as follows:

Blue  - Sensor captured 24 readings for each chemical during the course of the day. This is the normal case since each chemical
		should have one reading per hour.
Red   - Sensor captured insufficient number of readings for the day.
Green - For at least one chemical, the sensor captured more readings than expected.

Interacting with the visualisation:
* Click on a square to show the readings for the day. The readings show up as lines in the linked plot below the heatmap.
* The line plot has the X-axis as hours of the day, and Y-axis is the reading for the chemical.
""",
width=1200, height=200)

curdoc().add_root(pre_HM)
curdoc().add_root(HM)


# ------------------------------------------------------------------------------
# -------------------- Create the lineplot linked to heatmap -------------------
# ------------------------------------------------------------------------------

LCDS = ColumnDataSource(data={
	'x':[[x for x in range(24)]]*4,
	'y':[np.zeros(24)]*4,
	'legend':chems,
	'colors':['red', 'blue', 'green', 'orange']})

LinePlot = figure(title="", width=1200, height=500, x_range=[0,23])
LinePlot.xaxis.axis_label = "Hour (24-hr Format)"
LinePlot.yaxis.axis_label = "Sensor Reading"
Multi_Line = LinePlot.multi_line(xs='x', ys='y', legend='legend', line_color='colors', source=LCDS)

curdoc().add_root(LinePlot)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# create timeArr
timeArr = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,0]);

sensor_loc_x = np.array([62,66,76,88,103,102,89,74,119]);
sensor_loc_y = np.array([21,35,41,45,43,22,3,7,42]);

factory_loc_x = np.array([89,90,109,120]);
factory_loc_y = np.array([27,21,26,22]);
factory_names = np.array(['Roadrunner','Kasios','Radiance','Indigo'])

# read the data and clean
metData['Date'] = pd.to_datetime(metData['Date']);
# correct the readings to show to Direction of wind
# the readings now show the direction they are flowing towards
metData['Wind Direction'] = (metData['Wind Direction']+180)%360;

sensorData['DateTime'] = pd.to_datetime(sensorData['Date Time ']);
sensorData = sensorData.drop('Date Time ',axis=1);

pre_chem_patterns = PreText(text="""
Data Exploration: The Chemicals
---------------------------------
The plot below shows the average chemical release patterns throughout the day. The plot was obtained
by taking the max readings for every chemical on a particular date time and then by averaging the
readings by hour of the day.

* Its interesting to see that the dangerous Methylosmolene’s maximum release usually occurs between
10pm and 5am. This is different from the other chemical release patterns and hence needs to be further
investigated. Another thing to note is liquid Methylosmolene needs to be disposed off. This may
correlate with the suspicious activities occurring in the forest reserve’s restricted zones
between 2am and 5am. (as found in MC1)
* The safer VOC AGOC-3A has an almost opposite release pattern. Most of the release occurs 6am and 8pm.
* Appluimonia and Chlorodinine have a consistent release pattern through the 24hrs
""",
width=1000, height=200);
curdoc().add_root(pre_chem_patterns);

# ------------------------------------------------------------------------------
# -------------- create radial plots to show chemical patterns -----------------
# ------------------------------------------------------------------------------

#group data by chemical and datetime and take max reading among all the sensors
data_monitor_marginalized = sensorData.groupby(['Chemical','DateTime'])['Reading'].max().reset_index();
radial_fig1 = createChemicalPatternPlot(data_monitor_marginalized, 'Methylosmolene', '#EC7063');
radial_fig2 = createChemicalPatternPlot(data_monitor_marginalized, 'AGOC-3A', '#1ABC9C');
radial_fig3 = createChemicalPatternPlot(data_monitor_marginalized, 'Appluimonia', '#2E86C1');
radial_fig4 = createChemicalPatternPlot(data_monitor_marginalized, 'Chlorodinine', '#F39C12');
radial_figs = gridplot([[radial_fig1, radial_fig2], [radial_fig3, radial_fig4]]);
curdoc().add_root(radial_figs);

wind_chemdist_patterns = PreText(text="""





The below plots show the wind direction and distribution of chemicals across the sensors by month.
The wind direction plot displays the number of times wind blew in a particular direction during the
given month. For e.g in the month of April on most days wind blew towards the south east direction
between 60 degrees and 180 degrees from the north. On a few days it blew towards the North west
direction between 330 and 360 degrees. The sensors are plotted as red dots to help understand
what effect, wind direction may have on sensor readings.
The second plot displays the distribution of max recorded readings for each chemical by sensors.
This gives us a basic understanding of what each of the sensors are recording and if any sensor
may have a particular liking for any chemical.
Both the plots are linked together by the Month slider. The two plots in union may help us understand
the effect of wind direction on the sensors and their readings.

An Open Problem
----------------
Ideally the two plots should have been aligned next to each other. When the plots are kept next to
each other, the radial plot gets distorted and becomes elliptical. The problem seems to be with Bokeh
forcing all objects to have the same sizing_mode within a row layout. We played around with different
attributes of layout, but it did not solve the problem. Hence we were forced to display the 2 plots
one below the other even though they are controlled by the slider.


Analysis
---------
It can be seen from the two plots that sensors don’t have any preference towards any particular
chemicals. The wind direction does have small effects on sensor readings e.g.
* Sensor 1 has very low recordings in the entire dataset, primarily because no wind blows in
the direction of sensor 1
* Sensor 8 has almost no recordings in December again because of the wind directions
* Sensors 6 and 7 have higher recordings in April as compared to other months probably
because of the wind directions.""",
width=1000, height=600);
curdoc().add_root(wind_chemdist_patterns);

# ------------------------------------------------------------------------------
# ------------- Create the wind circle and sensor chemical distribution --------
# ------------------------------------------------------------------------------

wind_circle_radius = 15.0;
winddir_split = 30;
windDirPlot = getWindCircle(wind_circle_radius,np.arange(0,360,winddir_split));
wind_ang_x,wind_ang_y = getWindDirectionByMonth(4);
winddir_source = ColumnDataSource(data=dict(x=wind_ang_x,y=wind_ang_y));
windDirPlot.line(x='x',y='y',line_width=8, line_alpha=0.6, line_color = '#5DADE2',source=winddir_source);
windDirPlot.circle(x='x',y='y',size=4, fill_color='black',source=winddir_source);
windDirPlot.title.text = 'Wind Direction in April';
plotSensors(windDirPlot,89.0,27.0,10,'red');

sensorDataMaxReadings = getSensorDataWithMaxReadings(sensorData);
sensorBarKeys, sensorBarValues = getSensorChemicalDistributionByMonth(4);
sensorBarPlot = figure(x_range=FactorRange(*sensorBarKeys), plot_height=500, plot_width=500,
title="Sensor-Chemical Distribution");
sensorBarPlot.xaxis.major_label_orientation = "vertical";
sensorChemicalDistSource = ColumnDataSource(data=dict(x=sensorBarKeys, y=sensorBarValues));
sensorBarPlot.vbar(x='x', top='y', width=0.5, source=sensorChemicalDistSource);


monthSlider = Slider(title="Month of the Year", value=4, start=4, end=12,step=4);
monthSlider.on_change('value',windAndDistUpdate);
curdoc().add_root(column(widgetbox(monthSlider,width=300),column(windDirPlot,sensorBarPlot)));
#curdoc().add_root(sensorBarPlot);


# ------------------------------------------------------------------------------
# ---------------- Create the main chemical responsibility plot ----------------
# ------------------------------------------------------------------------------
responsibility_pretext = PreText(text="""



Who's Responsible
-------------------
The below interactive plot is an attempt to find out who is responsible for which chemical release
The factories, Roadrunner, Kasios, Radiance and Indigo have been plotted as black squares
The sensors have been plotted as purple circles. The other circles indicate sensor readings.
The sensor reading points are plot in the opposite direction of the wind flowing from the factores
to sensors. The distance of the sensor reading point from the sensor represents the magnitude of
the sensor reading. Overall this gives a trailing effect based on the wind direction.
For instance in the below plot the red points indicate the sensor readings and the red points
around 6 indicate that based on the wind direction the chemical release has most probably come
from the direction of Kasios factory. All the chemicals can be analyzed seperately using the
buttons provided in the top right corner of the plot.

Analysis
----------
From the above plots it can be seen that
* Kasios is the major contributor to Methylosmolene. Due to the position of Kasios and RoadRunner,
it may seem that RoadRunner also produces Methylosmolene, but sensor 6, the closest to the two
factories comes to our rescue and points to Kasios
* RoadRunner is a major contributor to Chlorodinine. Again due to the position of Kasios and
RoadRunner it may seem confusing, but sensor 6 acts as the diffrentiator
* Kasios and Radiance are the major contributors for AGOC-3A
* Major contributor to Appluimonia is Indigo. Radiance may also be a minor contributor

""",
width=1000, height=450);
curdoc().add_root(responsibility_pretext);

chemicalFactoryPlot = figure(title="Methylosmolene Responsibility",
plot_width=1000, plot_height=750,x_range=(50,130),y_range=(-10,60));
chemicalFactoryPlot.title.text_color = '#E74C3C';
chemicalFactoryPlot.add_tools(TapTool());
factory_source = ColumnDataSource(data=dict(x=factory_loc_x,y=factory_loc_y,names=factory_names));
factories = chemicalFactoryPlot.square(x='x',y='y',size=15,fill_color='black',source=factory_source);
factories_labels = LabelSet(x='x', y='y', text='names', level='glyph',
x_offset=8, y_offset=8, render_mode='canvas',text_font_size='10pt',source=factory_source);
chemicalFactoryPlot.add_layout(factories_labels);
factorySensorLinesSource = ColumnDataSource(data=dict(xs=[],ys=[]));
chemicalFactoryPlot.multi_line(xs='xs',ys='ys',line_alpha=0.25,line_color='grey', line_width=10, source=factorySensorLinesSource);
factories.data_source.on_change('selected',selected_factory);


sensorAndMetData = joinSensorAndMetData(sensorData,metData);
sensorAndMetData = prepareAngularCoordinates(sensorAndMetData);
readings_x, readings_y = getReadingsByWindDirectionAndChemical(0,360,'Methylosmolene');
readings_color = ['#E74C3C']*len(readings_x);
readingsByWindAndChemicalSource = ColumnDataSource(data=dict(x=readings_x,y=readings_y,color=readings_color));
chemicalFactoryPlot.circle(x='x',y='y',color='color',size=5,alpha=0.75,source=readingsByWindAndChemicalSource);
plotSensors(chemicalFactoryPlot,0,0,10,'#9B5DE2');
meth_button = Button(label="Methylosmolene", width=150,height=50, button_type='danger');
meth_button.on_click(callback_meth_button);
chloro_button = Button(label="Chlorodinine", width=150,height=50, button_type='warning');
chloro_button.on_click(callback_chloro_button);
agoc_button = Button(label="AGOC-3A", width=150,height=50,button_type='success');
agoc_button.on_click(callback_agoc_button);
appl_button = Button(label="Appluimonia", width=150,height=50,button_type='primary');
appl_button.on_click(callback_appl_button);
curdoc().add_root(row(chemicalFactoryPlot,column(meth_button,chloro_button,agoc_button,appl_button)));

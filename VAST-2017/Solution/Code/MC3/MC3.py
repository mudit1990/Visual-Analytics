import os
import time
import pickle
import numpy as np
import pandas as pd
from scipy import misc;
import pickle

from bokeh.io import curdoc
from bokeh.colors import RGB
from bokeh.plotting import figure
from bokeh.layouts import row,column
from bokeh.models.sources import ColumnDataSource
from bokeh.models.widgets import RadioButtonGroup, Slider, Button, PreText

##### GET ALL CSV FILES IN PATH

Files = []
for File in os.listdir("./MC3_Image_data/"): #/MC3 Image Data/"):
	if File.endswith(".csv"):
		Files.append(File)

Files = sorted(Files)


##### SOME GLOBAL INITIALISATIONS

chosen_bands = ["B3", "B2", "B1"]

possible_bands = ["B1", "B2", "B3", "B4", "B5", "B6"]

dates = ["March 17 2014", "August 24 2014", "November 28 2014", "December 30 2014",
		 "February 15 2015", "June 24 2015", "September 12 2015", "November 15 2015",
		 "March 06 2016", "June 26 2016", "September 06 2016", "December 19 2016"]

CDSs = []


##### COMPRESS IMAGE DATA

def compressIm(Im, File):
	newX = list(range(326))*326
	newY = np.repeat(list(range(326)), 326)
	newDF = pd.DataFrame(data={"X" : newX, "Y" : newY})
	for band in possible_bands:
		color_band = Im[band]
		color_band = np.reshape(color_band, (651,651))
		newIm = []
		for row in range(0,651,2):
			tmp = []
			for col in range(0,651,2):
				tmp.append(np.mean(color_band[row:row+1, col:col+1]))
			newIm.append(tmp)
		newIm = np.array(newIm, dtype=np.uint8)
		imVec = np.reshape(newIm.T, np.prod(newIm.shape), -1)
		newDF[band] = imVec
	newCDS = ColumnDataSource(data=newDF)
	# saveName = File[-4:] + ".pickle"
	newDF.to_csv(File)
	print("Saved", File, " as csv.")
	return newCDS


##### READ INPUT DATA

start_time = time.time()

for File in Files:
	fileName = "./MC3_Image_data/" + File
	Im = pd.read_csv(fileName)
	CDSs.append(ColumnDataSource(data=Im))

end_time = time.time()

print("Time to load data =", end_time - start_time, " seconds.")

##### CALLBACKS

def time_callback(attr, old, new):
	new = int(new)
	start_time = time.time()
	R = CDSs[new - 1].data[chosen_bands[0]]
	G = CDSs[new - 1].data[chosen_bands[1]]
	B = CDSs[new - 1].data[chosen_bands[2]]
	plotCDS.data["colors"] = ["#" + format(r, "02X") + format(g, "02X") + format(b, "02X") for r,g,b in zip(R, G, B)]
	imageToShow.title.text = "Park Image on " + dates[new - 1]
	time_to_compute_colors = time.time() - start_time

	print("----- RENDERING -----")
	print("R = ", R[:5])
	print("G = ", G[:5])
	print("B = ", B[:5])
	print(plotCDS.data["colors"][:5])
	print("Colors computed in ", time_to_compute_colors, "seconds.")
	print("Sending to browser..")
	print("---------------------")

#####

def change_band_red(new):
	print("New red = ", possible_bands[new])
	chosen_bands[0] = possible_bands[new]

#####

def change_band_green(new):
	print("New green = ", possible_bands[new])
	chosen_bands[1] = possible_bands[new]

#####

def change_band_blue(new):
	print("New blue = ", possible_bands[new])
	chosen_bands[2] = possible_bands[new]

#####

def render():
	start_time = time.time()
	R = CDSs[time_slider.value - 1].data[chosen_bands[0]]
	G = CDSs[time_slider.value - 1].data[chosen_bands[1]]
	B = CDSs[time_slider.value - 1].data[chosen_bands[2]]
	plotCDS.data["colors"] = ["#" + format(r, "02X") + format(g, "02X") + format(b, "02X") for r,g,b in zip(R, G, B)]
	time_to_compute_colors = time.time() - start_time
	imageToShow.title.text = "Park Image on " + dates[time_slider.value - 1]

	print("----- RENDERING -----")
	print("R = ", R[:5])
	print("G = ", G[:5])
	print("B = ", B[:5])
	print(plotCDS.data["colors"][:5])
	print("Colors computed in ", time_to_compute_colors, "seconds.")
	print("Sending to browser..")
	print("---------------------")


# def saveLakeImage():
# 	im = misc.imread('boonsong_lake.jpg',mode='RGB');
# 	x = [];
# 	y = [];
# 	colors = [];
# 	for i in range(im.shape[0]):
# 		for j in range(im.shape[1]):
# 			xpos=j;
# 			ypos=im.shape[0]-i-1;
# 			if xpos>110 and xpos<310 and ypos>80 and ypos<360:
# 				x.append(xpos-110);
# 				y.append(ypos-80);
# 				colors.append("#" + format(im[i][j][0], "02X") + format(im[i][j][1], "02X") + format(im[i][j][2], "02X"));
# 	data = dict(x=x,y=y,colors=colors);
# 	with open('lake_image.pickle', 'wb') as handle:
# 	    pickle.dump(data, handle);

def getLakeImage():
	#saveLakeImage();
	# with open('lake_image.pickle', 'rb') as handle:
	#     data = pickle.load(handle);
	im = misc.imread('boonsong_lake.jpg',mode='RGB');
	x = [];
	y = [];
	colors = [];
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			xpos=j;
			ypos=im.shape[0]-i-1;
			if xpos>110 and xpos<310 and ypos>80 and ypos<360:
				x.append(xpos-110);
				y.append(ypos-80);
				colors.append("#" + format(im[i][j][0], "02X") + format(im[i][j][1], "02X") + format(im[i][j][2], "02X"));
	data = dict(x=x,y=y,colors=colors);
	return data;

def transformToBinary(l,Rw,Gw,Bw,c0,c1):
	colors = [];
	for i in range(l):
		if(Rw[i]<50 and Gw[i]<50): #and Gw[i]<50 and Bw[i]<50):
			colors.append(c0);
		else:
			colors.append(c1);
	return colors;

def insertIfValid(x,y,c,tr,x1,x2,y1,y2):
	if x>=x1 and x<=x2 and y>=y1 and y<=y2:
		if(c == '#FFFFFF'):
			tr[y-y1][x-x1]=0;
		else:
			tr[y-y1][x-x1]=1;

def getContigousIdx(tr,threshold,reverse,xlim,ylim):
	dp = np.zeros((ylim+1,xlim+1));
	for i in range(1,tr.shape[0]):
		if reverse == True:
			idx = tr.shape[0]-i-1;
		else:
			idx = i;
		dp[i] = dp[i-1]+1;
		dp[i] = dp[i]*tr[idx];
		if np.max(dp[i]>threshold):
			return idx;

def findVerticalTransition(source,plot,x1,x2,y1,y2):
	tr_array = np.zeros((y2-y1+1,x2-x1+1));
	x = source.data['x'];
	y = source.data['y'];
	colors = source.data['colors'];
	for i in range(len(x)):
		insertIfValid(x[i],y[i],colors[i],tr_array,x1,x2,y1,y2);
	threshold=5;
	mn = getContigousIdx(tr_array,threshold,False,x2-x1,y2-y1)-threshold;
	mx = getContigousIdx(tr_array,threshold,True,x2-x1,y2-y1)+threshold;
	plot.line(x=[x1,x2],y=[mn+y1,mn+y1],color='red',line_width=2);
	plot.line(x=[x1,x2],y=[mx+y1,mx+y1],color='red',line_width=2);

def trim_image(im,x1,x2,y1,y2):
	return im[(im['X']>=x1) & (im['X']<=x2) & (im['Y']>=y1) & (im['Y']<=y2)];

def animate_update():
	image_number = time_slider.value + 1
	if image_number == 13:
		image_number = 1
	time_slider.value = image_number

def animate():
	if play_pause_button.label == '► Play':
		play_pause_button.label = '❚❚ Pause'
		curdoc().add_periodic_callback(animate_update, 5000)
	else:
		play_pause_button.label = '► Play'
		curdoc().remove_periodic_callback(animate_update)

#####

pre_intro = PreText(text="""
Mudit Bhargava (mbhargava@umass.edu)
Priyadarshi Rath (priyadarshir@umass.edu)

VAST Challenge 2017 - Mini Challenge 3
MC3 provides us with 3 years of multi-spectral images of the reserve. In the following analysis we explore these images
to understand the changes in flora and fauna of the reserve and find possible anamolies or changes that may have lead to
the declining population of the blue pipit bird.


The first step was to understand the scale and orientation of the satellite images. For reference we have been given image
of Boonsong Lake (left plot) with its actual length. The plot on the right shows the satellite image of Boonsong lake.
The two horizontal red lines mark the extreme ends of the lake. The red lines were found analytically using dynamic
programming. The algorithm counts the number vertical contiguous lake (blue) pixels above and below the given pixel.
It then determines the first pixel from top and bottom that have more than 5 contiguous pixels ahead of it. These are
marked as boundaries of the lake. From the plot we can see, that difference between the 2 horizontal boundaries is
30 pixels, which corresponds to 3000 feet in the real world. Hence one pixel corresponds to 30m in the real world.
The location of Boonsong lake and orientation of satellite image was determined visually by comparing the left and
the right plots
""",
width=1200, height=310);
curdoc().add_root(pre_intro);

### plot the reference lake image ###
lakeImageSource = ColumnDataSource(data=getLakeImage());
lakeImagePlot = figure(x_range=[0,200], y_range=[0,280], title='Reference Image', plot_width=400, plot_height=400);
lakeImagePlot.rect(x='x',y='y',width=0.99, height=0.99,color='colors',source=lakeImageSource);

lake_img = pd.read_csv("image02_2014_08_24.csv");
lake_img['Y'] = 650 - lake_img["Y"];
lake_img = trim_image(lake_img,140,170,140,180);
Rw = lake_img['B5'].values; Gw = lake_img['B4'].values; Bw = lake_img['B2'].values;
lakeRefSource = ColumnDataSource(data={
"x" : lake_img["X"].values,
"y" : lake_img["Y"].values,
"colors" : transformToBinary(len(lake_img["X"]),Rw,Gw,Bw,'#0000FF','#FFFFFF')
#"colors" : ["#" + format(r, "02X") + format(g, "02X") + format(b, "02X") for r,g,b in zip(Rw,Gw,Bw)]
});
lakeRefPlot = figure(x_range=[140,170], y_range=[140,180], title='August 24 2014', plot_width=400, plot_height=400);
lakeRefPlot.rect(x='x',y='y',width=0.99, height=0.99,color='colors',source=lakeRefSource);
findVerticalTransition(lakeRefSource,lakeRefPlot,140,170,140,180);
#lakeRefPlot.line(x=[70,85],y=[mn,mn]);
#lakeRefPlot.line(x=[70,85],y=[mx,mx]);
curdoc().add_root(row(lakeImagePlot,lakeRefPlot));

##### INITIALISE PLOTS

R = CDSs[0].data[chosen_bands[0]]
G = CDSs[0].data[chosen_bands[1]]
B = CDSs[0].data[chosen_bands[2]]

plotCDS = ColumnDataSource(data={
"x" : CDSs[0].data["X"],
"y" : 325 - CDSs[0].data["Y"],
"colors" : ["#" + format(r, "02X") + format(g, "02X") + format(b, "02X") for r,g,b in zip(R,G,B)]
})

pre_multispectral = PreText(text="""
The next step is to view the actual images and analyse what features are present in the park. For this purpose, we created an
interface shown below. There is a main panel that shows the image, and there are various controls beside it. There are three rows
of mappings. The top row represents which band the user wants to map to red, the middle row represents the same for green, and the
bottom row represents the same for blue. For example, selecting B5 in the top row, B4 in the middle row and B2 in the bottom row
corresponds to B5,B4,B2 -> Red,Green,Blue.

Users may select any mapping they want.

We also have a slider control for viewing any of the 12 images in the dataset. In addition, we have a play/pause button that loops
through the images, so that the user can view changes over time.


Some of the features present are listed below:

* There are a number of lakes clearly visible. The evidence is provided by seasonal freezing observed during the months of November
  to March. To observe the lakes, the following mapping is helpful: B1,B5,B6 -> Red,Green,Blue.
  During summer, the lakes are black, and during winter, they are red in color.
  Another mapping that helps identify lakes, is B4,B3,B2. Under this mapping, lakes appear black.

* There is a village near the north-western region of image. This is clear from the unusually dense vegetation cover observed during
  the month of September, and little vegetation in other months. This periodic increase and decrease in vegetation cover could imply
  kind of farming activity.
  To observe this, the user can use the following mapping : B4,B3,B2 -> Red,Green,Blue.

  Another mapping that helps is : B5,B4,B2 -> Red,Green,Blue, which shows buildings and settlements as pink/purple, and thickly
  vegetated areas as bright green.

* There is some sort of road that roughly bisects the reserve. This is seen clearly as a curved line running vertically through the
  middle of the image, in all images except those that are occluded by clouds. This road can be seen in all bands.

* In the southern regions of the image, there a patch of mostly barren land as shown by B5,B4,B2. It is barren(purple) during summer
  months and dark green during winter months. This suggests that some seasonal plants grow in the winter in that region. Also, this
  region does not freeze, so it is probably a desert.
""",
width=1200, height=550)

##### MAIN PANEL FOR IMAGE DISPLAY

titleStr = "Park Image on " + dates[0]
imageToShow = figure(x_range=[0,325], y_range=[0,325], title=titleStr)
imageToShow.rect(x="x", y="y", width=0.99, height=0.99, color="colors", source=plotCDS)


##### COLOR BAND MAPPINGS

redText = PreText(text="""Red:""", width=60, height=15)
redBandMapping = RadioButtonGroup(labels=possible_bands, active=2, name="Red")
greenText = PreText(text="""Green:""", width=60, height=15)
greenBandMapping = RadioButtonGroup(labels=possible_bands, active=1, name="Green")
blueText = PreText(text="""Blue:""", width=60, height=15)
blueBandMapping = RadioButtonGroup(labels=possible_bands, active=0, name="Blue")


##### COLOR BAND MAPPING INTERACTIONS

redBandMapping.on_click(change_band_red)
greenBandMapping.on_click(change_band_green)
blueBandMapping.on_click(change_band_blue)


##### BUTTON TO MATERIALISE CHANGES

renderButton = Button(label="Confirm Mappings", button_type="success")
renderButton.on_click(render)


##### TIME SLIDER

time_slider = Slider(start=1, end=12, value=1, step=1, title="Image Number(between 1 and 12)")
time_slider.on_change("value", time_callback)


##### PLAY/PAUSE

play_pause_button = Button(label='► Play', width=60)
play_pause_button.on_click(animate)

##### COLOR MAPPING DESCRIPTIONS

mapText = PreText(text="""
Mapping Guide:
B1 : Blue
B2 : Green
B3 : Red
B4 : Near Infrared
B5 : ShortWave Infrared 1
B6 : ShortWave Infrared 2
""", width=200, height=50)

##### LAYOUT

RButton = row(redText, redBandMapping)
GButton = row(greenText, greenBandMapping)
BButton = row(blueText, blueBandMapping)

color_mapping_layout = column(RButton, GButton, BButton, renderButton, time_slider, play_pause_button, mapText)
layout = column(pre_multispectral, row(imageToShow, color_mapping_layout))
curdoc().add_root(layout)

feature_changes_Text = PreText(text="""
Based on the above image, the following changes are noticed over time:

* Vegetation cover decreases during winter months, seen under bands B4,B3,B2. This could be due to the fact that the trees shed their
  leaves in autumn.

* The lakes freeze during winter months, as already stated above. In addition to this usual seasonal pattern, there is an unusual
  pattern noticed. In June 2015(Image 6), under bands B4,B3,B2, the water in all major lakes appears green in color. This is strange,
  since liquid water in B4,B3,B2 should appear black, and it does so indeed in 2014(Image 2) and 2016(Image 10). This suggests the
  introduction of some impurities into the lake. This strange behaviour, however, seems gone pretty fast, since September 2015
  (Image 7) shows reasonably clear waters.

* Another change in the water bodies is noticed in winter of 2016(Image 12), under bands B1,B5,B6. Normally, frozen water should
  show up in a red color, but in this image, the water shows a mixture between blue and pink. This is peculiar, and a similar
  conclusion can be drawn for this event.

* Some fires are seen during in the winter of 2014(Image 4), seen under bands B5,B4,B2. These are too big to be accidents, and are
  probably intentional. This could be just the town people holding some public event around bonfires, or campers in the forest
  starting fires during the night.
""",
width=1200,height=300)

curdoc().add_root(feature_changes_Text)
############################################################################################################################

##### NDVI

# B4 = CDSs[0].data["B4"]
# B3 = CDSs[0].data["B3"]
# pixel_intensities = np.array((B4 - B3)/(B4 + B3))
# pixel_intensities[np.argwhere(np.isnan(pixel_intensities))] = 0
# pixel_intensities = pixel_intensities + np.amin(pixel_intensities)
# pixel_intensities = np.array((pixel_intensities/np.amax(pixel_intensities)) * 255, dtype=np.uint8)

# NDVI_CDS = ColumnDataSource(data={
# "x" : CDSs[0].data["X"],
# "y" : 325 - CDSs[0].data["Y"],
# "colors" : ["#" + format(x, "02X") + format(x, "02X") + format(x, "02X") for x in pixel_intensities]
# })

# def NDVI_callback(attr, old, new):
# 	start_time = time.time()
# 	B4 = CDSs[new - 1].data["B4"]
# 	B3 = CDSs[new - 1].data["B3"]
# 	pixel_intensities = np.array((B4 - B3)/(B4 + B3))
# 	pixel_intensities[np.argwhere(np.isnan(pixel_intensities))] = 0
# 	pixel_intensities = pixel_intensities + np.amin(pixel_intensities)
# 	pixel_intensities = np.array((pixel_intensities/np.amax(pixel_intensities)) * 255, dtype=np.uint8)
# 	NDVI_CDS.data["colors"] = ["#" + format(x, "02X") + format(x, "02X") + format(x, "02X") for x in pixel_intensities]
# 	NDVI.title.text = "Park NDVI Image on " + dates[new - 1]
# 	time_to_compute_colors = time.time() - start_time

# 	print("----- RENDERING -----")
# 	print("Colors computed in ", time_to_compute_colors, "seconds.")
# 	print("Sending to browser..")
# 	print("---------------------")

# ##### MAIN PANEL FOR IMAGE DISPLAY

# titleStr = "Park NDVI Image on " + dates[0]
# NDVI = figure(x_range=[0,325], y_range=[0,325], title=titleStr)
# NDVI.rect(x="x", y="y", width=0.99, height=0.99, color="colors", source=NDVI_CDS)

# NDVI_slider = Slider(start=1, end=12, value=1, step=1, title="Image Number(between 1 and 12)")
# NDVI_slider.on_change("value", NDVI_callback)

# ##### PLAY/PAUSE

# def animate_update_NDVI():
# 	image_number = NDVI_slider.value + 1
# 	if image_number == 13:
# 		image_number = 1
# 	NDVI_slider.value = image_number

# def animate_NVDI():
# 	if NDVI_play_pause_button.label == '► Play':
# 		NDVI_play_pause_button.label = '❚❚ Pause'
# 		curdoc().add_periodic_callback(animate_update_NDVI, 5000)
# 	else:
# 		NDVI_play_pause_button.label = '► Play'
# 		curdoc().remove_periodic_callback(animate_update_NDVI)

# NDVI_play_pause_button = Button(label='► Play', width=60)
# NDVI_play_pause_button.on_click(animate_NVDI)


# ##### LAYOUT

# layout2 = row(NDVI, column(NDVI_slider, NDVI_play_pause_button))
# curdoc().add_root(layout2)

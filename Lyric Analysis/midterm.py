from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import column,row,gridplot
from bokeh.models import ColumnDataSource;
from bokeh.models.tools import HoverTool;
import pandas as pd;
import numpy as np;
from collections import Counter;
from bokeh.models.markers import Circle
from bokeh.models.widgets import PreText
from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def createData(df_lyrics,df_time):
    all_artist = pd.unique(df_lyrics['artist']);
    a_list = [];
    x_list = [];
    y_list = [];
    word_list = [];
    count_list = [];
    years_list = [];
    year_tracks = [];
    cols = df_time.columns.values;
    years = [a[1] for a in cols];
    for artist in all_artist:
        a_list.append(artist);
        artist_df = df_lyrics[df_lyrics['artist'] == artist];
        songList = artist_df['Word_list'].values;
        songs = [item for sublist in songList for item in sublist];
        avg_words = len(songs)/len(songList);
        avg_word_len = sum(len(word) for word in songs) / len(songs);
        x_list.append(avg_words);
        y_list.append(avg_word_len);
        songCount = Counter(songs).most_common(50);
        words = [];
        counts = [];
        for k,v in songCount:
            words.append(k);
            counts.append(v);
        word_list.append(words);
        count_list.append(counts);
        # extract the track count based on year
        track_count = df_time.loc[artist].values;
        year_tracks.append(track_count);
        years_list.append(years);
    return a_list,x_list,y_list,word_list,count_list,years_list,year_tracks;


def callback(attr,old,new):
    print("oh look I just got tapped");
    pts = new['1d']['indices'];
    if(len(pts) == 0):
        r = 0;
    else:
        r = pts[0];
    data = dict(x=word_list[r],y=count_list[r]);
    artist_wordfreq_plot.x_range.factors = word_list[r];
    artist_wordfreq_plot.y_range.start=-0.5;
    artist_wordfreq_plot.y_range.end= max(count_list[r])+500;
    artists = artist_xy_source.data['artist'];
    artist_wordfreq_plot.title.text = "Artist: "+artists[r];
    artist_wordfreq_source.data = data;
    xs = artist_overtime_source.data['xs'];
    ys = artist_overtime_source.data['ys'];
    colors = artist_overtime_source.data['colors'];
    alpha = artist_overtime_source.data['alpha'];
    width = artist_overtime_source.data['width'];
    colors = ['grey']*len(colors);
    colors[r] = 'firebrick';
    alpha = [0.2]*len(years_list);
    alpha[r] = 1;
    width = [2]*len(width);
    width[r] = 5;
    artist_overtime_source.data = dict(xs=xs,ys=ys,colors=colors,alpha=alpha,width=width);
    artist_overtime_plot.title.text = "Artist: "+artists[r];

# read dataset
df = pd.read_csv("lyrics.csv")
# get each artist name, and their counts
ct = Counter(df['artist'])
# take artists with most songs
top_artists = []
for i in np.arange(0,50):
    top_artists.append(ct.most_common()[i][0])
# make new dataframe with top artists
final_df = df.loc[df['artist'].isin(top_artists)]
final_df = final_df.dropna(axis = 0, how='any')

# tokenize and stem the lyrics of each song, finally store in dataframe
tokenizer = RegexpTokenizer(r'\w+') # to omit punctuations
wl = WordNetLemmatizer() # to get root word
stop_words = set(stopwords.words('english'))

# word list of each song, will add into dataframe in last step
Word_List = []

for i in final_df.index:
    # get the lyrics
    Lyrics = final_df.loc[i, 'lyrics']
    # tokenize lyrics, omit punctuations using regexp tokenizer
    words  = tokenizer.tokenize(Lyrics)
    words = [word.lower() for word in words]
    # remove stopwords and get the root using lemmatize
    filtered_lyrics = [wl.lemmatize(w) for w in words if not w in stop_words]
    Word_List.append(filtered_lyrics)

final_df = final_df.astype('object')
All_lyrics = final_df['lyrics']
final_df.drop(['lyrics'], axis=1, inplace=True)
final_df['Word_list'] = pd.Series(Word_List, index=final_df.index)

# data load from pickle file
#fulldata = pd.read_pickle('clean_song_data.pickle');
fulldata = final_df;
# creating the time series specific data
artist_overtime = fulldata[['artist','year']].copy(deep=True);
artist_overtime['val']=1;
artist_overtime_group = artist_overtime.groupby(['artist','year']).sum();
artist_overtime = artist_overtime_group.unstack(level=1);
artist_overtime = artist_overtime.fillna(0);
# creating artist_lyrics data
artist_lyrics = fulldata[['artist','Word_list']];
alist,xlist,ylist,word_list,count_list,years_list,year_tracks = createData(artist_lyrics,artist_overtime);
artist_xy_source = ColumnDataSource(data=dict(artist=alist,x=xlist,y=ylist));
artist_wordfreq_source = ColumnDataSource(data=dict(x=word_list[0],y=count_list[0]));
hover = HoverTool(tooltips=[
    ("Artist", "@artist")
]);

pre_intro = PreText(text="""
Mudit Bhargava (mbhargava@umass.edu)
Priyadarshi Rath (priyadarshir@umass.edu)
We have chosen to explore the lyrics dataset. The data was accessed from https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics.
The data consists of lyrics from multiple artists from 1970 to 2016 along with the genre they belong to. In all the data has 18231
artists and 362237. The data set was cleaned up by removing the n/a rows, and removing all songs where the genre was other/unavailable.
After that we cleaned up the lyrics by Stemming and removing all the english stop words. Post that we have taken the top 50 artists
(by number of songs) for our analysis. The analyzed data contains 21247 songs""",
width=1200, height=150);
curdoc().add_root(pre_intro);

artist_xy_plot = figure(plot_width=450, plot_height=450, tools=[hover,'tap','reset'],toolbar_location='above');
artist_xy_circle = artist_xy_plot.circle('x','y',size=15,source=artist_xy_source,fill_alpha=0.7);
# setting unselected selected render properties
artist_xy_circle.selection_glyph = Circle(fill_color='firebrick', fill_alpha=0.8);
artist_xy_circle.nonselection_glyph = Circle(fill_color='yellow', fill_alpha=0.2);
# setting artist_xy_plot properties
artist_xy_plot.xaxis.axis_label="Average Words/Song";
artist_xy_plot.yaxis.axis_label="Average Word Length";

artist_wordfreq_plot = figure(plot_width=1000, plot_height=450, x_range=artist_wordfreq_source.data['x'],toolbar_location='above', tools=['xpan, reset']);
artist_wordfreq_plot.vbar(x='x',top='y', width=0.25, source=artist_wordfreq_source);
artist_wordfreq_plot.line(x='x',y='y',line_width=5,line_color='green',line_alpha=0.4,source=artist_wordfreq_source);
# setting artist_wordfreq_source plot properties
artist_wordfreq_plot.min_border_left = 100;
artist_wordfreq_plot.xaxis.major_label_orientation = "vertical";
artist_wordfreq_plot.xaxis.major_label_text_font_size='13pt';
artist_wordfreq_plot.yaxis.axis_label="Frequency";
artist_wordfreq_plot.title.text_color = 'firebrick';
artist_wordfreq_plot.title.text_font_size='13pt';

# configuring the call-back for on-tap
artist_xy_circle.data_source.on_change('selected',callback);

artist_overtime_source = ColumnDataSource(data=dict(xs=years_list,ys=year_tracks,
colors=(['grey']*len(years_list)),
alpha=([0.2]*len(years_list)),
width=([2]*len(years_list))));
artist_overtime_plot = figure(plot_width=600, plot_height=400);
artist_overtime_plot.multi_line(xs='xs',ys='ys',line_alpha='alpha',line_color='colors', line_width='width', source=artist_overtime_source);
artist_overtime_plot.xaxis.axis_label = "Year";
artist_overtime_plot.yaxis.axis_label = "Number of Tracks";
artist_overtime_plot.title.text_color = 'firebrick';
artist_overtime_plot.title.text_font_size='13pt';
artist_overtime_plot.min_border_top = 50;

pre_explore = PreText(text="""The 3 linked graphs attempt to explore the data. The first scatter plot shows the
distribution of every artist by avergae word length and avergae number of words in the song.
The hover tool will help you identify, the artist that every point represents. The 2nd and
3rd plot are linked to this scatter plot. Based on the artist selected the 2nd plot shows
the frequency distribution of words used by artist. The 3rd plot shows the artist activity
from 1970s to 2016
------------------------------------------------------------------------------------------
Some interesting observations
* Eminem songs have a lot more words in them
* Chris Brown's most popular words are 'girl','love','oh'""",
width=750, height=250);

layout = row(artist_xy_plot,artist_wordfreq_plot);
curdoc().add_root(layout);
curdoc().add_root(row(artist_overtime_plot, pre_explore));

topic_pre = PreText(text="""Topic modeling
Definition : a method for finding a group of words (i.e topic) from a collection of documents that best represents the information
in the collection. Our motivation for performing topic modeling was to find important topical words used by popular artists,
so that we could analyse their songs based on their distinctive vocabulary. Also, some artists might use similar vocabulary, so we
can cluster them. Procedure: Used sklearn’s CountVectorizer to get term frequencies, and on the term vectors, preformed
Latent Dirichlet Allocation to get the important topics. Finally, combined the LDA probabilities for each artist, and generated a heatmap.
---------------------------------------------------------------------------------------------------------------------------"
From the heatmap it can be seen that one topic (based on common words like love,heart,never) is predominant across all artists.
Probably these are commonly used words in songs. One of the topics were found to have all the Rap singers like Eminem, 50 cents
game and E-40""", width=1200, height=150);

final_df.index = np.arange(final_df.shape[0])

def my_tokenize(text):
    tokens = tokenizer.tokenize(text)
    return tokens

no_topics = 10

tf_vectorizer = CountVectorizer(min_df=10, tokenizer = my_tokenize, stop_words=stop_words, lowercase=True)
tf = tf_vectorizer.fit_transform(All_lyrics.values)
tf_feature_names = tf_vectorizer.get_feature_names()

# Run LDA
lda = LatentDirichletAllocation(n_topics=10)
topics = lda.fit_transform(tf)

# Get words from important topics
def return_topics(model, feature_names, no_top_words):
    Topics = []
    for topic_idx, topic in enumerate(model.components_):
        Topics.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return Topics

Topics = return_topics(lda, tf_feature_names, 5)

HeatMapData = []

for artist in pd.unique(final_df['artist']):
    rows = final_df.loc[final_df['artist'] == artist].index
    total = np.sum(topics[rows, :], axis=0)
    total = total / np.sum(total)
    HeatMapData.append(total)

HeatMapData = np.asarray(HeatMapData)

HeatMapDF = pd.DataFrame(data=HeatMapData, index = pd.unique(final_df['artist']), columns=Topics)

y = list(np.repeat(HeatMapDF.index.values, len(HeatMapDF.columns.values)))
x = list(HeatMapDF.columns.values) * len(HeatMapDF.index.values)

factors = HeatMapDF.index

hover_heatmap = HoverTool(tooltips=[('artist', '@y'), ('topic', '@x')])

hm = figure(title="Topic Modelling Heatmap", toolbar_location=None,
            x_range = list(HeatMapDF.columns.values), y_range = list(HeatMapDF.index.values),plot_width=750, plot_height=750)
hm.add_tools(hover_heatmap)

color_map =  ['#EBF5FB','#D6EAF8' ,'#AED6F1','#85C1E9','#5DADE2' ,'#3498DB','#2E86C1','#2874A6','#21618C','#21618C', '#21618C']
colors = []

max_val = HeatMapDF.values.max()

for i in HeatMapDF.index.values:
    row = HeatMapDF.loc[i].values
    ID = np.floor(row/(max_val/10))
    for j in ID:
        colors.append(color_map[np.int(j)])

hm.rect(x, y, color=colors, width=1, height=1)
hm.xaxis.major_label_orientation = "vertical";
hm.xaxis.major_label_text_font_size='12pt';
curdoc().add_root(topic_pre)
curdoc().add_root(hm);
curdoc().title = "MidTerm"

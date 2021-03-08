import os
import base64
import re
import pickle

from pandas import DataFrame
import numpy as np
import pandas as pd
import xgboost as xgb
import zipcodes
from datetime import datetime

pd.options.mode.chained_assignment = None  # default='warn'
from urllib.request import urlopen
from sklearn.cluster import KMeans
from dash.dependencies import Input, Output, State
import plotly.express as px
import json
import dash
import plotly.io as pio
import json
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import requests
import io
from sklearn.preprocessing import MinMaxScaler

from dash_extensions import Download
from dash_extensions.snippets import send_data_frame

# training = pd.read_csv('adv_df.csv')

christine_subset = pd.read_csv('veterans_christine.csv')
df = pd.read_csv(
    'https://raw.githubusercontent.com/cmhamakawa/Data_Science_NBA_Project/master/Datasets/latest_dataframe.csv')

goats = ['Aaron Gordon', 'Al Horford', 'Allan Houston', 'Allen Crabbe',
         'Allen Iverson', 'Alonzo Mourning', 'Andre Drummond',
         'Andre Iguodala', 'Andrei Kirilenko', 'Andrew Bynum',
         'Andrew Wiggins', 'Anfernee Hardaway', 'Antawn Jamison',
         'Anthony Davis', 'Baron Davis', 'Ben Simmons', 'Ben Wallace',
         'Bismack Biyombo', 'Blake Griffin', 'Bradley Beal',
         'Brandon Ingram', 'Brandon Knight', 'Brandon Roy', 'Brook Lopez',
         'Buddy Hield', 'CJ McCollum', 'Caris LeVert', 'Carlos Boozer',
         'Carmelo Anthony', 'Chandler Parsons', 'Chris Bosh', 'Chris Paul',
         'Chris Webber', 'Clint Capela', 'Cody Zeller', "D'Angelo Russell",
         'Damian Lillard', 'Danilo Gallinari', 'Danny Green', 'David Lee',
         'David Robinson', 'DeAndre Jordan', 'DeMar DeRozan',
         'DeMarcus Cousins', 'DeMarre Carroll', 'Deron Williams',
         'Derrick Favors', 'Derrick Rose', 'Devin Booker', 'Dirk Nowitzki',
         'Domantas Sabonis', 'Draymond Green', 'Dwight Howard',
         'Dwyane Wade', 'Eddie Jones', 'Elton Brand', 'Enes Kanter',
         'Eric Bledsoe', 'Eric Gordon', 'Evan Fournier', 'Evan Turner',
         'Fred VanVleet', 'Gary Harris', 'George Hill',
         'Giannis Antetokounmpo', 'Gilbert Arenas', 'Gordon Hayward',
         'Gorgui Dieng', 'Grant Hill', 'Greg Monroe', 'Hakeem Olajuwon',
         'Harrison Barnes', 'Hassan Whiteside', 'Ian Mahinmi',
         'Jabari Parker', 'Jalen Rose', 'Jamal Murray', 'James Harden',
         'James Johnson', 'Jason Kidd', 'Jaylen Brown', 'Jeff Teague',
         'Jerami Grant', "Jermaine O'Neal", 'Jimmy Butler', 'Joakim Noah',
         'Joe Harris', 'Joe Johnson', 'Joel Embiid', 'John Wall',
         'Jrue Holiday', 'Julius Randle', 'Juwan Howard', 'Karl Malone',
         'Karl-Anthony Towns', 'Kawhi Leonard', 'Keith Van Horn',
         'Kemba Walker', 'Kent Bazemore', 'Kentavious Caldwell-Pope',
         'Kenyon Martin', 'Kevin Durant', 'Kevin Garnett', 'Kevin Love',
         'Khris Middleton', 'Klay Thompson', 'Kobe Bryant', 'Kyle Lowry',
         'Kyrie Irving', 'LaMarcus Aldridge', 'LeBron James', 'Luol Deng',
         'Malcolm Brogdon', 'Marc Gasol', 'Michael Finley',
         'Michael Jordan', 'Michael Redd', 'Mike Conley', 'Myles Turner',
         'Nicolas Batum', 'Otto Porter', 'Pascal Siakam', 'Patrick Ewing',
         'Pau Gasol', 'Paul George', 'Paul Millsap', 'Paul Pierce',
         'Rashard Lewis', 'Rasheed Wallace', 'Ray Allen', 'Reggie Jackson',
         'Richard Hamilton', 'Ricky Rubio', 'Robert Covington',
         'Roy Hibbert', 'Rudy Gay', 'Rudy Gobert', 'Russell Westbrook',
         'Ryan Anderson', 'Scottie Pippen', 'Serge Ibaka',
         "Shaquille O'Neal", 'Shawn Marion', 'Stephen Curry',
         'Stephon Marbury', 'Steve Francis', 'Steven Adams', 'Terry Rozier',
         'Tim Duncan', 'Timofey Mozgov', 'Tobias Harris', 'Tony Parker',
         'Tracy McGrady', 'Tristan Thompson', 'Tyler Johnson',
         'Victor Oladipo', 'Vince Carter', 'Wesley Matthews', 'Yao Ming',
         'Zach LaVine', 'Zach Randolph']
yoe_1 = {'PF': 0,
         'C': 1,
         'PF-C': 2,
         'SF-PF': 3,
         'PG': 4,
         'SF-SG': 5,
         'SF': 6,
         'PG-SG': 7,
         'C-PF': 8,
         'SG-SF': 9,
         'SG': 10,
         'SG-PG': 11}

yoe_2 = {'SF-C': 0,
         'PF-SF': 1,
         'SF-SG': 2,
         'PF': 3,
         'SG-PG': 4,
         'SG-PF': 5,
         'C': 6,
         'SG-SF': 7,
         'C-PF': 8,
         'SG': 9,
         'SF': 10,
         'PG-SF': 11,
         'PG-SG': 12,
         'SF-PF': 13,
         'PF-C': 14,
         'PG': 15}

map_years = {1991: 0,
             1992: 1,
             1993: 2,
             1994: 3,
             1995: 4,
             1996: 5,
             1997: 6,
             1998: 7,
             1999: 8,
             2000: 9,
             2001: 10,
             2002: 11,
             2003: 12,
             2004: 13,
             2005: 14,
             2006: 15,
             2007: 16,
             2008: 17,
             2009: 18,
             2010: 19,
             2011: 20,
             2012: 21,
             2013: 22,
             2014: 23,
             2015: 24,
             2016: 25,
             2017: 26,
             2018: 27,
             2019: 28,
             2020: 29}

image_filename = 'NBA_pic.png'  # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])
server = app.server

app.title = 'Predicting NBA Player Salaries'

with open('rookie_RMSE.json', 'r') as f:
    fig1 = pio.from_json(f.read())

with open('veteran_RMSE.json', 'r') as f:
    fig2 = pio.from_json(f.read())

app.layout = html.Div(children=[
    html.H1('Predicting NBA Player Salaries',
            style={'font-weight': 'bold', 'font-size': '360%', 'padding-left': '2px'}),
    html.H4('Created by Adhvaith Vijay - The Data Science Union at UCLA',
            style={'font-weight': 'bold', 'padding-left': '3px', 'font-size': '230%', 'font-style': 'italic'}),
    html.Br(),
    html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))], style={'text-align': 'center'}),
    html.Br(),
    html.Br(),
    html.Div([html.H2(children='There Are Two Distinct Models: Rookies and Veterans',
                      style={'text-align': 'center', 'font-weight': 'bold', 'font-size': '270%'})]),
    html.Div([dcc.Graph(id='Rookie', figure=fig1)]),
    html.Div([dcc.Graph(id='Veteran', figure=fig2)]),
    html.Div([
        html.H2(children='Predicting Salaries With XGBoost', style={'font-size': '250%'}),
        # dcc.Markdown('''Achieved Overall Root Mean Square Error (RMSE) of 895,303.2''',
        #              style={'font-weight': 'bold', 'padding-left': '3px', 'font-size': '160%'}),
        dcc.Markdown('''
                     If an input field (e.g. "Year's of Experience") is disabled, this means it does not 
                     contribute as a predictor for the model.
                     ''',
                     style={"white-space": "pre", 'font-weight': 'normal', 'font-size': '140%'}),
        dcc.Markdown('''
                     If you want to input values for a hypothetical player, click "Clear All Input"
                     and proceed to enter information for all fields except "Player's Name".
                     ''',
                     style={"white-space": "pre", 'font-weight': 'normal', 'font-size': '140%'}),
        # html.Label(["Player's Name: ",
        #             dcc.Input(id='name', value='Zion Williamson', type='text')], style={'font-weight': 'bold'}),
        html.Label(["Player's Name",
                    dcc.Dropdown(id="name",
                                 options=[{'label': k, 'value': k} for k in
                                          ['Mahmoud Abdul-Rauf', 'Mark Acres', 'Michael Adams',
                                           'Mark Aguirre', 'Danny Ainge', 'Mark Alarie', 'Greg Anderson',
                                           'Nick Anderson', 'Ron Anderson', 'Willie Anderson', 'Keith Askins',
                                           'Thurl Bailey', 'Ken Bannister', 'Charles Barkley', 'Dana Barros',
                                           'John Battle', 'William Bedford', 'Larry Bird', 'Rolando Blackman',
                                           'Lance Blanks', 'Mookie Blaylock', 'Muggsy Bogues', 'Manute Bol',
                                           'Sam Bowie', 'Frank Brickowski', 'Scott Brooks', 'Chucky Brown',
                                           'Dee Brown', 'Tony Brown', 'Mark Bryant', 'Matt Bullard',
                                           'Willie Burton', 'Michael Cage', 'Tony Campbell', 'Antoine Carr',
                                           'Terry Catledge', 'Cedric Ceballos', 'Tom Chambers', 'Rex Chapman',
                                           'Maurice Cheeks', 'Derrick Coleman', 'Bimbo Coles',
                                           'Lester Conner', 'Anthony Cook', 'Wayne Cooper', 'Tyrone Corbin',
                                           'Terry Cummings', 'Dell Curry', 'Quintin Dailey', 'Brad Daugherty',
                                           'Brad Davis', 'Terry Davis', 'Walter Davis', 'Johnny Dawkins',
                                           'Vlade Divac', 'Sherman Douglas', 'Greg Dreiling', 'Larry Drew',
                                           'Clyde Drexler', 'Kevin Duckworth', 'Joe Dumars', 'Ledell Eackles',
                                           'Blue Edwards', 'James Edwards', 'Kevin Edwards', 'Craig Ehlo',
                                           'Mario Elie', 'Sean Elliott', 'Dale Ellis', 'Pervis Ellison',
                                           'A.J. English', 'Alex English', 'Patrick Ewing', 'Dave Feitl',
                                           'Duane Ferrell', 'Danny Ferry', 'Vern Fleming', 'Sleepy Floyd',
                                           'Greg Foster', 'Kevin Gamble', 'Winston Garland', 'Tom Garrick',
                                           'Kenny Gattison', 'Tate George', 'Kendall Gill', 'Armen Gilliam',
                                           'Gerald Glass', 'Mike Gminski', 'Dan Godfread', 'Gary Grant',
                                           'Harvey Grant', 'Horace Grant', 'Jeff Grayer', 'A.C. Green',
                                           'Rickey Green', 'Sidney Green', 'Darrell Griffith', 'Tom Hammonds',
                                           'Bob Hansen', 'Tim Hardaway', 'Derek Harper', 'Ron Harper',
                                           'Scott Hastings', 'Hersey Hawkins', 'Steve Henson', 'Rod Higgins',
                                           'Craig Hodges', 'Dave Hoppen', 'Dennis Hopson', 'Jeff Hornacek',
                                           'Jay Humphries', 'Mark Jackson', 'Dave Jamerson', 'Henry James',
                                           'Les Jepsen', 'Buck Johnson', 'Eddie Johnson', 'Kevin Johnson',
                                           'Magic Johnson', 'Vinnie Johnson', 'Michael Jordan', 'Shawn Kemp',
                                           'Steve Kerr', 'Jerome Kersey', 'Alec Kessler', 'Bo Kimble',
                                           'Bernard King', 'Stacey King', 'Joe Kleine', 'Negele Knight',
                                           'Jon Koncak', 'Bill Laimbeer', 'Jerome Lane', 'Andrew Lang',
                                           'Jim Les', 'Fat Lever', 'Cliff Levingston', 'Reggie Lewis',
                                           'Marcus Liberty', 'Todd Lichti', 'Alton Lister', 'Brad Lohaus',
                                           'Grant Long', 'Dan Majerle', 'Jeff Malone', 'Karl Malone',
                                           'Moses Malone', 'Danny Manning', 'Vernon Maxwell', 'Travis Mays',
                                           'George McCloud', 'Tim McCormick', 'Rodney McCray',
                                           'Xavier McDaniel', 'Kevin McHale', 'Derrick McKey',
                                           'Nate McMillan', 'Reggie Miller', 'Terry Mills', 'Sam Mitchell',
                                           'Chris Morris', 'John Morton', 'Chris Mullin', 'Tod Murphy',
                                           'Jerrod Mustaf', 'Larry Nance', 'Ed Nealy', 'Johnny Newman',
                                           'Ken Norman', 'Charles Oakley', 'Alan Ogg', 'Hakeem Olajuwon',
                                           'Brian Oliver', 'Walter Palmer', 'Robert Parish', 'John Paxson',
                                           'Kenny Payne', 'Gary Payton', 'Will Perdue', 'Sam Perkins',
                                           'Tim Perry', 'Chuck Person', 'Jim Petersen', 'Ricky Pierce',
                                           'Ed Pinckney', 'Scottie Pippen', 'Olden Polynice', 'Terry Porter',
                                           'Paul Pressey', 'Mark Price', 'Kevin Pritchard', 'Brian Quinnett',
                                           'Kurt Rambis', 'Blair Rasmussen', 'J.R. Reid', 'Jerry Reynolds',
                                           'Glen Rice', 'Pooh Richardson', 'Mitch Richmond', 'Doc Rivers',
                                           'Fred Roberts', 'Alvin Robertson', 'David Robinson',
                                           'Rumeal Robinson', 'Dennis Rodman', 'Delaney Rudd', 'John Salley',
                                           'Ralph Sampson', 'Mike Sanders', 'Danny Schayes',
                                           'Dwayne Schintzius', 'Detlef Schrempf', 'Byron Scott',
                                           'Dennis Scott', 'Rony Seikaly', 'Brian Shaw', 'Lionel Simmons',
                                           'Scott Skiles', 'Charles Smith', 'Kenny Smith', 'Michael Smith',
                                           'Otis Smith', 'Tony Smith', 'Rory Sparrow', 'Felton Spencer',
                                           'John Starks', 'John Stockton', 'Rod Strickland', 'Jon Sundvold',
                                           'Terry Teagle', 'Isiah Thomas', 'LaSalle Thompson',
                                           'Mychal Thompson', 'Otis Thorpe', 'Sedale Threatt',
                                           'Wayman Tisdale', 'Tom Tolbert', 'Andy Toolson', 'Trent Tucker',
                                           'Andre Turner', 'Jeff Turner', 'Kelvin Upshaw',
                                           'Darnell Valentine', 'Kiki Vandeweghe', 'Loy Vaught',
                                           'Sam Vincent', 'Darrell Walker', 'Spud Webb', 'Doug West',
                                           'Randy White', 'Morlon Wiley', 'Dominique Wilkins',
                                           'Gerald Wilkins', 'Herb Williams', 'Hot Rod Williams',
                                           'Jayson Williams', 'John Williams', 'Kenny Williams',
                                           'Micheal Williams', 'Reggie Williams', 'Scott Williams',
                                           'Kevin Willis', 'Kennard Winchester', 'David Wingate',
                                           'Randy Wittman', 'Joe Wolf', 'Mike Woodson', 'Orlando Woolridge',
                                           'Haywoode Workman', 'James Worthy', 'A.J. Wynder', 'Danny Young',
                                           'Rafael Addison', 'Victor Alexander', 'Kenny Anderson',
                                           'Greg Anthony', 'Vincent Askew', 'Stacey Augmon', 'John Bagley',
                                           'Kenny Battle', 'Benoit Benjamin', 'David Benoit',
                                           'Anthony Bonner', 'Anthony Bowie', 'Terrell Brandon',
                                           'Kevin Brooks', 'Mike Brown', 'Randy Brown', 'Jud Buechler',
                                           'Elden Campbell', 'Duane Causwell', 'Pete Chilcutt',
                                           'Chris Corchiani', 'Corey Crowder', 'Dale Davis', 'Rick Fox',
                                           'Chris Gatling', 'Paul Graham', 'Greg Grant', 'Sean Green',
                                           'Carl Herrera', 'Sean Higgins', 'Tyrone Hill', 'Brian Howard',
                                           'Mike Iuzzolino', 'Avery Johnson', 'Larry Johnson', 'Rich King',
                                           'Greg Kite', 'Bart Kofoed', 'Larry Krystkowiak', 'Doug Lee',
                                           'Kevin Lynch', 'Mark Macon', 'Bob McCann', 'Tracy Moore',
                                           'Eric Murdock', 'Billy Owens', 'Robert Pack', 'Bobby Phills',
                                           'Mark Randall', 'Stanley Roberts', 'Cliff Robinson',
                                           'Brad Sellers', 'Charles Shackleford', 'Doug Smith',
                                           'LaBradford Smith', 'Larry Smith', 'Steve Smith', 'Rik Smits',
                                           'Larry Stewart', 'Mitchell Wiggins', 'Buck Williams',
                                           'Alaa Abdelnaby', 'Isaac Austin', 'Anthony Avent', 'Jon Barry',
                                           'Tony Bennett', 'Walter Bond', 'Doug Christie', 'Marty Conlon',
                                           'Duane Cooper', 'John Crotty', 'Lloyd Daniels', 'Hubert Davis',
                                           'Todd Day', 'Vinny Del Negro', 'Bison Dele', 'Richard Dumas',
                                           'LaPhonso Ellis', 'Matt Geiger', 'Litterial Green',
                                           'Tom Gugliotta', 'Robert Horry', 'Byron Houston', 'Jim Jackson',
                                           'Keith Jennings', 'Dave Johnson', 'Frank Johnson', 'Adam Keefe',
                                           'Christian Laettner', 'Tim Legler', 'Don MacLean', 'Rick Mahorn',
                                           'Marlon Maxey', 'Lee Mayberry', 'Oliver Miller', 'Harold Miner',
                                           'Alonzo Mourning', 'Tracy Murray', "Shaquille O'Neal",
                                           'Doug Overton', 'Anthony Peeler', 'Brent Price',
                                           'Clifford Robinson', 'Sean Rooks', 'Donald Royal', 'Malik Sealy',
                                           'Chris Smith', 'Reggie Smith', 'Andre Spencer', 'Latrell Sprewell',
                                           'Bryant Stith', 'Derek Strong', 'Clarence Weatherspoon',
                                           'Robert Werdann', 'Eddie Lee Wilkins', 'Walt Williams',
                                           'David Wood', 'Randy Woods', 'Vin Baker', 'Shawn Bradley',
                                           'Scott Burrell', 'Mitchell Butler', 'Sam Cassell',
                                           'Calbert Cheaney', 'Joe Courtney', 'Antonio Davis', 'Terry Dehere',
                                           'Acie Earl', 'Doug Edwards', 'Harold Ellis', 'Jo Jo English',
                                           'Greg Graham', 'Anfernee Hardaway', 'Lucious Harris',
                                           'Tony Harris', 'Allan Houston', 'Lindsey Hunter', 'Bobby Hurley',
                                           'Jaren Jackson', 'Chris Jent', 'Popeye Jones', 'Eric Leckner',
                                           'Luc Longley', 'George Lynch', 'Gerald Madkins', 'Jamal Mashburn',
                                           'Anthony Mason', 'Darnell Mee', 'Chris Mills', 'Dikembe Mutombo',
                                           'Pete Myers', 'Bo Outlaw', 'Elliot Perry', 'Dino Radja',
                                           'Isaiah Rider', 'Eric Riley', 'James Robinson', 'Larry Robinson',
                                           'Rodney Rogers', 'Bryon Russell', 'Elmore Spencer',
                                           'Nick Van Exel', 'Kenny Walker', 'Rex Walters', 'Chris Webber',
                                           'Bill Wennington', 'David Wesley', 'Chris Whitney',
                                           'Lorenzo Williams', 'Trevor Wilson', 'Luther Wright',
                                           'Darrell Armstrong', 'Elmer Bennett', 'Corie Blount', 'Tim Breaux',
                                           'Chris Childs', 'Steve Colter', 'Chris Dudley', 'Tony Dumas',
                                           'Corey Gaines', 'Brian Grant', 'Jack Haley', 'Darrin Hancock',
                                           'Antonio Harvey', 'Grant Hill', 'Donald Hodge', 'Juwan Howard',
                                           'Ervin Johnson', 'Eddie Jones', 'Randolph Keys', 'Jason Kidd',
                                           'Ryan Lorthridge', 'Donyell Marshall', 'Darrick Martin',
                                           'Tony Massenburg', 'Aaron McKie', 'Anthony Miller', 'Greg Minor',
                                           'Eric Mobley', 'Eric Montross', 'Dwayne Morton', 'Lamond Murray',
                                           'Wesley Person', 'Eric Piatkowski', 'Eldridge Recasner',
                                           'Khalid Reeves', 'Glenn Robinson', 'Carlos Rogers', 'Jalen Rose',
                                           'Clifford Rozier', 'Trevor Ruffin', 'Greg Sutton', 'Roy Tarpley',
                                           'Brooks Thompson', 'Anthony Tucker', 'Henry Turner', 'B.J. Tyler',
                                           'Charlie Ward', 'Jamie Watson', 'Ennis Whatley', 'Monty Williams',
                                           'Dontonio Wingfield', 'Sharone Wright', 'Cory Alexander',
                                           'Jerome Allen', 'Ashraf Amaya', 'Brent Barry', 'Mario Bennett',
                                           'Travis Best', 'Donnie Boyce', 'Jason Caffey', 'Chris Carr',
                                           'Randolph Childress', 'Michael Curry', 'Mark Davis',
                                           'Andrew DeClercq', 'Tyus Edney', 'Howard Eisley', 'Michael Finley',
                                           'Sherell Ford', 'Kevin Garnett', 'Anthony Goldwire',
                                           'Alan Henderson', 'Fred Hoiberg', 'Jimmy King', 'Antonio Lang',
                                           'Voshon Lenard', 'Sam Mack', 'Rich Manning', 'Donny Marshall',
                                           'Antonio McDyess', 'Loren Meyer', 'Lawrence Moten', "Ed O'Bannon",
                                           'Cherokee Parks', 'Theo Ratliff', 'Bryant Reeves', 'Shawn Respert',
                                           'Lou Roe', 'Arvydas Sabonis', 'Steve Scheffler', 'Dickey Simpkins',
                                           'Joe Smith', 'Eric Snow', 'Jerry Stackhouse', 'Damon Stoudamire',
                                           'Bob Sura', 'Kurt Thomas', 'Gary Trent', 'David Vaughn',
                                           'Rasheed Wallace', 'Eric Williams', 'Corliss Williamson',
                                           'Shareef Abdur-Rahim', 'Ray Allen', 'Shandon Anderson',
                                           'Marcus Brown', 'Kobe Bryant', 'Adrian Caldwell', 'Marcus Camby',
                                           'Erick Dampier', 'Emanual Davis', 'Tony Delk', 'Jamie Feick',
                                           'Derek Fisher', 'Reggie Geary', 'Othella Harrington', 'Shane Heal',
                                           'Allen Iverson', 'Kerry Kittles', 'Priest Lauderdale',
                                           'Randy Livingston', 'John Long', 'Matt Maloney', 'Stephon Marbury',
                                           'Walter McCarty', 'Amal McCaskill', 'Jim McIlvaine', 'Steve Nash',
                                           'Ruben Nembhard', "Jermaine O'Neal", 'Jimmy Oliver',
                                           'Greg Ostertag', 'Vitaly Potapenko', 'Don Reid', 'Chris Robinson',
                                           'Roy Rogers', 'Malik Rose', 'Reggie Slater', 'Erick Strickland',
                                           'Mark Strickland', 'Carl Thomas', 'Antoine Walker',
                                           'Samaki Walker', 'John Wallace', 'Donald Whiteside',
                                           'Aaron Williams', 'Lorenzen Wright', 'Tariq Abdul-Wahad',
                                           'Derek Anderson', 'Chris Anstey', 'Drew Barry', 'Tony Battie',
                                           'Corey Beck', 'Chauncey Billups', 'Keith Booth', 'Bruce Bowen',
                                           'Rick Brunson', 'Kelvin Cato', 'Chris Crawford', 'Austin Croshere',
                                           'Bill Curley', 'Antonio Daniels', 'Tim Duncan', 'Brian Evans',
                                           'Danny Fortson', 'Adonal Foyle', 'Todd Fuller',
                                           'Lawrence Funderburke', 'Ed Gray', 'Mark Hendrickson',
                                           'Jerald Honeycutt', 'Zydrunas Ilgauskas', 'Bobby Jackson',
                                           'Anthony Johnson', 'Brevin Knight', 'Travis Knight', 'Rusty LaRue',
                                           'Tracy McGrady', 'Ron Mercer', "Charles O'Bannon",
                                           'Anthony Parker', 'Mark Pope', 'Rodrick Rhodes', 'Johnny Taylor',
                                           'Maurice Taylor', 'Tim Thomas', 'Keith Van Horn', 'Jacque Vaughn',
                                           'Eric Washington', 'Bubba Wells', 'DeJuan Wheat',
                                           'Brandon Williams', 'Jerome Williams', 'Toby Bailey',
                                           'Corey Benjamin', 'Mike Bibby', 'Earl Boykins', 'Gerald Brown',
                                           'Vince Carter', 'Keon Clark', 'Keith Closs', 'James Cotton',
                                           'Ricky Davis', 'Michael Dickerson', 'Bryce Drew', 'Pat Garrity',
                                           'Matt Harpring', 'Al Harrington', 'Michael Hawkins',
                                           'Cedric Henderson', 'Troy Hudson', 'Larry Hughes',
                                           'Randell Jackson', 'Sam Jacobson', 'Antawn Jamison',
                                           'Charles Jones', 'Damon Jones', 'Raef LaFrentz', 'Rashard Lewis',
                                           'Tyronn Lue', 'Jeff McInnis', 'Roshown McLeod', 'Brad Miller',
                                           'Cuttino Mobley', 'Tyrone Nesby', 'Dirk Nowitzki',
                                           'Andrae Patterson', 'Ruben Patterson', 'Paul Pierce',
                                           'Vladimir Stepania', 'John Thomas', 'Robert Traylor',
                                           'Bonzi Wells', 'Alvin Williams', 'Jason Williams', 'Rafer Alston',
                                           'John Amaechi', 'Chucky Atkins', 'William Avery',
                                           'Jonathan Bender', 'Lazaro Borrell', 'Cal Bowdler', 'Ryan Bowen',
                                           'Elton Brand', 'Greg Buckner', 'Rodney Buford', 'Anthony Carter',
                                           'Vonteego Cummings', 'Baron Davis', 'Derrick Dial',
                                           'Michael Doleac', 'Jeff Foster', 'Steve Francis', 'Devean George',
                                           'Dion Glover', 'Adrian Griffin', 'Darvin Ham', 'Richard Hamilton',
                                           'Chris Herren', 'Jermaine Jackson', 'Jumaine Jones',
                                           'Trajan Langdon', 'Quincy Lewis', 'Corey Maggette', 'Shawn Marion',
                                           'Andre Miller', 'Moochie Norris', 'Lamar Odom', 'Scott Padgett',
                                           'Milt Palacio', 'James Posey', 'Laron Profit', 'Eddie Robinson',
                                           'Wally Szczerbiak', 'Jason Terry', 'Kenny Thomas',
                                           'Shammond Williams', 'Metta World Peace', 'Courtney Alexander',
                                           'Brian Cardinal', 'Mateen Cleaves', 'Jason Collier',
                                           'Jamal Crawford', 'Keyon Dooling', 'Khalid El-Amin',
                                           'Marcus Fizer', 'Steve Goodrich', 'Eddie House', 'Marc Jackson',
                                           'Stephen Jackson', 'Tim James', 'DerMarr Johnson', 'Dan Langhi',
                                           'Mark Madsen', 'Jamaal Magloire', 'Kenyon Martin', 'Desmond Mason',
                                           'Stanislav Medvedenko', 'Chris Mihm', 'Darius Miles',
                                           'Mike Miller', 'Nazr Mohammed', 'Mikki Moore', 'Lee Nailon',
                                           'Kevin Ollie', 'Mike Penberthy', 'Morris Peterson', 'Scot Pollard',
                                           'Chris Porter', 'Lavor Postell', 'Michael Redd',
                                           'Quentin Richardson', 'DeShawn Stevenson', 'Stromile Swift',
                                           'Ben Wallace', 'Jahidi White', 'Wang Zhizhi', 'Malik Allen',
                                           'Chris Andersen', 'Gilbert Arenas', 'Brandon Armstrong',
                                           'Carlos Arroyo', 'Erick Barkley', 'Mengke Bateer', 'Shane Battier',
                                           'Raja Bell', 'Michael Bradley', 'Kedrick Brown', 'Kwame Brown',
                                           'Speedy Claxton', 'Jarron Collins', 'Jason Collins',
                                           'Predrag Drobnjak', 'Joseph Forte', 'Pau Gasol', 'Eddie Griffin',
                                           'Trenton Hassell', 'Kirk Haston', 'Mike James',
                                           'Richard Jefferson', 'Joe Johnson', 'Andrei Kirilenko',
                                           'Terence Morris', 'Troy Murphy', 'Ira Newble', 'Dean Oliver',
                                           'Tony Parker', 'Joel Przybilla', 'Jason Richardson',
                                           'Kenny Satterfield', 'Brian Scalabrine', 'Bobby Simmons',
                                           'Jamaal Tinsley', 'Gerald Wallace', 'Earl Watson', 'Rodney White',
                                           'Loren Woods', 'Lonny Baxter', 'Calvin Booth', 'Carlos Boozer',
                                           'Jamison Brewer', 'Damone Brown', 'Devin Brown', 'Caron Butler',
                                           'Rasual Butler', 'Dan Dickau', 'Juan Dixon', 'Mike Dunleavy',
                                           'Tremaine Fowlkes', 'Dan Gadzuric', 'Drew Gooden',
                                           'Marcus Haislip', 'Donnell Harvey', 'Nenê', 'Casey Jacobsen',
                                           'Chris Jefferies', 'Jared Jeffries', 'Fred Jones', 'Sean Lampley',
                                           'Sean Marks', 'Roger Mason', 'Yao Ming', 'Ronald Murray',
                                           'Mehmet Okur', 'Jannero Pargo', 'Tayshaun Prince', 'Zach Randolph',
                                           'Efthimios Rentzias', 'Kareem Rush', 'John Salmons',
                                           'Jeryl Sasser', 'Tamar Slay', 'Jeff Trepagnier',
                                           'Nikoloz Tskitishvili', 'Dajuan Wagner', 'Mike Wilks',
                                           'Frank Williams', 'Jay Williams', 'Qyntel Woods',
                                           'Carmelo Anthony', 'Marcus Banks', 'Leandro Barbosa',
                                           'Matt Barnes', 'Troy Bell', 'Steve Blake', 'Keith Bogans',
                                           'Curtis Borchardt', 'Chris Bosh', 'Tyson Chandler', 'Brian Cook',
                                           'Eddy Curry', 'Samuel Dalembert', 'Marquis Daniels', 'Josh Davis',
                                           'Boris Diaw', 'Ronald Dupree', 'Francisco Elson', 'Melvin Ely',
                                           'Reggie Evans', 'Richie Frahm', 'Reece Gaines', 'Eddie Gill',
                                           'Willie Green', 'Jason Hart', 'Udonis Haslem', 'Jarvis Hayes',
                                           'Brendan Haywood', 'Kirk Hinrich', 'Josh Howard', 'Brandon Hunter',
                                           'LeBron James', 'Britton Johnsen', 'Linton Johnson',
                                           'Dahntay Jones', 'James Jones', 'Chris Kaman', 'Jason Kapono',
                                           'Kyle Korver', 'Maciej Lampe', 'Keith McLeod', 'Luke Ridnour',
                                           'Ansu Sesay', 'Jabari Smith', 'Theron Smith', 'Mike Sweetney',
                                           'Dwyane Wade', 'Luke Walton', 'David West', 'Chris Wilcox',
                                           'Tony Allen', 'Trevor Ariza', 'Mark Blount', 'Matt Bonner',
                                           'Antonio Burks', 'Matt Carroll', 'Lionel Chalmers',
                                           'Josh Childress', 'Nick Collison', 'Carlos Delfino', 'Luol Deng',
                                           'Kaniel Dickens', 'DeSagana Diop', 'Chris Duhon', 'Ndudi Ebi',
                                           'Maurice Evans', 'Ben Gordon', 'Ben Handlogten', 'Devin Harris',
                                           'Dwight Howard', 'Kris Humphries', 'Steven Hunter',
                                           'Andre Iguodala', 'Royal Ivey', 'Luke Jackson', 'Jerome James',
                                           'Al Jefferson', 'Mario Kasun', 'Viktor Khryapa',
                                           'Shaun Livingston', 'Kevin Martin', 'Jameer Nelson',
                                           'Emeka Okafor', 'Travis Outlaw', 'Zaza Pachulia', 'Smush Parker',
                                           'Justin Reed', 'Bernard Robinson', 'Quinton Ross',
                                           'Michael Ruffin', 'Brian Skinner', 'Donta Smith', 'Josh Smith',
                                           'Kirk Snyder', 'Darius Songaila', 'Awvee Storey',
                                           'Sebastian Telfair', 'Beno Udrih', 'Jackson Vroman',
                                           'Delonte West', 'Damien Wilkins', 'Dorell Wright', 'Alan Anderson',
                                           'Andre Barrett', 'Esteban Batista', 'Charlie Bell',
                                           'Andray Blatche', 'Andrew Bogut', 'Pat Burke', 'Kevin Burleson',
                                           'Travis Diener', 'Monta Ellis', 'Daniel Ewing', 'Raymond Felton',
                                           'Channing Frye', 'Ryan Gomes', 'Joey Graham', 'Stephen Graham',
                                           'Danny Granger', 'Gerald Green', 'Orien Greene', 'Chuck Hayes',
                                           'Luther Head', 'Jarrett Jack', 'Amir Johnson', 'Linas Kleiza',
                                           'Yaroslav Korolev', 'Arvydas Macijauskas', 'Rawle Marshall',
                                           'Sean May', 'Rashad McCants', "Boniface N'Dong", 'Fabricio Oberto',
                                           'Andre Owens', 'Chris Paul', 'Kendrick Perkins', 'Ronnie Price',
                                           'Anthony Roberson', 'Lawrence Roberts', 'Nate Robinson',
                                           'James Singleton', 'Salim Stoudamire', 'Donell Taylor',
                                           'Dijon Thompson', 'Ime Udoka', 'Charlie Villanueva',
                                           'Jake Voskuhl', 'Von Wafer', 'Martell Webster', 'Deron Williams',
                                           'Marvin Williams', 'Antoine Wright', 'Bracey Wright',
                                           'Maurice Ager', 'LaMarcus Aldridge', 'Kelenna Azubuike',
                                           'Renaldo Balkman', 'Andrea Bargnani', 'Earl Barron',
                                           'Brandon Bass', 'Maceo Baston', 'Ronnie Brewer', 'Andre Brown',
                                           'Shannon Brown', 'Andrew Bynum', 'Rodney Carney', 'Mardy Collins',
                                           'Yakhouba Diawara', 'Quincy Douby', 'Jordan Farmar', 'Randy Foye',
                                           'Jorge Garbajosa', 'Rudy Gay', 'Daniel Gibson', 'Walter Herrmann',
                                           'Bobby Jones', 'Solomon Jones', 'Tarence Kinsey', 'Kyle Lowry',
                                           'Paul Millsap', 'Adam Morrison', 'David Noel', 'Steve Novak',
                                           'Leon Powe', 'Chris Quinn', 'Rajon Rondo', 'Brandon Roy',
                                           'Thabo Sefolosha', 'Craig Smith', 'Vassilis Spanoulis',
                                           'Tyrus Thomas', 'Ronny Turiaf', 'Marcus Vinicius', 'Hakim Warrick',
                                           'Justin Williams', 'Marcus Williams', 'Shawne Williams',
                                           'Shelden Williams', 'Arron Afflalo', 'Blake Ahearn',
                                           'Morris Almond', 'Marco Belinelli', 'Corey Brewer', 'Aaron Brooks',
                                           'Wilson Chandler', 'Mike Conley', 'Daequan Cook',
                                           'Javaris Crittenton', 'Jared Dudley', 'Kevin Durant',
                                           'Thomas Gardner', 'Aaron Gray', 'Jeff Green', 'Spencer Hawes',
                                           'Al Horford', 'Yi Jianlian', 'Dwayne Jones', 'Carl Landry',
                                           'Acie Law', 'David Lee', 'Jason Maxiell', 'Dominic McGuire',
                                           'Jamario Moon', 'Joakim Noah', 'Oleksiy Pecherov', 'Johan Petro',
                                           'Josh Powell', 'Gabe Pruitt', 'Jeremy Richardson', 'Luis Scola',
                                           'Ramon Sessions', 'Jason Smith', 'Rodney Stuckey', 'Al Thornton',
                                           'Alando Tucker', 'Mario West', 'Julian Wright', 'Nick Young',
                                           'Thaddeus Young', 'Joe Alexander', 'Ryan Anderson',
                                           'Hilton Armstrong', 'Darrell Arthur', 'Nicolas Batum',
                                           'Jerryd Bayless', 'Michael Beasley', 'Bobby Brown', 'Will Bynum',
                                           'Mario Chalmers', 'Glen Davis', 'Paul Davis', 'Ike Diogu',
                                           'Chris Douglas-Roberts', 'Danilo Gallinari', 'Marc Gasol',
                                           'Eric Gordon', 'Marcin Gortat', 'Malik Hairston', 'George Hill',
                                           'Darnell Jackson', 'Courtney Lee', 'Brook Lopez', 'Robin Lopez',
                                           'Kevin Love', 'Cartier Martin', 'Luc Mbah a Moute',
                                           'Josh McRoberts', 'Pops Mensah-Bonsu', 'Anthony Morrow',
                                           "Patrick O'Bryant", 'Anthony Randolph', 'Shavlik Randolph',
                                           'Derrick Rose', 'Brandon Rush', 'Marreese Speights',
                                           'Jason Thompson', 'Anthony Tolliver', 'Henry Walker',
                                           'Kyle Weaver', 'Sonny Weems', 'Russell Westbrook',
                                           'Brandan Wright', 'David Andersen', 'Rodrigue Beaubois',
                                           'DeJuan Blair', 'Derrick Brown', 'Chase Budinger',
                                           'DeMarre Carroll', 'Omri Casspi', 'Earl Clark', 'Darren Collison',
                                           'Dante Cunningham', 'Stephen Curry', 'Austin Daye',
                                           'DeMar DeRozan', 'Toney Douglas', 'Wayne Ellington',
                                           'Tyreke Evans', 'Kyrylo Fesenko', 'Jonny Flynn', 'Sundiata Gaines',
                                           'Alonzo Gee', 'Danny Green', 'Hamed Haddadi', 'Tyler Hansbrough',
                                           'James Harden', 'Gerald Henderson', 'Roy Hibbert', 'Jordan Hill',
                                           'Jrue Holiday', 'Ryan Hollins', 'Lester Hudson', 'Serge Ibaka',
                                           'Brandon Jennings', 'Jonas Jerebko', 'James Johnson',
                                           'DeAndre Jordan', 'Ty Lawson', 'Wesley Matthews', 'Eric Maynor',
                                           'JaVale McGee', 'Jodie Meeks', 'DaJuan Summers', 'Jermaine Taylor',
                                           'Jeff Teague', 'Garrett Temple', 'Marcus Thornton',
                                           'Jawad Williams', 'Terrence Williams', 'Sam Young',
                                           'Al-Farouq Aminu', 'James Anderson', 'Luke Babbitt',
                                           'Eric Bledsoe', 'Trevor Booker', 'Avery Bradley',
                                           'DeMarcus Cousins', 'Jordan Crawford', 'Devin Ebanks',
                                           'Jeremy Evans', 'Christian Eyenga', 'Landry Fields', 'Gary Forbes',
                                           'Paul George', 'Taj Gibson', 'Blake Griffin', 'Luke Harangody',
                                           'Manny Harris', 'Gordon Hayward', 'Lazar Hayward', 'Xavier Henry',
                                           'Damion James', 'Armon Johnson', 'Wesley Johnson',
                                           'Dominique Jones', 'Kosta Koufos', 'Jeremy Lin', 'Ian Mahinmi',
                                           'Greg Monroe', 'Gary Neal', 'Larry Owens', 'Patrick Patterson',
                                           'Quincy Pondexter', 'Andy Rautins', 'Samardo Samuels',
                                           'Tiago Splitter', 'Lance Stephenson', 'Evan Turner', 'Ben Uzoh',
                                           'John Wall', 'Jon Brockman', 'MarShon Brooks', 'Alec Burks',
                                           'Jimmy Butler', 'Norris Cole', 'Ed Davis', 'Justin Dentmon',
                                           'Jimmer Fredette', 'Andrew Goudelock', 'Jordan Hamilton',
                                           'Justin Harper', 'Josh Harrellson', 'Terrel Harris',
                                           'Tobias Harris', 'Cory Higgins', 'Tyler Honeycutt', 'Kyrie Irving',
                                           'Reggie Jackson', 'Charles Jenkins', 'Ivan Johnson', 'Cory Joseph',
                                           'Enes Kanter', 'Brandon Knight', 'Malcolm Lee', 'Kawhi Leonard',
                                           'Jon Leuer', 'DeAndre Liggins', 'Shelvin Mack', "E'Twaun Moore",
                                           'Darius Morris', 'Marcus Morris', 'Markieff Morris',
                                           'Daniel Orton', 'Jeremy Pargo', 'Chandler Parsons', 'Ricky Rubio',
                                           'Larry Sanders', 'Josh Selby', 'Iman Shumpert', 'Chris Singleton',
                                           'Donald Sloan', 'Nolan Smith', 'Julyan Stone', 'Isaiah Thomas',
                                           'Lance Thomas', 'Trey Thompkins', 'Klay Thompson',
                                           'Tristan Thompson', 'Jeremy Tyler', 'Ekpe Udoh', 'Kemba Walker',
                                           'Derrick Williams', 'Elliot Williams', 'Quincy Acy', 'Jeff Adrien',
                                           'Lavoy Allen', 'Jeff Ayres', 'Harrison Barnes', 'Will Barton',
                                           'Aron Baynes', 'Kent Bazemore', 'Bradley Beal', 'Patrick Beverley',
                                           'Victor Claver', 'Chris Copeland', 'Jae Crowder',
                                           'Jared Cunningham', 'Anthony Davis', 'Nando De Colo',
                                           'Andre Drummond', 'Derrick Favors', 'Evan Fournier',
                                           'Diante Garrett', 'Draymond Green', 'John Henson', 'Bernard James',
                                           'John Jenkins', 'Orlando Johnson', 'Perry Jones', 'Terrence Jones',
                                           'Michael Kidd-Gilchrist', 'Doron Lamb', 'Jeremy Lamb',
                                           'Meyers Leonard', 'Damian Lillard', 'Kendall Marshall',
                                           'Khris Middleton', 'Darius Miller', 'Quincy Miller',
                                           'Timofey Mozgov', "Kyle O'Quinn", 'Pablo Prigioni',
                                           'Austin Rivers', 'Brian Roberts', 'Thomas Robinson',
                                           'Terrence Ross', 'Mike Scott', 'Tornike Shengelia', 'Alexey Shved',
                                           'Kyle Singler', 'Greg Smith', 'Jared Sullinger', 'Marquis Teague',
                                           'Dion Waiters', 'Maalik Wayns', 'Tony Wroten', 'Tyler Zeller',
                                           'Giannis Antetokounmpo', 'Anthony Bennett', 'Vander Blue',
                                           'Lorenzo Brown', 'Reggie Bullock', 'Trey Burke', 'Dwight Buycks',
                                           'Nick Calathes', 'Kentavious Caldwell-Pope', 'Isaiah Canaan',
                                           'Michael Carter-Williams', 'Ian Clark', 'Allen Crabbe',
                                           'Gigi Datome', 'Brandon Davies', 'Matthew Dellavedova',
                                           'Gorgui Dieng', 'Kenneth Faried', 'Carrick Felix',
                                           'Jamaal Franklin', 'Joel Freeland', 'Archie Goodwin',
                                           'Justin Hamilton', 'Solomon Hill', 'Robbie Hummel',
                                           'Sergey Karasev', 'Ryan Kelly', 'Shane Larkin', 'Ricky Ledo',
                                           'Ray McCallum', 'CJ McCollum', 'Ben McLemore', 'Gal Mekel',
                                           'Tony Mitchell', 'Shabazz Muhammad', 'Mike Muscala',
                                           'Andrew Nicholson', 'Victor Oladipo', 'Kelly Olynyk',
                                           'Mason Plumlee', 'Otto Porter', 'Phil Pressey', 'Andre Roberson',
                                           'Henry Sims', 'Tony Snell', 'Greg Stiemsma', 'Malcolm Thomas',
                                           'Hollis Thompson', 'Jeff Withey', 'Nate Wolters', 'Cody Zeller',
                                           'Jordan Adams', 'Steven Adams', 'Furkan Aldemir', 'Kyle Anderson',
                                           'Tarik Black', 'Markel Brown', 'Jordan Clarkson', 'Bryce Cotton',
                                           'Robert Covington', 'Troy Daniels', 'Dewayne Dedmon',
                                           'Spencer Dinwiddie', 'Joey Dorsey', 'Cleanthony Early',
                                           'James Ennis', 'Tyler Ennis', 'Dante Exum', 'Tim Frazier',
                                           'Langston Galloway', 'Rudy Gobert', 'Aaron Gordon', 'Jerami Grant',
                                           'Erick Green', 'JaMychal Green', 'Gary Harris', 'Joe Harris',
                                           'Justin Holiday', 'Rodney Hood', 'Joe Ingles', 'Cory Jefferson',
                                           'Nick Johnson', 'Tyler Johnson', 'Sean Kilpatrick',
                                           'Joffrey Lauvergne', 'Zach LaVine', 'Alex Len', 'Devyn Marble',
                                           'Doug McDermott', 'Mitch McGary', 'Elijah Millsap',
                                           'Shabazz Napier', 'Kostas Papanikolaou', 'Jabari Parker',
                                           'Adreian Payne', 'Elfrid Payton', 'Dwight Powell',
                                           'Glenn Robinson III', 'Robert Sacre', 'JaKarr Sampson',
                                           'Marcus Smart', 'Russ Smith', 'Nik Stauskas', 'Noah Vonleh',
                                           'Shayne Whittington', 'Andrew Wiggins', 'James Young',
                                           'Justin Anderson', 'Bismack Biyombo', 'Nemanja Bjelica',
                                           'Devin Booker', 'Anthony Brown', 'Clint Capela',
                                           'Willie Cauley-Stein', 'Pat Connaughton', 'Seth Curry',
                                           'Jarell Eddie', 'Jerian Grant', 'Montrezl Harrell',
                                           'Aaron Harrison', 'Mario Hezonja', 'Darrun Hilliard',
                                           'Rondae Hollis-Jefferson', 'Richaun Holmes', 'Marcelo Huertas',
                                           'Josh Huestis', 'Stanley Johnson', 'Tyus Jones', 'Frank Kaminsky',
                                           'Trey Lyles', 'Jarell Martin', 'James Michael McAdoo',
                                           'Chris McCullough', 'Jordan McRae', 'Salah Mejri',
                                           'Emmanuel Mudiay', 'Raul Neto', 'Nerlens Noel', 'Lucas Nogueira',
                                           "Johnny O'Bryant", 'Jahlil Okafor', 'Lamar Patterson',
                                           'Cameron Payne', 'Bobby Portis', 'Norman Powell', 'Julius Randle',
                                           'Josh Richardson', 'Terry Rozier', "D'Angelo Russell",
                                           'Jonathon Simmons', 'Axel Toupane', 'Karl-Anthony Towns',
                                           'Myles Turner', 'Rashad Vaughn', 'Briante Weber',
                                           'Justise Winslow', 'Christian Wood', 'Delon Wright', 'Ron Baker',
                                           'Wade Baldwin', 'Malik Beasley', 'Dragan Bender', 'Joel Bolomboy',
                                           'Malcolm Brogdon', 'Jaylen Brown', 'Marquese Chriss', 'Quinn Cook',
                                           'Sam Dekker', 'Malcolm Delaney', 'Kris Dunn', 'Henry Ellenson',
                                           'Joel Embiid', 'Kay Felder', 'Yogi Ferrell', 'Dorian Finney-Smith',
                                           'Bryn Forbes', 'Michael Gbinije', 'Marcus Georges-Hunt',
                                           'Jonathan Gibson', 'Treveon Graham', 'Andrew Harrison',
                                           'Buddy Hield', 'Brandon Ingram', 'Demetrius Jackson',
                                           'Mindaugas Kuzminskas', 'Jake Layman', 'Caris LeVert',
                                           'Kevon Looney', 'Sheldon Mac', 'Thon Maker', 'Patrick McCaw',
                                           'Rodney McGruder', 'Jordan Mickey', 'Dejounte Murray',
                                           'Jamal Murray', 'Georges Niang', 'David Nwaba',
                                           'Georgios Papagiannis', 'Gary Payton II', 'Alex Poythress',
                                           'Taurean Prince', 'Tim Quarterman', 'Chasson Randle',
                                           'Willie Reed', 'Malachi Richardson', 'Domantas Sabonis',
                                           'Wayne Selden', 'Pascal Siakam', 'Isaiah Taylor', 'Tyler Ulis',
                                           'Denzel Valentine', 'Fred VanVleet', 'Okaro White',
                                           'Isaiah Whitehead', 'Alan Williams', 'Troy Williams',
                                           'Paul Zipser', 'Ivica Zubac', 'Bam Adebayo', 'Jarrett Allen',
                                           'Kadeem Allen', 'OG Anunoby', 'Ryan Arcidiacono', 'Dwayne Bacon',
                                           'Lonzo Ball', 'Jordan Bell', 'Jabari Bird', 'Antonio Blakeney',
                                           'Tony Bradley', 'Dillon Brooks', 'Sterling Brown', 'Thomas Bryant',
                                           'Bruno Caboclo', 'Alex Caruso', 'Tyler Cavanaugh', 'John Collins',
                                           'Zach Collins', 'Charles Cooke', 'Torrey Craig', 'Tyler Dorsey',
                                           'Damyean Dotson', 'Jawun Evans', 'Terrance Ferguson',
                                           "De'Aaron Fox", 'Markelle Fultz', 'Shaquille Harrison',
                                           'Josh Hart', 'Isaiah Hicks', 'John Holland', 'Danuel House',
                                           'Andre Ingram', 'Jonathan Isaac', 'Wesley Iwundu', 'Josh Jackson',
                                           'Justin Jackson', 'Jalen Jones', 'Luke Kennard', 'Maxi Kleber',
                                           'Furkan Korkmaz', 'Luke Kornet', 'Kyle Kuzma', 'Damion Lee',
                                           'Lauri Markkanen', 'Alfonzo McKinnie', 'Malcolm Miller',
                                           'Donovan Mitchell', 'Malik Monk', 'Monte Morris',
                                           'Johnathan Motley', 'Abdel Nader', 'Frank Ntilikina',
                                           "Royce O'Neale", 'Semi Ojeleye', 'Cedi Osman', 'Jakob Poeltl',
                                           'Zhou Qi', 'Davon Reed', 'Ben Simmons', 'Kobi Simmons',
                                           'Caleb Swanigan', 'Jayson Tatum', 'Daniel Theis',
                                           'Sindarius Thornwell', 'Tyrone Wallace', 'Derrick White',
                                           'Hassan Whiteside', 'Guerschon Yabusele', 'Jaylen Adams',
                                           'Grayson Allen', 'Deandre Ayton', 'Mo Bamba', 'Keita Bates-Diop',
                                           'Khem Birch', 'Jonah Bolden', 'Isaac Bonga', 'Chris Boucher',
                                           'Mikal Bridges', 'Miles Bridges', 'Ryan Broekhoff', 'Bruce Brown',
                                           'Jalen Brunson', 'Deonte Burton', 'Jevon Carter', 'Gary Clark',
                                           'Cheick Diallo', 'Hamidou Diallo', 'Donte DiVincenzo', 'PJ Dozier',
                                           'Jacob Evans', 'Melvin Frazier', 'Harry Giles',
                                           'Shai Gilgeous-Alexander', 'Brandon Goodwin', 'Aaron Holiday',
                                           'Kevin Huerter', 'Chandler Hutchison', 'Frank Jackson',
                                           'Alize Johnson', 'Kevin Knox', 'Rodions Kurucs',
                                           "De'Anthony Melton", 'Chimezie Metu', 'Shake Milton',
                                           'Sviatoslav Mykhailiuk', 'Elie Okobo', 'Josh Okogie',
                                           'Justin Patton', 'Theo Pinson', 'Duncan Robinson',
                                           'Jerome Robinson', 'Collin Sexton', 'Landry Shamet',
                                           'Anfernee Simons', 'Zhaire Smith', 'Omari Spellman',
                                           'Edmond Sumner', 'Khyri Thomas', 'Jarred Vanderbilt',
                                           'Moritz Wagner', 'Lonnie Walker', 'Brad Wanamaker',
                                           'Yuta Watanabe', 'Johnathan Williams', 'Kenrich Williams',
                                           'Trae Young', 'Nickeil Alexander-Walker', 'Thanasis Antetokounmpo',
                                           'RJ Barrett', 'Darius Bazley', 'Goga Bitadze', 'Bol Bol',
                                           'Jarrell Brantley', 'Ignas Brazdeikis', 'Oshae Brissett',
                                           'Chris Chiozza', 'Brandon Clarke', 'Nicolas Claxton',
                                           'Chris Clemons', 'Amir Coffey', 'Jarrett Culver', 'Terence Davis',
                                           'Luguentz Dort', 'Sekou Doumbouya', 'Carsen Edwards',
                                           'Drew Eubanks', 'Bruno Fernando', 'Wenyen Gabriel',
                                           'Darius Garland', 'Javonte Green', 'Marko Guduric',
                                           'Rui Hachimura', 'Isaiah Hartenstein', 'Jaxson Hayes',
                                           'Tyler Herro', 'Talen Horton-Tucker', "De'Andre Hunter",
                                           'Justin James', 'DaQuan Jeffries', 'Ty Jerome', 'Cameron Johnson',
                                           'Keldon Johnson', 'Damian Jones', 'Mfiondu Kabengele',
                                           'John Konchar', 'Romeo Langford', 'Jalen Lecque', 'Nassir Little',
                                           'Terance Mann', 'Caleb Martin', 'Cody Martin', 'Kelan Martin',
                                           'Garrison Mathews', 'Jalen McDaniels', 'Jordan McLaughlin',
                                           'Adam Mokoka', 'Ja Morant', 'Juwan Morgan', 'Mychal Mulder',
                                           'Jaylen Nowell', 'Kendrick Nunn', 'KZ Okpala', 'Miye Oni',
                                           'Eric Paschall', 'Vincent Poirier', 'Jordan Poole', 'Cam Reddish',
                                           'Naz Reid', 'Admiral Schofield', 'Chris Silva', 'Max Strus',
                                           'Matt Thomas', 'Matisse Thybulle', 'Juan Toscano-Anderson',
                                           'Rayjon Tucker', 'Dean Wade', 'Tremont Waters', 'Paul Watson',
                                           'Quinndary Weatherspoon', 'Coby White', 'Grant Williams',
                                           'Zion Williamson']],
                                 placeholder="Select a player",
                                 clearable=True,
                                 style={'font-weight': 'normal'})], style={"width": "13%", 'font-weight': 'bold'}),
        html.Br(),
        html.Label(["Choose a Valid Season to Predict",
                    dcc.Dropdown(id="year",
                                 clearable=False,
                                 multi=False, style={'font-weight': 'normal'})], style={'font-weight': 'bold'}),
        html.Br(),
        html.Label(
            dbc.Button(id='clear_button', n_clicks=0, children="Clear All Input", color="primary",
                       size="lg", style={'float': 'right', 'margin': 'auto'}),
            style={'font-weight': 'bold'}),
        html.Label(
            dbc.Button(id='pre_button', n_clicks=0, children="Click To Autofill This Player's Stats", color="success",
                       size="lg", style={'float': 'right', 'margin': 'auto'}),
            style={'font-weight': 'bold', 'padding-left': '10px'}),
        # html.Br(),
        # html.Br(),
        # html.Div(id='pre_output',
        #          children='',
        #          style={'font-weight': 'bold', 'font-size': '240%'}
        #          ),
        html.Br(),
        html.Br(),
        html.Label(["Position (Pos)",
                    dcc.Dropdown(id="POS",
                                 clearable=False,
                                 multi=False, style={'font-weight': 'normal'})], style={'font-weight': 'bold'}),
        html.Br(),
        html.Label(["Player's Salary Last Season: ",
                    dcc.Input(id='last_salary', type='text', disabled=False)], style={'font-weight': 'bold'}),
        html.Br(),
        html.Label(["Player's Age: ",
                    dcc.Input(id='Age', type='text', disabled=False)], style={'font-weight': 'bold'}),
        html.Br(),
        html.Label(["Number of Turnovers (TOV): ",
                    dcc.Input(id='TOV', type='text', disabled=False)], style={'font-weight': 'bold'}),
        html.Br(),
        html.Label(["Games Played (G): ",
                    dcc.Input(id='G', type='text', disabled=False)], style={'font-weight': 'bold'}),
        html.Br(),
        html.Label(["Field Goals Made (FG): ",
                    dcc.Input(id='FG', type='text')], style={'font-weight': 'bold'}),
        html.Br(),
        html.Label(["Free Throw Attempts (FTA): ",
                    dcc.Input(id='FTA', type='text')], style={'font-weight': 'bold'}),
        html.Br(),
        html.Label(["Assists (AST): ",
                    dcc.Input(id='AST', type='text')], style={'font-weight': 'bold'}),
        html.Br(),
        html.Label(["Total Rebounds (TRB): ",
                    dcc.Input(id='TRB', type='text')], style={'font-weight': 'bold'}),
        html.Br(),
        html.Label(["Points (PTS): ",
                    dcc.Input(id='PTS', type='text')], style={'font-weight': 'bold'}),
        html.Br(),
        html.Label(["Blocks (BLK): ",
                    dcc.Input(id='BLK', type='text')], style={'font-weight': 'bold'}),
        html.Br(),
        html.Label(["Years of Experience: ",
                    dcc.Input(id='YOE', type='text', disabled=False)], style={'font-weight': 'bold'}),
        html.Br(),
        dbc.Button(id='button', n_clicks=0, children="Submit", color="primary", size="lg"),
        html.Br(),
        html.Br(),
        html.Div(id='salary_output',
                 children='',
                 style={'font-weight': 'bold', 'font-size': '240%'}
                 ),
        # html.Div(id='actual_output',
        #          children='',
        #          style={'font-weight': 'normal', 'font-size': '160%'}
        #          ),
        html.Br(),
        html.Br()
        # style={'text-align': 'center'}
    ], style={'text-align': 'center'}),
    html.Br()
])


# @app.callback(
#     Output('TOV', 'disabled'), Output('Age', 'disabled'), Output('G', 'disabled'),
#     [Input('YOE', 'value')])
# def set_input2(YOE):
#     if YOE > 1:
#         return True, True, True
#     else:
#         return False, False, False


# @app.callback(
#     dash.dependencies.Output('YOE', 'value'), dash.dependencies.Output('last_salary', 'value'),
#     [dash.dependencies.Input('name', 'value')]
# )
# def update_yoe(name):
#     if name is None:
#         return "", ""

@app.callback(
    dash.dependencies.Output('POS', 'options'),
    [dash.dependencies.Input('YOE', 'value'), dash.dependencies.Input('name', 'value')]
)
def update_Pos_dropdown(YOE, name):
    if (YOE == 1) or (name is None):
        options = list(yoe_1.keys())
    else:
        options = list(yoe_2.keys())
    return [{'label': k, 'value': k} for k in options]


@app.callback(
    dash.dependencies.Output('year', 'options'),
    [dash.dependencies.Input('name', 'value')]
)
def update_season_dropdown(name):
    if name is None:
        valid_seasons = ['1991-1992', '1992-1993', '1993-1994', '1994-1995', '1995-1996', '1996-1997', '1997-1998',
                         '1998-1999', '1999-2000', '2000-2001', '2001-2002', '2002-2003', '2003-2004', '2004-2005',
                         '2005-2006', '2006-2007', '2007-2008', '2008-2009', '2009-2010', '2010-2011', '2011-2012',
                         '2012-2013', '2013-2014', '2014-2015', '2015-2016', '2016-2017', '2017-2018', '2018-2019',
                         '2019-2020', '2020-2021']
    else:
        row = df[df['Player'] == str(name)]
        valid_seasons = [str(row['SalStartYr'].values[i]) + "-" + str(row['SalEndYr'].values[i]) for i in
                         range(len(row))]

    return [{'label': k, 'value': k} for k in valid_seasons]


@app.callback(
    Output('YOE', 'disabled'), Output('last_salary', 'disabled'), Output('TOV', 'disabled'), Output('Age', 'disabled'),
    Output('G', 'disabled'),
    [Input('YOE', 'value'), Input('name', 'value')])
def set_input1(YOE, name):
    if (YOE == 1) or (name is None):
        return True, True, False, False, False
    else:
        return False, False, True, True, True


# list1 = [[float(FG), float(FTA), float(AST), float(TRB), float(PTS), float(BLK), float(t_year), float(TOV),
#                      float(Age), float(G), float(t_POS)]]

# list1 = [ [float(row_in_question['Player'].values[0]), float(t_year), float(FG), float(FTA), float(AST),
# float(TRB), float(PTS),float(BLK), float(YOE), float(t_POS), float(last_salary)]]


# @app.callback(
#     Output('last_salary', 'disabled'),
#     [Input('last_salary', 'value')])
# def set_input2(last_salary):
#     if last_salary == 'N/A':
#         return True
#     else:
#         return False


def update(btn1, btn2, name, year):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'pre_button' in changed_id:
        try:
            int_year = int(str(year).split("-")[0])
            row = df[(df['SalStartYr'] == int_year) & (df['Player'] == str(name))]

            if row['years_of_exp'].values[0] == 1:
                salary_before = "N/A"
            else:
                row_in_question = christine_subset[
                    (christine_subset['True'] == str(name)) & (christine_subset['SalStartYr'] == map_years[int_year])]
                salary_before = row_in_question['salary_before'].values[0]

            return name, year, row['Pos'].values[0], salary_before, row['Age'].values[0], row['TOV'].values[0], \
                   row['G'].values[0], row['FG'].values[0], row['FTA'].values[0], \
                   row['AST'].values[0], \
                   row['TRB'].values[0], row['PTS'].values[0], row['BLK'].values[0], \
                   row['years_of_exp'].values[0]
        except (IndexError, ValueError):
            return "", "", "", "", "", "", "", "", "", "", "", "","",""
    elif 'clear_button' in changed_id:
        return None, "", "", "", "", "", "", "", "", "", "", "", "", ""




app.callback(
    dash.dependencies.Output('name', 'value'),
    dash.dependencies.Output('year', 'value'),
    dash.dependencies.Output('POS', 'value'),
    dash.dependencies.Output('last_salary', 'value'),
    dash.dependencies.Output('Age', 'value'),
    dash.dependencies.Output('TOV', 'value'),
    dash.dependencies.Output('G', 'value'),
    dash.dependencies.Output('FG', 'value'),
    dash.dependencies.Output('FTA', 'value'),
    dash.dependencies.Output('AST', 'value'),
    dash.dependencies.Output('TRB', 'value'),
    dash.dependencies.Output('PTS', 'value'),
    dash.dependencies.Output('BLK', 'value'),
    dash.dependencies.Output('YOE', 'value'),
    dash.dependencies.Input('pre_button', 'n_clicks'),
    dash.dependencies.Input('clear_button', 'n_clicks'),
    [dash.dependencies.State('name', 'value'),
     dash.dependencies.State('year', 'value')],
    prevent_initial_call=True
)(update)

#
# def clear(ignore):
#     return "","","", "", "", "", "", "", "", "", "", "", "", ""
#
# app.callback(
#     dash.dependencies.Output('name', 'value'),
#     dash.dependencies.Output('year', 'value'),
#     dash.dependencies.Output('POS', 'value'),
#     dash.dependencies.Output('last_salary', 'value'),
#     dash.dependencies.Output('Age', 'value'),
#     dash.dependencies.Output('TOV', 'value'),
#     dash.dependencies.Output('G', 'value'),
#     dash.dependencies.Output('FG', 'value'),
#     dash.dependencies.Output('FTA', 'value'),
#     dash.dependencies.Output('AST', 'value'),
#     dash.dependencies.Output('TRB', 'value'),
#     dash.dependencies.Output('PTS', 'value'),
#     dash.dependencies.Output('BLK', 'value'),
#     dash.dependencies.Output('YOE', 'value'),
#     dash.dependencies.Input('clear_button', 'n_clicks'),
#     prevent_initial_call=True
# )(clear)





@app.callback(
    Output(component_id='salary_output', component_property='children'),
    Input(component_id='button', component_property='n_clicks'),
    [State(component_id='POS', component_property='value'),
     State(component_id='last_salary', component_property='value'),
     State(component_id='name', component_property='value'),
     State(component_id='Age', component_property='value'),
     State(component_id='TOV', component_property='value'),
     State(component_id='G', component_property='value'),
     State(component_id='year', component_property='value'),
     State(component_id='FG', component_property='value'),
     State(component_id='FTA', component_property='value'),
     State(component_id='AST', component_property='value'),
     State(component_id='TRB', component_property='value'),
     State(component_id='PTS', component_property='value'),
     State(component_id='BLK', component_property='value'),
     State(component_id='YOE', component_property='value')],
    prevent_initial_call=True
)
def get_salary(n, POS, last_salary, name, Age, TOV, G, year, FG, FTA, AST, TRB, PTS, BLK, YOE):
    try:
        t_year = map_years[int(str(year).split("-")[0])]

        if (YOE == 1) or (name is None):
            t_POS = float(yoe_1[POS])

            list1 = [[float(FG), float(FTA), float(AST), float(TRB), float(PTS), float(BLK), float(t_year), float(TOV),
                      float(Age), float(G), float(t_POS)]]
        else:
            t_POS = float(yoe_2[POS])

            row_in_question = christine_subset[christine_subset['True'] == name]

            list1 = [
                [float(row_in_question['Player'].values[0]), float(t_year), float(FG), float(FTA), float(AST),
                 float(TRB),
                 float(PTS), float(BLK), float(YOE), float(t_POS), float(last_salary)]]
    except (ValueError, KeyError, TypeError):
        return "Invalid Entry."

    if (YOE == 1) or (name is None):
        if name is None:
            name = "this hypothetical player"
        t_POS = float(yoe_1[POS])

        list1 = [[float(FG), float(FTA), float(AST), float(TRB), float(PTS), float(BLK), float(t_year), float(TOV),
                  float(Age), float(G), float(t_POS)]]
        df2 = pd.DataFrame(columns=['FG', 'FTA', 'AST', 'TRB', 'PTS', 'BLK', 'SalStartYr', 'TOV', 'Age', 'G', 'Pos'],
                           data=list1)
        decision_tree_model_pkl = open('rookie_dt.pkl', 'rb')
        dt1 = pickle.load(decision_tree_model_pkl)
        df2['Cluster'] = float(dt1.predict(df2))
        final_df = pd.DataFrame(df2.values.tolist())
        loaded_model = xgb.Booster()

        # loaded_model.load_model('overall_rookie.model')
        # predictions = np.array([i for i in loaded_model.predict(xgb.DMatrix(final_df))])

        if int(df2['Cluster']) == 0:
            loaded_model.load_model('cluster0_rookie.model')
            predictions = np.array([i for i in loaded_model.predict(xgb.DMatrix(final_df))])
        elif int(df2['Cluster']) == 1:
            loaded_model.load_model('cluster1_rookie.model')
            predictions = np.array([i for i in loaded_model.predict(xgb.DMatrix(final_df))])
        else:
            loaded_model.load_model('cluster2_rookie.model')
            predictions = np.array([i for i in loaded_model.predict(xgb.DMatrix(final_df))])

    else:
        t_POS = float(yoe_2[POS])

        row_in_question = christine_subset[christine_subset['True'] == name]

        list1 = [
            [float(row_in_question['Player'].values[0]), float(t_year), float(FG), float(FTA), float(AST), float(TRB),
             float(PTS),
             float(BLK), float(YOE), float(t_POS), float(last_salary)]]
        df2 = pd.DataFrame(
            columns=['Player', 'SalStartYr', 'FG', 'FTA', 'AST', 'TRB', 'PTS', 'BLK', 'years_of_exp', 'Pos',
                     'salary_before'],
            data=list1)
        decision_tree_model_pkl = open('veterans_dt.pkl', 'rb')
        dt2 = pickle.load(decision_tree_model_pkl)
        df2['Cluster'] = float(dt2.predict(df2))
        final_df = pd.DataFrame(df2.values.tolist())
        loaded_model = xgb.Booster()

        # loaded_model.load_model('overall_veteran.model')
        # predictions = np.array([i for i in loaded_model.predict(xgb.DMatrix(final_df))])

        if int(df2['Cluster']) == 0:
            loaded_model.load_model('cluster0_veteran.model')
            predictions = np.array([i for i in loaded_model.predict(xgb.DMatrix(final_df))])
        elif int(df2['Cluster']) == 1:
            loaded_model.load_model('cluster1_veteran.model')
            predictions = np.array([i for i in loaded_model.predict(xgb.DMatrix(final_df))])
        elif name in goats:
            loaded_model.load_model('outliers_veteran.model')
            predictions = np.array([i for i in loaded_model.predict(xgb.DMatrix(final_df))])
        else:
            loaded_model.load_model('cluster2_veteran.model')
            predictions = np.array([i for i in loaded_model.predict(xgb.DMatrix(final_df))])

    return "I predict that " + str(name) + " will make " + "$" + "{:,}".format(int(predictions[0])) + " for the " + \
           str(year) + " season."


if __name__ == '__main__':
    app.run_server(debug=True)

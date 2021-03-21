# Data_Science_NBA_Project
Our project aims to predict NBA players' annual salaries based on their game statistics using machine learning.

Try it out! 
https://nba-salary-predictor.herokuapp.com/

# PROJECT OVERVIEW
Each year, a player’s annual salary is determined by the player’s impact, which is measured by various factors: points per game, number of rebounds, 3-point field goal percentage, and so forth. Our goal was to create a single dataset with players’ salaries along with their statistics over the years, determine the most influential factors on a player’s salary, and then leverage machine learning to determine an algorithm that will enable us to best predict a player’s annual salary based on these statistics. Our main language for this project was Python.

# DATA PREPROCESSING
Our data sources for this project were Basketball Reference, which we used to obtain individual players’ game statistics per season (i.e. average number of points per game), and Hoops Hype, which we used to obtain the individual players’ salaries per season. As the NBA determines a player’s salary for that season prior to the start of the season, we made the assumption that the player’s performance in the season prior would be used to determine the player’s salary the following season. Using a combination of Excel and Python, we mapped the player’s game statistics to their respective salaries, ultimately merging the players’ seasonal statistics data set with the players’ salaries dataset from the following season. We also added the columns:
StartYr - the season’s start year from which the statistics were recorded
EndYr - the season’s end year from which the statistics were recorded
SalStartYr - the following season’s start year (also the season when player is paid)
SalEndYr - the following season’s end year (also the season when player is paid)
Using python, we merged all the seasonal data sets with the salaries and statistics from the year 1980 on. However, we could not use 1980-1989 data as there was no record of the players’ salaries on Hoops Hype.

Working on Google Colab and using Python (specifically the Pandas package), we worked on preprocessing the now merged data set with all the players’ statistics and salaries. We subsetted the data set to only include the years 2000 to 2020 on as we wanted the data to be as current as possible. We later decided to use the years 1990 to 2020. Furthermore, we dropped the rows with missing values.

After doing some exploratory data analysis and modeling, we added the column years of experience by counting the number of times a player appeared in the data set.  For instance, a player who only appeared once would have 1 year of experience.

Towards the end of our project, we also decided to add the column Salary from the previous season to the players who appeared more than once in the dataset.

Finally, we tried two different target variables. In our initial regression and classification models, we decided to normalize the Salary variable by dividing by Salary Cap. Our reasoning behind this was because Salary Cap changes each season (a team’s salary cap is the amount of money a team has to spend on its players), we would predict the percentage of the salary cap a player would be paid. In later approaches, we predicted purely salary as it seemed that the year accounted for the change in salary cap.

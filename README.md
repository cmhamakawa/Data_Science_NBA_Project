# Data_Science_NBA_Project
Our project aims to predict NBA players' annual salaries based on their game statistics using machine learning.

Try it out! 
https://nba-salary-predictor.herokuapp.com/

# Project Overview
Each year, a player’s annual salary is determined by the player’s impact, which is measured by various factors: points per game, number of rebounds, 3-point field goal percentage, and so forth. Our goal was to create a single dataset with players’ salaries along with their statistics over the years, determine the most influential factors on a player’s salary, and then leverage machine learning to determine an algorithm that will enable us to best predict a player’s annual salary based on these statistics. Our main language for this project was Python.

# Data Preprocessing
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

# Exploratory Data Analysis
Over the course of this project, we tried several different models, both regression and classification. As different people were delegated to do different models, we had various visuals that both contributed to the intended model and contributed to the overall project as well.

The main goal of our exploratory data analysis was to determine the best predictors (and best combination of predictors) that we could use to predict a player’s normalized salary or salary. We first used a heat map to provide us with a general idea of the predictors that were most highly correlated with salary.

<img width="586" alt="dsu_1" src="https://user-images.githubusercontent.com/76538403/111924635-ce72f300-8a62-11eb-94b1-4e82ac5dfcdd.png">

<img width="573" alt="dsu_2" src="https://user-images.githubusercontent.com/76538403/111924648-d59a0100-8a62-11eb-90d1-0fd3e9f80a43.png">

<img width="919" alt="dsu_3" src="https://user-images.githubusercontent.com/76538403/111924656-db8fe200-8a62-11eb-84a7-74c19c1842ef.png">

The use of lasso regression (and this visual) also helped point us towards a subset of variables that we could use for our model. Another method that we used to find a subset of variables for our model was forward selection in R.

After doing some broad analysis of variables, we delved deeper to see if there was correlation between our predictor variables along with the strength of the correlation between individual variables and our target variable.

<img width="664" alt="dsu_4" src="https://user-images.githubusercontent.com/76538403/111924663-e2b6f000-8a62-11eb-949b-d3af72d03e27.png">

This image shows the relationship between points and salary, color coded by the position of the player. In this graph, we see that there is a generally positive correlation between points and salary, which corroborates our findings from our previous visuals. Furthermore, we see that the position variable may not be a strong predictor for our target variable Salary.

<img width="679" alt="dsu_6" src="https://user-images.githubusercontent.com/76538403/111924682-f3676600-8a62-11eb-8c95-208e603347fd.png">

One column or variable that we added later on in the process was years of experience. After adding this variable, we observed a fairly strong positive correlation between years of experience and both salary and normalized salary. However, upon looking at this visual, we did also observe a peak and then a dip, so there is a point where an increase in years of experience resulted in a lower salary. A reason for this dip in salary as years increase is due to the fact that a players’ older age may impact their performance.

<img width="555" alt="dsu_7" src="https://user-images.githubusercontent.com/76538403/111924694-feba9180-8a62-11eb-92c4-55f5a99769cc.png">

This visualization brings to light the fact that there are a large number of players that are paid in a certain range or amount; as anticipated, there are a relatively small number of players paid higher than 0.2 and an even smaller number of players paid more than 0.3 of the salary cap. 

<img width="812" alt="dsu_8" src="https://user-images.githubusercontent.com/76538403/111924703-067a3600-8a63-11eb-8a51-bbedd0fb87e9.png">

This visual was a visual for lasso regression. Like the visual prior, we notice that there are a large cluster of players towards the bottom of the fitted line. We then see the higher paid players’ actual salaries dip away from the line.

From the last two visuals, we were able to see that our model would somehow have to account for the imbalance in players’ salaries. As a majority of players are paid in a similar salary range and the “all stars” or more well-known players are a relatively smaller group, we would have to somehow account for the imbalance in data and possibly outliers.

This section does not cover all of our data visualizations: it simply covers the major observations we made. We further cover in detail our visualizations for each model in the following sections.


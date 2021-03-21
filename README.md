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


# Machine Learning Models
## Regression
### Ridge Regression

Ridge Regression is similar to standard linear regression, but it adds a penalty to the weight matrix, performing L2 regularization. This helps alleviate the effects of interdependence of different variables. In our dataset, it was anticipated that different variables would be highly correlated. For example, years of experience would probably yield higher points, assists, etc. due to the added experience. For this reason, Ridge Regression seemed like a good idea to try. Overall, the results were not very good. The R-squared score was 0.272. R-squared generally ranges from 0-1, 0 meaning no learning occurred, 1 meaning the model has learned perfectly. Note: the R-squared score can be negative. It essentially means the same as 0 (nothing learned). When testing on the rookie dataset, the RMSE score was 1,190,732 which is also quite high considering that averages were in the low millions.

### Lasso Regression

To predict Salary (and Per_of_Salary_Cap initially), we tried using lasso regression because of its regularization and feature selection properties. Lasso regression is very similar to ridge regression in that it adds a penalty value to the simple linear regression model to prevent overfitting. Lasso regression is also useful for feature selection because it is able to reduce coefficients of insignificant variables in the model to 0. 

For the most part, lasso regression was not very effective at predicting Salary. With our best subset of predictors (FG, FTA, AST, TRB, PTS, BLK, SalStartYr, TDV, Age, G, Pos), lasso regression had mediocre performance at best, predicting rookie salaries with an R2 score of 0.47615 and RMSE of 1214715.13122, and predicting veteran salaries with an R2 score of 0.59326 and RMSE of 3067946.81366. It performed even worse for clusters within the rookie and veteran groups, with most tests never reaching an R2 score above 0.2 and having a very high RMSE. The one exception was Cluster 1 of the rookies, which had an R2 score of 0.50758 and an RMSE of 1033420.19457. Lasso regression was ultimately not as successful at predicting Salary as classification models we used, but it did help point us to which variables were best at predicting Salary.

<img width="653" alt="dsu_9" src="https://user-images.githubusercontent.com/76538403/111924812-81435100-8a63-11eb-961e-4a0e7393f6cb.png">

<img width="479" alt="dsu_10" src="https://user-images.githubusercontent.com/76538403/111924824-886a5f00-8a63-11eb-8a37-c029fe65c180.png">

<img width="503" alt="dsu_11" src="https://user-images.githubusercontent.com/76538403/111924829-8d2f1300-8a63-11eb-9eaa-4469feff428a.png">

**Note**: The blue dashed line at the x-axis means that the coefficient of the variable is 0. If a variable’s coefficient is 0, this means that the variable is insignificant.

### Gradient Boost Regression
Gradient boost regression is a type of machine learning boosting model. This means that it uses several different models, with each successive model trying to correct the errors of the last. Gradient boosting aims to predict the target outcomes for the next model to minimize error. The target outcomes are based on the gradient of the error with respect to the prediction.

With regards to our project, gradient boost regression was more accurate when we were predicting the percent of salary cap. The predictors that were used years_of_exp, MP, GS, PTS, FG, AST, STL, BLK, FTA. And, with gradient boosting, we had an R2 score of 0.6750. But, when we switched to predicting actual salary, gradient boost regression did not prove to be useful. The results when predicting salary were worse than predicting percent of salary cap, regardless of the predictors used.


### Linear Regression
Initially, we tried linear and polynomial regression using all the predictors to see results. We had two different analysts try linear and polynomial regression.

The results from the first linear and polynomial regression models we tried are shown below. This used all players (rookies and veterans) to predict a percent of the total salary cap. Our validation results were:
Linear Regression: R2 score .597
Polynomial Regression (Deg 2): R2 score 0.627
Polynomial Regression (Deg 3): R2 score -10.9
Choosing the best (degree 2), the test results were an R2 score of 0.66 with an RMSE of 0.052. It should be noted that because this was predicting percentages, the RMSE is much lower.

The  results from the second linear and polynomial regression models we tried are also shown below.

Predicting Per_of_Salary_Cap using linear regression
R2: 0.579
RMSE: 0.05739

<img width="643" alt="dsu_12" src="https://user-images.githubusercontent.com/76538403/111924906-ee56e680-8a63-11eb-8280-6dfec4bffedb.png">

<img width="633" alt="dsu_13" src="https://user-images.githubusercontent.com/76538403/111924911-f3b43100-8a63-11eb-9c1c-0fea7221edfe.png">


## Classification

We also wanted to attempt some pure classification models in order to predict salary groupings rather than a specific number. Starting with our cleaned data set, we first used clustering in order to cluster the salary data and see if there was any relationship between players’ stats and their respective salary cluster. In order to find the optimal number of salary clusters, we used the elbow method, testing values 1 - 10 and found 4 to be the optimal number of clusters. Thus, each row had a cluster number (0,1,2,3) which we appended to the data set. 

From the above graph, we see that the point of inflection or “elbow” occurs at k = 4. From here, we then implemented some classification models to see how well we could predict cluster values (0,1,2,3), which corresponds to a range of salary values. In other words, we wanted to use player data/statistics to predict what salary cluster they might end up in. We implemented 4 models: logistic regression, decision tree, random forest, and support vector machine. The score for the classification models were as follows:

<img width="634" alt="dsu_14" src="https://user-images.githubusercontent.com/76538403/111924966-21997580-8a64-11eb-9351-4ab44370b463.png">


From the above scores we notice that they are consistently around high 50s with some slight evidence of overfitting in the data. These results taught us that we needed more data in order to better predict salary as each model would max out at these scores even after using cross validation. These findings led us to our next idea which involved adding previous year salary for each veteran in the original data set. Since we earlier found that not any one player statistic directly correlated with salary, then adding previous year salary would be a good indicator as to what a player would make the following year. After adding previous year salary to the data set and redoing the above steps, we achieved much better results (around 90% for Veterans for each model for each score and near perfect for Rookies for each model and each score). These steps were implemented in our final model in which we used both classification and regression models. 

## Classification with Regression
 
After evaluating all our models, we came to the conclusion that classification with regression would yield the best outcome. Having grouped our data by players with years of experience equal to 1 (i.e. Rookies) and experienced players (i.e. Veterans), we proceeded to further create subgroups in an effort to reduce variation. By using K-Means clustering, we achieved this task and landed on creating three separate sub groups across both the Rookie and Veteran subsets.

Based on previous feature selection from earlier, we knew what subsets of columns could be considered good predictors for both the Rookie and Veterans subsets. At this point in time, we decided to transition from pure regression techniques to XGBoost. XGBoost is a decision-tree based algorithm that incorporates gradient boosting. What this means is that XGBoost minimizes errors in sequential models, and in turn is well tuned to medium data that is structured or tabular - like our NBA dataset.

Once more we use RMSE as a measure of how well our model is predicting salaries. At the end of the day, RMSE is simply a standard way to measure the error in a model. Heuristically, RMSE can be considered to be the “normalized distance” between for example a vector of observed values and a vector of predicted values. After optimizing hyperparameters with XGBoost in an effort to decrease RMSE, we ended with the following results:

<img width="634" alt="dsu_15" src="https://user-images.githubusercontent.com/76538403/111925097-a8e6e900-8a64-11eb-850d-d564d6579eba.png">


The results of XGBoost can be illustrated in the below plots which look at how well our predicted results line up with the actual observed values. On the x-axis, we have a sample of 100 players from each of the subsets (Rookies and Veterans). On the y-axis, we have their salary: both predicted by our model and the actual observed value.

<img width="648" alt="dsu_16" src="https://user-images.githubusercontent.com/76538403/111925102-ad130680-8a64-11eb-852c-28ab657f1d63.png">

<img width="670" alt="dsu_17" src="https://user-images.githubusercontent.com/76538403/111925111-b4d2ab00-8a64-11eb-8a3f-0d67867d6555.png">

From the above plots, it’s clear that our Veteran model typically outperforms our model for Rookie players. The reason for this discrepancy can likely be attributed to the fact that our Veteran model takes into account a player’s salary the season before when predicting a player’s salary for the next season. The Rookie model does not have this luxury seeing as Rookie player salaries are modeled based on raw stats alone and not their salary from a previous year. 


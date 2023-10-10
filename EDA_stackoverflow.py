"""
Author: Pachimatla Rajesh

Exploratory Data Analysis

Case Study: 2023 Stack Overflow Developer Survey, 
the longest runningsurvey of software developers (and anyone else who codes!)

The following areas been analysed in this code:
1. Demographics of the survey respondents and the global programming community
2. Distribution of programming skills, experience, and preferences
3. mployment-related information, preferences, and opinions
    
"""
#### import section #######

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

######### 1. Import the data ##########
survey_raw_df = pd.read_csv('survey_results_public.csv')
#print(survey_raw_df.columns)

#Since, Schema file has only two columns, upload it as pandas series rather than dataframe
#take column as index and read the question text
schema_fname = 'survey_results_schema.csv'
schema_raw = pd.read_csv(schema_fname, index_col='Column').QuestionText
#print(schema_raw)

#Check whether we read correctly
print("YearsCodePro question is:=>> ", schema_raw['YearsCodePro'])

##### 2. DATA CLEANING AND PREPARATION ###########

selected_columns = [
    # Demographics
    'Country',
    'Age',
    'Gender',
    'EdLevel',
    'UndergradMajor',
    # Programming experience
    'Hobbyist',
    'Age1stCode',
    'YearsCode',
    'YearsCodePro',
    'LanguageWorkedWith',
    'LanguageDesireNextYear',
    'NEWLearn',
    'NEWStuck',
    # Employment
    'Employment',
    'DevType',
    'WorkWeekHrs',
    'JobSat',
    'JobFactors',
    'NEWOvertime',
    'NEWEdImpt'
]

print("Length of selected column is :=>>", len(selected_columns))

#It is always better to create duplicate dataset and schema (in this case)

survey_df = survey_raw_df[selected_columns].copy()
schema = schema_raw[selected_columns]

#print("Shape of the survey df is :=>>", survey_df.shape)

survey_df.info()
# Only two of the columns were detected as numeric columns (Age and WorkWeekHrs),
# even though a few other columns have mostly numeric values.
# To make our analysis easier, let's convert some other columns into numeric data
# types while ignoring any non-numeric value. 
#The non-numeric are converted to NaN.

survey_df['Age1stCode'] = pd.to_numeric(survey_df.Age1stCode, errors='coerce')
#errors='coerce': The 'errors' parameter is set to 'coerce', which means that 
#if any values in the 'Age1stCode' column cannot be converted to numeric, 
#they will be replaced with NaN (Not-a-Number) values instead of raising an error.
survey_df['YearsCode'] = pd.to_numeric(survey_df.YearsCode, errors='coerce')
survey_df['YearsCodePro'] = pd.to_numeric(survey_df.YearsCodePro, errors='coerce')

survey_df.describe(include='all')
#After looking at the descriptive information, we drop the data which doesnt 
#make sense like we can remove outliers

survey_df.drop(survey_df[survey_df.Age < 10].index, inplace = True)
survey_df.drop(survey_df[survey_df.Age > 100].index, inplace = True)

survey_df.drop(survey_df[survey_df.WorkWeekHrs > 140].index, inplace = True)

survey_df.where(~(survey_df.Gender.str.contains(';', na=False)), np.nan, inplace=True)
# it essentially checks for rows where the 'Gender' column does not contain a semicolon.
#This is the value that will replace the rows where the condition is not met

print(survey_df.sample(10))

################## 3.DATA VISUALISATION ###################

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
#he rcParams attribute of the matplotlib library to configure various properties 
#of the Matplotlib library, including the font size.
matplotlib.rcParams['figure.figsize'] = (8, 4)
matplotlib.rcParams['figure.facecolor'] = 'white'

#Figure 1: where do you live?
#Lets see how respondants are from top 15 countries
top_countries = survey_df.Country.value_counts().head(15)
plt.figure(figsize=(12,6))
plt.xticks(rotation=75)
plt.title(schema.Country)
sns.barplot(x=top_countries.index, y=top_countries)
#plt.bar(top_countries.index, top_countries);
plt.show()

#Figure 2: What is your Age?
#To see the age group of respondant, lt us use histogram
plt.figure(figsize=(12, 6))
plt.title(schema.Age)
plt.xlabel('Age')
plt.ylabel('Number of respondents')
plt.hist(survey_df.Age, bins=np.arange(10,80,5), color='purple');
plt.show()

#Figure 3: Which of the following describes you? 
#To check the count of respondants based on their gender we can barplot or pie plot
gender_counts = survey_df.Gender.value_counts()
#print(gender_counts)
plt.figure(figsize=(12,6))
plt.title(schema.Gender)
plt.pie(gender_counts, labels=gender_counts.index, autopct="%1.1f%%",startangle=180)
#sns.barplot(x=gender_counts.index, y=gender_counts)

#Figure 4: Highest level formal education, in count?
#ude countplot to see the each level of education of respondants
plt.figure(figsize=(12,6))
sns.countplot(y=survey_df.EdLevel)
plt.xticks(rotation=75);
plt.title(schema['EdLevel'])
plt.ylabel(None);

#Figure 5: Highest level formal education, in percantage?
EdLevel_pct = survey_df.EdLevel.value_counts()*100/survey_df.EdLevel.count()
#print(EdLevel_pct)
plt.figure(figsize=(12,6))
sns.barplot(x=EdLevel_pct, y=EdLevel_pct.index)
plt.title(schema.EdLevel)
plt.ylabel(None);
plt.xlabel('Percentage');

#Figure 6: Primary field of study, undergraduation?
undergrad_pct = survey_df.UndergradMajor.value_counts() * 100 / survey_df.UndergradMajor.count()
plt.figure(figsize=(12,6))
sns.barplot(x=undergrad_pct, y=undergrad_pct.index)
plt.title(schema.UndergradMajor)
plt.ylabel(None);
plt.xlabel('Percentage');

#Figure 7: Current Employemnt Status?
Empl_count = survey_df.Employment.value_counts()*100/survey_df.Employment.count()
plt.figure(figsize=(12,6))
sns.barplot(y=Empl_count.index,x=Empl_count)
plt.title(schema.Employment)
plt.ylabel(None)
plt.xlabel('Employment')
plt.show()


print(survey_df.DevType)
#For Devype column/feature multiple skills are entered seperated by semicolon (;)
#Lets create a function to split the column and identify developer type
#i.e. it creates DataFrame where skill type columns and True or False are assigned to
#each respondants

def split_multicolumn(col_series):
    result_df = col_series.to_frame()
    #Converts the series to dataframe
    options = []
    # Iterate over the column
    for idx, value  in col_series[col_series.notnull()].items():
        #idx is index and value is string in second column where somethin is there, not null
        # Break each value into list of options
        for option in value.split(';'):
            # Add the option as a column to result
            if not option in result_df.columns:
                options.append(option)
                result_df[option] = False
            # Mark the value in the option column as True
            result_df.at[idx, option] = True
    return result_df[options]

dev_type_df = split_multicolumn(survey_df.DevType)
dev_type_totals = dev_type_df.sum().sort_values(ascending=False)
#print(dev_type_totals)

#Figure 9: Developers type total?

plt.figure(figsize=(12,6))
sns.barplot(y=dev_type_totals.index,x=dev_type_totals)
plt.title("Developers type total")
plt.ylabel(None)
plt.xlabel('Employment')
plt.show()

############# 4. Exploratory DATA ANALYSIS ################3

#Q.1. What percentage of respondents work in roles related to data science?
ds_rolespercent = round(dev_type_df['Data scientist or machine learning specialist'].sum()*100/dev_type_totals.sum(),2)
print("The percentage of respondents work in roles related to data science is:==>", ds_rolespercent)

#Q.2. What are the most popular programming languages 2020?
languages_worked_df = split_multicolumn(survey_df.LanguageWorkedWith)
languages_worked_percentages = languages_worked_df.mean().sort_values(ascending=False) * 100
plt.figure(figsize=(12, 12))
sns.barplot(x=languages_worked_percentages, y=languages_worked_percentages.index)
plt.title("Languages used in the past year");
plt.xlabel('count');

#Q.3. Which languages are the most people interested to learn over the next year
languages_interested_df = split_multicolumn(survey_df.LanguageDesireNextYear)
languages_interested_percentages = languages_interested_df.mean().sort_values(ascending=False) * 100

plt.figure(figsize=(12, 12))
sns.barplot(x=languages_interested_percentages, y=languages_interested_percentages.index)
plt.title("Languages people are intersted in learning over the next year");
plt.xlabel('count');

#Q.4.Which are the most loved languages, i.e., a high percentage of people who have used the language want to continue
# learning & using it over the next year
languages_loved_df = languages_worked_df & languages_interested_df
#To perform a logical AND operation between two DataFrames and create a new 
#DataFrame with the result, you can use the & operator between the DataFrames only 
#if both DataFrames have the same shape (i.e., the same number of rows and columns) 
#and you want to perform element-wise logical AND between corresponding elements.
languages_loved_percentages = (languages_loved_df.sum() * 100/ languages_worked_df.sum()).sort_values(ascending=False)
plt.figure(figsize=(12, 12))
sns.barplot(x=languages_loved_percentages, y=languages_loved_percentages.index)
plt.title("Most loved languages");
plt.xlabel('count');

#Q.5.In which countries do developers work the highest number of hours per week? Consider countries with
# more than 250 responses only
countries_df = survey_df.groupby('Country')[['WorkWeekHrs']].mean().sort_values('WorkWeekHrs', ascending=False)
high_response_countries_df = countries_df.loc[survey_df.Country.value_counts() > 250].head(15)
print(high_response_countries_df)

#Q.6.How important is it to start young to build a career in programming
plt.figure(figsize=(12, 12))
sns.scatterplot(x='Age', y='YearsCodePro', hue='Hobbyist', data=survey_df)
#hue='Hobbyist': This specifies that the 'Hobbyist' column from the 'survey_df' 
#DataFrame will be used to color the points in the scatterplot.
plt.xlabel("Age")
plt.ylabel("Years of professional coding experience");

plt.figure(figsize=(12, 12))
plt.title(schema.Age1stCode)
sns.histplot(x=survey_df.Age1stCode, bins=30, kde=True);
plt.show()

################################# END OF SCRIPT ############################3
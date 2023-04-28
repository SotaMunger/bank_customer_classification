## Bank Customer Classification Ensemble Models

This end-to-end data science project compares the performance of three ensemble classification models that predict bank customers' employment sectors (public, private, or unemployed) based on their demographic data and credit card spending habits. The three models tested here include a bagging classifier, an Adaboost classifier, and a gradient boost classifier (from the sklearn.ensemble class).  All three models use decision trees as their base algorithms.

The steps in the project workflow (explained in more detail below) include:

1. Data cleaning and feature engineering
2. Exploratory data analysis
3. Data processing
4. Dimensionality reduction
5. Hyperparameter tuning
6. Training
7. Testing

The data set used in this analysis, which consists of deidentified customer data from a Turkish bank, is proprietary and cannot be shared in this repository.

**Data cleaning and feature engineering**

The first data set, consisting of 103,202 entries, provided demographic information for individual customers. After dropping geographical coordinate columns for branch, home, and work locations (which were redundant with the branch, home, and work region columns), the demographic data set consisted of the following features:

Cust_segment - the type of account the customer held
Branch_ID - the bank branch number where the customer opened the account
Gender (binary)
Marital_Status (see the translation section in the addendum for a list of categories)
Education_lv (see the translation section in the addendum for a list of categories)
Job_Status - employment categories (later grouped as private, public, or unemployed for this analysis and used as labels)
Income - annual earnings
Age
Years_w_Bank - number of years as a customer
Penalty_ (12 columns) - credit card payment delinquincy status for each month from July 2014 to June 2015
Work_Region - geographical region in Turkey of the customer's workplace
Home_Region - geographical region of the customer's home
Branch_Region - geographical region of the customer's primary bank branch

The data types are found in the df.info() output below:

![Screenshot from 2023-04-27 21-31-57](https://user-images.githubusercontent.com/91567553/235192424-40110bcc-44e6-40b2-85fa-7c959ca2b0ea.png)

The penalty columns for each month were first converted to columns consisting of counts for each category i.e. one column for RISKSIZ (Good Standing), one column for GECIKME 1-15 GUN (Late 1-15 days), etc. with counts for the number of times the customer held that status through the 12 month period.  Then, the Job_Status categories were grouped accordingly:

![Screenshot from 2023-04-27 21-57-36](https://user-images.githubusercontent.com/91567553/235192635-0662de8a-56be-41be-9133-16352109f93f.png)

The categories 'DİĞER', 'TANIMSIZ', and 'YURTDIŞINDA ÇALIŞAN' were later dropped, primarily because they did not fit neatly into one of the three label groups (see the Exploratory Data Analysis Section below for more details). After regrouping, the three label categories had the following value counts:

![Screenshot from 2023-04-27 22-04-16](https://user-images.githubusercontent.com/91567553/235196577-fe98f444-2273-4251-9aea-c94ec6ded78f.png)

Next, the credit card transaction data set, which consisted of 9,335,625 unique transactions and features customer ID (Cust_ID), spending category (Category), and transaction amount (Trans_Amt) collected for each transaction, was re-engineered.  Transactions were grouped by customer ID and spending category and aggregated by the sum of the transaction amounts.  A pivot table was then created that displayed the amount each customer ID (rows) spent in each category (columns).  There were 25 spending categories in all (see the Spending Categories translation section in the addendum for the full list).  The transaction pivot table was then incorporated into the demographic data set by joining on customer ID.  The final data set contained the following features:

![image](https://user-images.githubusercontent.com/91567553/235198069-90b50bcd-bdcb-4297-92db-1a31401cf06c.png)

**Exploratory Data Analysis**

The data set was first checked for null values.  Oddly, there were 1345 customers that had null values for all credit card spending categories and were categorized in the 'DİĞER' (Other) employment category.  This finding gave further reason to drop 'DİĞER' customers from the data set.  Furthermore, there were only four customers that had null values in the Home_Region column and four customers that had null values in the Branch_Region column so these customers were likewise dropped.

Histograms created for each of the spending categories revealed that, in every category, spending had a prominent rightward skew as the majority of customers spent very little but at least a few customers spent a lot:

![image](https://user-images.githubusercontent.com/91567553/235198763-ded57a7b-d696-4ed1-a519-fa8c0591830f.png)

**Data Processing**
The data set was split into training and test sets at an 80/20 ratio using the StratifiedShuffleSplit class, which preserves the ratio of the employment categories in the training and test sets.  The 'Job_Status' column was split off from the data set and treated as the label.

A data processing pipeline was then built which included an imputer (sklearn.impute.SimpleImputer) to fill in any remaining null values with the median value of the column, a StandardScaler (sklearn.preprocessing.StandardScaler) to standardize all numerical values so that each entry in a column is converted to a z-score centered at mean 0, and categorical encoder (sklearn.preprocessing.OneHotEncoder) to convert categorical values into dummy variables in a sparse matrix.  The training set was then processed through the pipeline.

**Dimensionality Reduction**

Truncated singular value decomposition (TruncatedSVD from the sklearn.decomposition class) was then used to perform principle component analysis.  This algorithm identifies the vector components that contain the highest degree of variability (and thus the highest degree of predictive power). The TruncatedSVD implementation in this project set the explained variance cutoff value at 0.9, meaning that only components that cumulatively contribute the greatest amount of total data variance up to 90% should be retained.  Of the original 86 components, 46 were deemed unimportant for prediction.  Thus, the original 86 components contained in the data set were projected down to 40 components.

**Hyperparameter Tuning**

Hyperparameter tuning (shown in the accompanying bank_customer_identification_tuning.ipynb notebook) was performed on a small portion of the training set (10,000 observations) because tuning on the entire training set exhausted compute resources. RandomizedSearchCV, from the sklearn.model_selection class, which chooses randomly from a range of hyperparameter values was used to perform the tuning operation for each ensemble model.  Depending on the classifier model, such hyperparameters as number of estimators, learning rate, and decision tree depth were optimized.

**Training**

Each model was then trained on the entire training set, using the hyperparameter settings determined in the previous step. Feature importance scores were calculated to determine the features that held the greatest predictive power in each model.  Both the bagging classifier and gradient boost classifier considered gender, income, and customer segment to be the most important features, while the adaboost classifier considered customer segment, branch ID, and maritial status to be the most important.

**Testing**

The test sets were then run through the data processing pipeline and tested in each ensemble model. Performance was even across the board, with all models scoring 88% in accuracy.  The bagging classifier, however, was much less likely to predict that customers were public sector employees or unemployed, and instead favored the private sector category.  Both gradient boost and Adabost were much more likely to predict that customers worked in the public sector or were unemployed.

**Addendum**

Because the bank that shared its dataset is headquartered and operates in Turkey, some feature headings (demographic designations) and feature categories are written in Turkish.  Translations (Derived partly from Google Translate) for those headings and categories are as follows:

Customer Segment Categories:
BIREYSEL - Individual
BIREBIR - One to One
MİKRO - Micro
EXI26 - Exi26
ÖZEL BANKACILIK MÜŞTERİLERİ - Private Banking

Marital Status Categories:
EVLI - Married
BILINMIYOR - Unknown
BEKAR - Single
DUL - Widowed
BOSANMIS - Divorced

Education Level Categories:	
LISE - High school
ORTAOKUL - Elementary
UNIVERSITE - University
ILKOKUL - Primary school
BILINMIYOR - Unknown
LISASUSTU - Graduate school
YUKSEKOKUL - College (Junior)

Delinquency Status Headings:
RISKSIZ - Good standing
GECIKME 1-15 GUN - Late 1-15 days
GECIKME 16-29 GUN - Late 16-29 days
GECIKME 30-59 GUN - Late 30-59 days
GECIKME 60+ GUN - Late 60+ days
TAKIP - Referred to collection agency

Spending Categories:
SAGLIK - Health
NAKİT AVANS	- Cash Advance
DİĞER	- Other
RESTORAN	- Restaurant
TEKSTİL	- Textile
AYAKKABI	- Shoes
TEKNOLOJİ	- Technology
SİGORTA-MAIL ORDER - Insurance-Mail Order
BEYAZ EŞYA - Household Appliances
HİZMET SEKTÖRLERİ	- Service Sector
HAVAYOLLARI	- Airlines
ALIŞVERİŞ MERKEZLERİ - Shopping Malls
AKARYAKIT - Liquid Fuel
KUYUMCU - Jewelry
MUZIK MARKET KIRTASİYE - Music Market Stationery
MOBİLYA, DEKORASYON - Furniture, Decoration
KOZMETİK - Cosmetics
YAPI MALZ., HIRDAVAT, NALBURİYE - Building Materials, Hardware, Hardware
SPOR GİYİM - Sportswear
OPTİK - Optical
İÇKİLİ YERLER, KUMARHANE - Bars, Casinos
SİNEMA TİYATRO SANAT - Cinema, Theater, Art
ARABA KİRALAMA - Car Rental
HOTEL - Hotels
SİGORTA - Insurance
OTOMOTİV - Automotive
EĞİTİM - Education
EĞLENCE VE SPOR - Entertainment And Sports
OYUNCAK - Toys
MOTOSİKLET - Motorcycle
DOĞRUDAN PAZARLAMA-MAIL ORDER - Direct Marketing-Mail Order

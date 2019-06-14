import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')
plt.close('all')

pd.set_option('display.max_columns', 23)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print 'training dataset size: ' + str(train.shape)
print 'test dataset size: ' + str(test.shape)

#%%
train.head(4)
#%% Get rid of unnecessary columns
drop_cols = ['homepage','imdb_id','original_title','poster_path']
train = train.drop(columns = drop_cols)
test = test.drop(columns = drop_cols)


#%% Look at data
plt.figure();
ax1 = plt.subplot(131)
train.hist('budget',bins=100,ax=ax1)
ax2 = plt.subplot(132)
train.hist('revenue',bins=100,ax=ax2)
ax3 = plt.subplot(133)
train.plot(x='budget',y='revenue',style='.',ax=ax3,legend=False)
plt.ylabel('revenue')


#%% Data is skewed, look at log
def logCol(df,cols):
    for col in cols:
        df['log_'+col] = 0
        df.loc[df[col]!=0,'log_'+col] = df.loc[df[col]!=0,col].apply(np.log)

logCol(train,['budget','revenue'])
logCol(test,['budget'])



plt.figure()
ax1 = plt.subplot(131)
train.hist('log_budget',bins=20,ax=ax1)
ax2 = plt.subplot(132)
train.hist('log_revenue',bins=20,ax=ax2)
ax3 = plt.subplot(133)
train.plot(x='log_budget',y='log_revenue',style='.',ax=ax3,legend=False)
plt.ylabel('log_revenue')

##%%
#train.loc[train['log_budget']==0,'log_budget'] = train.loc[train['log_budget']!=0,'log_budget'].mean()
#test.loc[test['log_budget']==0,'log_budget'] = train.loc[train['log_budget']!=0,'log_budget'].mean()
#%% get date in usable format
def extract_date(df):
    df['release_date']  = pd.to_datetime(df['release_date'])
    df['release_year']  = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_day']   = df['release_date'].dt.day
    df['release_dow']   = df['release_date'].dt.weekday
    return df

train = extract_date(train)
test = extract_date(test)

plt.figure()
ax1 = plt.subplot(121)
train.plot('release_date','log_revenue',style='.',ax=ax1,legend=False)
plt.ylabel('log_revenue')
ax2 = plt.subplot(122)
plt.scatter(x=train['release_year']*train['log_budget'],y=train['log_revenue'])
plt.xlabel('release year * log budget')
plt.ylabel('log revenue')


#%% correct future dates

def makeDate(year,month,day):
    date_out = pd.to_datetime(year*10000 + month*100 + day,format='%Y%m%d')
    return date_out

def correctFutureDates(df):
    future = df['release_date']>pd.to_datetime('today')
    df.loc[future,'release_year'] = df.loc[future,'release_year']-100
    year = df.loc[future,'release_year']
    month = df.loc[future,'release_month']
    day = df.loc[future,'release_day']
    df.loc[future,'release_date'] = makeDate(year,month,day)
    df['release_dow'] = df['release_date'].dt.weekday
    df.loc[df['release_year'].isnull(),'release_year'] = df['release_year'].mean()
    return df

train = correctFutureDates(train)
test = correctFutureDates(test)

plt.figure()
ax1 = plt.subplot(121)
train.plot('release_date','log_revenue',style='.',ax=ax1,legend=False)
plt.ylabel('log_revenue')
ax2 = plt.subplot(122)
plt.scatter(x=train['release_year']*train['log_budget'],y=train['log_revenue'])
plt.xlabel('release year * log budget')
plt.ylabel('log revenue')
#%%

def plotErrorBy(df,col1,col2):
    x   = sorted(df[col2].unique())
    y   = train.groupby(col2)[col1].mean()
    sd  = train.groupby(col2)[col1].std()
    plt.errorbar(x,y,sd)
    plt.xlabel(col2)
    plt.ylabel(col1)

month_list = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

plt.figure()
ax1 = plt.subplot(131)
train.plot('release_month','log_revenue',style='.',ax=ax1,legend=False)
plt.ylabel('log_revenue')
plt.xticks(range(1,13),month_list,rotation='vertical',fontsize=8)
plt.subplot(132)
plotErrorBy(train,'log_revenue','release_month')
plt.xticks(range(1,13),month_list,rotation='vertical',fontsize=8)
ax3 = plt.subplot(133)
train.hist('release_month',ax=ax3,bins=range(1,14))
plt.xticks(np.array(range(1,13))+.5,month_list,rotation='vertical',fontsize=8)
plt.ylabel('count')
#%%

dow_list = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

plt.figure()
ax1 = plt.subplot(131)
train.plot('release_dow','log_revenue',style='.',ax=ax1,legend=False)
plt.ylabel('log_revenue')
plt.xticks(range(7),dow_list,rotation='vertical',fontsize=8)
plt.subplot(132)
plotErrorBy(train,'log_revenue','release_dow')
plt.xticks(range(7),dow_list,rotation='vertical',fontsize=8)
ax3 = plt.subplot(133)
train.hist('release_dow',ax=ax3,bins=range(8))
plt.xticks(np.array(range(7))+.5,dow_list,rotation='vertical',fontsize=8)
plt.ylabel('count')
#%% Parse columns with JSON formatting

# columns with json formatting
json_cols = ['genres', 'production_companies', 'production_countries', 'cast',
             'crew', 'spoken_languages', 'Keywords', 'belongs_to_collection']

# use eval to convert json strings to dict objects
def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return [x['name'] for x in d]

for col in json_cols:
    train[col] = train[col].apply(lambda x: get_dictionary(x))
    test[col]  = test[col].apply(lambda x: get_dictionary(x))

#%% make dict of filmography for all json objects (actor,prod_co,keyword,genre,etc.)
def get_json_dict(df):
    global json_cols
    result = dict()
    for col in json_cols:
        d = dict()
        for i in range(len(df[col])):
            if df[col][i] is None: continue
            for j in df[col][i]:
                if j not in d:
                    d[j] = []
                d[j].append(df['title'][i])
        result[col] = d

    return result

train_dict = get_json_dict(train)
test_dict  = get_json_dict(test)

#%% get expected revenue of top keywords
def topKeys(d,col,n):
    t = sorted([(len(v),k) for k,v in d[col].iteritems()],reverse=True)
    k = [x[1] for x in t[:n]]
    return k,t[:n]

def featureRevenue(df,col,keys):
    d = dict()
    for k in keys:
        d[k]=[]
        for i in range(len(df.index)):
            for j in df[col][i]:
                if j==k:
                    d[k].append(df.loc[i,'log_revenue'])
    return d

def featureStats(df,col,keys):
    r = featureRevenue(df,col,keys)
    out = zip(*sorted([(np.mean(v)-np.mean(df['log_revenue']),np.std(v)/np.sqrt(len(v)),k) for k,v in r.iteritems()],reverse=True))
    return out

def pltFeatureStats(feat_stats,cut,xlabel):
    plt.figure()
    plt.bar(range(cut),feat_stats[0][:cut],yerr=feat_stats[1][:cut])
    plt.ylabel('delta log revenue')
    plt.xlabel(xlabel)
    plt.xticks(range(cut),feat_stats[2][:cut],rotation='vertical',fontsize=8)
    plt.subplots_adjust(bottom=0.4)

keyword_cut = 50
top_Keywords,t = topKeys(train_dict,'Keywords',keyword_cut)
keyword_stats = featureStats(train,'Keywords',top_Keywords)
pltFeatureStats(keyword_stats,keyword_cut,'Keywords')

#%%
cast_cut = 30
top_cast,t = topKeys(train_dict,'cast',cast_cut)
cast_stats = featureStats(train,'cast',top_cast)
pltFeatureStats(cast_stats,cast_cut,'cast')

#%%
genre_cut = 10
top_genres,t = topKeys(train_dict,'genres',genre_cut)
genre_stats = featureStats(train,'genres',top_genres)
pltFeatureStats(genre_stats,genre_cut,'genre')

#%%
pro_co_cut = 10
top_production_companies,t = topKeys(train_dict,'production_companies',pro_co_cut)
pro_co_stats = featureStats(train,'production_companies',top_production_companies)
pltFeatureStats(pro_co_stats,pro_co_cut,'production_company')


#%%
def makeDummies(df,keys,d):
    df = df.join(pd.DataFrame(columns=keys))
    df.loc[:,keys] = 0
    for k in keys:
        if k in d:
            df.loc[df['title'].isin(d[k]),k]=1
    return df

dummy_list = ['Keywords','genres','cast','production_companies']
for dum_col in dummy_list:
    train = makeDummies(train,eval('top_'+dum_col),train_dict[dum_col])
    test  = makeDummies(test,eval('top_'+dum_col),test_dict[dum_col])
#%%

def monthDummies(df):
    global month_list
    dummies = pd.DataFrame(data=pd.get_dummies(df['release_month'],drop_first=True))
    dummies = dummies.rename(mapper=dict(zip(range(2,13),month_list[1:])),axis='columns')

    df = df.join(dummies)
    return df

train = monthDummies(train)
test  = monthDummies(test)


#%%
def dowDummies(df):
    global dow_list
    dummies = pd.DataFrame(data=pd.get_dummies(df['release_dow'],drop_first=True))
    dummies = dummies.rename(mapper=dict(zip(range(1,7),dow_list[1:])),axis='columns')
    df = df.join(dummies)
    return df

train = dowDummies(train)
test  = dowDummies(test)
#%%


train['log_budget_x_year'] = train['log_budget']*train['release_year']
test['log_budget_x_year']  = test['log_budget']*test['release_year']

train['log_budget_x_runtime'] = train['log_budget']*train['runtime']
test['log_budget_x_runtime']  = test['log_budget']*test['runtime']

x_cols = ['log_budget','release_year','log_budget_x_year','runtime','log_budget_x_runtime']
dummy_cols = month_list[1:] + dow_list[1:] + top_Keywords + top_genres + top_cast + top_production_companies

for col in x_cols:
    train['z_' + col] = (train[col]-train[col].mean())/train[col].std()
    test['z_' + col]  = (test[col]-train[col].mean())/train[col].std()
    train.loc[train['z_'+ col].isnull(),['z_'+ col]] = 0
    test.loc[test['z_'+ col].isnull(),['z_'+ col]] = 0

z_cols = ['z_'+col for col in x_cols]
x_cols = z_cols + dummy_cols
y_cols = ['log_revenue']

category_list = [1]*(len(month_list)-1)+[2]*(len(dow_list)-1)+[3]*len(top_Keywords)+[4]*len(top_genres)+[5]*len(top_cast)+[6]*len(top_production_companies)
indices = [[x==i for x in category_list] for i in set(category_list)]

lm = linear_model.LinearRegression()
lm.fit(train[x_cols],train[y_cols])

plt.figure()
plt.bar(x_cols,lm.coef_[0])
plt.xticks(rotation='vertical',fontsize=7)
plt.subplots_adjust(bottom=0.4)
plt.figure()
bar_list = plt.bar(x_cols[len(z_cols):],lm.coef_[0][len(z_cols):])
for i in range(len(indices)):
    for j in range(len(bar_list)):
        if indices[i][j]:
            bar_list[j].set_color()
plt.xticks(rotation='vertical',fontsize=7)
plt.subplots_adjust(bottom=0.4)
#%%
log_predict = lm.predict(test[x_cols])
predict = np.e**log_predict

submission = pd.DataFrame(test['id'])
submission['revenue'] = predict

submission.to_csv(path_or_buf='submission.csv',index=False)

#!/home/pinoystat/Documents/python/environment/datascience/bin/python
# coding: utf-8
# Preprocessing Script

#importing:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy import sparse 

import re
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler 
from sklearn.preprocessing import SplineTransformer, PowerTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

#models
import joblib


# In[2]:


input_folder = "kaggle/input/linking-writing-processes-to-writing-quality/"
train_logs = pd.read_csv(input_folder + "train_logs.csv",delimiter = ",",header = 0)
train_scores = pd.read_csv(input_folder +"train_scores.csv", delimiter = ",", header = 0)
scores = pd.Series(data = train_scores['score'].values, index = train_scores['id'].values, name = 'score')
test_logs = pd.read_csv(input_folder + "test_logs.csv",delimiter = ",",header = 0)


# In[3]:


# Feature Engineering for transformer for cursor position
class CursorPositionTransformer(BaseEstimator, TransformerMixin):
    

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("Started Cursor position Feature Engineering")
        z = X.groupby('id')['cursor_position'].aggregate([self.cp_sum_backstep,self.cp_n_backstep, 
                                                          self.cp_sum_forwardstep, self.cp_n_forwardstep, 
                                                          lambda x: np.log(np.mean(x)+1)])

        print("Done..")
        print(z)
        return sparse.csr_matrix(z.values)

    def cp_sum_backstep(self,x):
        n1 = np.diff(np.log(x+1))
        return np.sum(n1[n1 < 0])
    def cp_n_backstep(self,x):
        n1 = np.diff(np.log(x+1))
        return np.log((n1<0).sum()+1)

    def cp_sum_forwardstep(self,x):
        n1 = np.diff(np.log(x+1))
        return np.sum(n1[n1 > 0])
                        
    def cp_skew_forwardstep(self,x):
        n1 = np.diff(np.log(x+1))
        return st.skew(n1[n1 > 0])
                                            
    def cp_n_forwardstep(self,x):
        n1 = np.diff(np.log(x+1))
        return np.log((n1>0).sum()+1)
                                                                
    def cp_change_stat(self,x):
        n1 = np.diff(np.log(x+1))
        return np.mean(n1)

 
 # word_count feature engineering
 # Based on the graph above, we can count the number of zero changes and get the mean:
 # wc_zero_change will return the count of all non-zero steps taken by the person

class WordCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def wc_non_zero_change(self, x):
        n1 = np.diff(np.log(x+1))
        n2 = np.count_nonzero(n1)
        return n2

    def wc_change_stat(self, x):
        n1 = np.diff(np.log(x+1))
        n2 = np.mean(n1)
        return n2
                                                                                                
    def transform(self, X):
        print("Started Word Count Transformer.")
        w =  X.groupby(['id'])['word_count'].aggregate([self.wc_non_zero_change,
                                                             lambda x: np.log(len(x)), 
                                                             lambda x: np.log(np.max(x+1)), 
                                                             self.wc_change_stat])
        w.columns = ["wc_changing_nsteps", "wc_step_count", "wc_max", "wc_change_stat"]
        #output.columns = ["wc_step_count", "wc_max", "wc_change_stat"]
        w['Interaction'] = np.log(w.wc_changing_nsteps * w.wc_max + 1) 


        print("Done..")
        return sparse.csr_matrix(w.values)



# eda textchange transformer:
# added tc
class TextChangeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def hasChar(self, x,character:str):
        out = 0 
        for strings in x:
            if character in strings:
                out = 1
                break
        return out
                                                                                       
    def qCounter(self,c):
        h = " ".join(c)
        return np.log(len(re.findall(r" q ", h))+1)
                                                                                                                                    
    def transform(self, X):
        print("Starting TextChange Transformer.")
        output = X.groupby(['id'])['text_change'].aggregate([
            ("tc_1", lambda x: self.hasChar(x,character = "?")), 
            ("tc_2", lambda x: self.hasChar(x,character = "=>")), 
            ("tc_3", lambda x: self.hasChar(x,character = "(")), 
            ("tc_4", lambda x: self.hasChar(x,character = "\"")), 
            ("tc_5", lambda x: self.hasChar(x,character = "-")), 
            ("tc_6", lambda c: self.qCounter(c))]) 
        self.feature_names = output.columns.values
        self.index_ids = output.index.values
        print("Done..")
        return sparse.csr_matrix(output.values)
                                                                                                                                                                 

# Eda Text Change  Part 2
class TextChangeTransformer2(BaseEstimator, TransformerMixin):
    def __init__(self, max_word_length):
        self.max_word_length = max_word_length

    def fit(self, X, y = None):
        return self
                                        
    def text_change_distribution(self, v:str):
        distribution_container = []
        start_flag = 1
        word_count = 0
        size = 0
        for i in range(1,self.max_word_length + 2):
            s = "q{%s} " % i
            f = re.findall(s, v)
            if(start_flag == 1):
                word_count = len(f)
                start_flag = 0
            else:
                size = word_count - len(f)
                distribution_container.append(size)
                word_count = len(f)
                                                                                                                                                      
        return distribution_container

    def transform(self, X):
        print("Starting TextChange Transformer2.")
        X = X.groupby('id')['text_change'].aggregate(lambda r: self.text_change_distribution("".join(r)))
        X = np.log(pd.DataFrame(np.stack(X, axis = 0), index = X.index)+1)
        print("Done..")
        return sparse.csr_matrix(X.values)


class TextChangeTransformer3(BaseEstimator, TransformerMixin):

    def __init__(self, scores, max_wl = 12):
        self.max_wl = max_wl
        self.scores = scores

    def fit(self, X, y = None):
        #Get the distribution
        X = X.groupby('id')['text_change'].aggregate(lambda r: self.text_change_distribution("".join(r)))
        self.column_names = []
        for n in range(1, self.max_wl+1):
            self.column_names.append('q_len={}'.format(n))
        X2 = pd.DataFrame(np.stack(X, axis = 0), index = X.index, columns = self.column_names)
        
        X2 = pd.merge(X2,scores, left_index = True, right_index = True)
        self.X2 = X2
        # The pop standard dev::
        self.prop_std = self.X2.groupby("score").std(ddof = 0)
        # The mean
        self.expected_count = self.X2.groupby("score").mean()
        

        return self
        
        
    def text_change_distribution(self, v:str):
        distribution_container = []
        start_flag = 1
        word_count = 0
        size = 0
        for i in range(1,self.max_wl + 2):
            s = "q{%s} " % i
            f = re.findall(s, v)
            if(start_flag == 1):
                word_count = len(f)
                start_flag = 0
            else:
                size = word_count - len(f)
                distribution_container.append(size)
                word_count = len(f)
                
        return distribution_container 


    def transform(self, X):
        print("Started TextChangeTransformer 3 Feature Engineering")
        grouped_X = X.groupby('id')['text_change'].aggregate(lambda r: self.text_change_distribution("".join(r)))
        grouped_X = pd.DataFrame(np.stack(grouped_X, axis = 0), index = grouped_X.index, columns = self.column_names)

        predictions = np.zeros(shape = grouped_X.shape)

        # Iterate in columns:
        for j in range(self.max_wl):
            mean = self.expected_count.iloc[:,j].copy()
            sd = self.prop_std.iloc[:,j].copy()
            
            for i in range(grouped_X.shape[0]):
                x_value = grouped_X.iloc[i,j]
                z_scores = (x_value - mean)/sd
                z_scores = np.abs(z_scores)
                
                min_values = np.partition(z_scores, 1)[0:3] 
                idx_0 = np.where(z_scores == min_values[0])[0][0]
                idx_1 = np.where(z_scores == min_values[1])[0][0]
                idx_2 = np.where(z_scores == min_values[2])[0][0]
                # mean or sd index is ok:
                predictions[i,j] = np.mean([sd.index[idx_0], sd.index[idx_1], sd.index[idx_2]])

        output = pd.DataFrame(predictions, index = grouped_X.index, columns = self.column_names)
        output['tc_ave'] = np.apply_along_axis(func1d = np.mean, axis = 1, arr = output)

        print("Done")
        return sparse.csr_matrix(output.values)
        

# Feature Engineering Up Event Variable Transformer:
class UpEventTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self


    
    def find_clicked(self, x, st:str):
        has_string = 0
        for event in x:
            if(event == st):
                has_string = 1
                break
        return has_string

    def transform(self, X):
        print("Started Up Event Feature Engineering")
        
        output = X.groupby(['id'])['up_event'].aggregate([('ue_1',lambda x: self.find_clicked(x,"|")),
                                                          ('ue_2', lambda x: self.find_clicked(x,"Shift")),
                                                          ('ue_3', lambda x: self.find_clicked(x,"Tab")),
                                                          ])
        self.feature_names = output.columns.values
        self.index_ids = output.index.values
        print("Done..")
        return sparse.csr_matrix(output.values)



# In[7]:


# Eda action_time variable ransformer: (AT)

class ActionTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, scores):
        self.scores = scores
        self.score_values = np.arange(start = 0.5, stop = 6.5, step = 0.5)

    def fit(self, X, y = None):
        #Get the action time proportion or distribution per score:
        at_init = X.groupby('id')['action_time'].aggregate([
            ('one', lambda x: self.above_log_count(x, from_zero = 1)),
            ('two', lambda x: self.above_log_count(x, from_zero = 2)),
            ('three', lambda x: self.above_log_count(x, from_zero = 3)),
            ('four', lambda x: self.above_log_count(x, from_zero = 4)),
            ('five', lambda x: self.above_log_count(x, from_zero = 5)),
        ])
        
        at_init2 = pd.merge(at_init, self.scores, left_index = True, right_index = True)
        at2 = at_init2.groupby(by = 'score').sum()
        self.at_proportion= at2.apply(lambda x: x/(np.sum(at2, axis = 1)))
        return self
        
    def above_log_count(self, x, from_zero = 1):
        z = np.diff(np.log(x+1))
        z = np.abs(z)
        if from_zero < 5:
            count= len(list(filter(lambda q: (q>from_zero -1) and (q < from_zero), z)))
        else:
            count= len(list(filter(lambda q: q>=from_zero, z )))
        return count 
        
    def above_log_ratio(self, x, from_zero = 1):
        z = np.diff(np.log(x+1))
        z = np.abs(z)
        if from_zero < 3:
            count= len(list(filter(lambda q: (q>from_zero -1) and (q < from_zero), z)))
        else:
            count= len(list(filter(lambda q: q>=from_zero, z )))
        return np.log((count+1)/len(z)) 

        
    # Use chi-square to select the score of the given participant id   
    def compute_score_by_chisquare(self, fo:pd.Series, distribution):
        fo =fo + 1 # to remove errors for those with zero values
        total = np.sum(fo)
        # print(total)
        expected_arrays = distribution * total
        # print(expected_arrays)
        chi_stat = []
        for j in range(expected_arrays.shape[0]):
            results = st.chisquare(f_obs = fo, f_exp = expected_arrays.iloc[j])
            chi_stat.append(results[1])
    
        chi_stat = np.array(chi_stat)
        # get the maximum p-value (-1) or second to the max (-2), etc
        score_idx_1 = np.where(chi_stat == np.partition(chi_stat,-1)[-1])[0][0]
        score_idx_2 = np.where(chi_stat == np.partition(chi_stat,-2)[-2])[0][0]
        score_idx_3 = np.where(chi_stat == np.partition(chi_stat,-3)[-3])[0][0]
        score_idx_4 = np.where(chi_stat == np.partition(chi_stat,-4)[-4])[0][0]
        score_list = [
            self.score_values[score_idx_1],
            self.score_values[score_idx_3],
            self.score_values[score_idx_3],
            self.score_values[score_idx_4]]
        
        return np.mean(score_list)
        
    def transform(self, X):
        print("Started Action Time Feature Engineering")
        transform_1 = X.groupby("id")['action_time'].aggregate([
        ('at_1', lambda x: self.above_log_ratio(x, from_zero = 1)),
        ('at_2', lambda x: self.above_log_ratio(x, from_zero = 2)),
        ('at_3', lambda x: self.above_log_ratio(x, from_zero = 3))
        ])
        
        at_init = X.groupby('id')['action_time'].aggregate([
            ('one', lambda x: self.above_log_count(x, from_zero = 1)),
            ('two', lambda x: self.above_log_count(x, from_zero = 2)),
            ('three', lambda x: self.above_log_count(x, from_zero = 3)),
            ('four', lambda x: self.above_log_count(x, from_zero = 4)),
            ('five', lambda x: self.above_log_count(x, from_zero = 5)),
        ])
        transform_2 = at_init.apply(
            lambda x: self.compute_score_by_chisquare(x, distribution = self.at_proportion),axis = 1)
        transform_2.name = "at_chisq"
        output = pd.merge(transform_1, transform_2, left_index = True, right_index = True)
        self.feature_names = output.columns.values
        self.index_ids = output.index.values
        print("Done..")
        return sparse.csr_matrix(output.values)

        
        


# In[8]:


# Transformer for Activity, act:
class ActivityTransformer(BaseEstimator, TransformerMixin):
    oneHot: OneHotEncoder
    scores: pd.Series
    act_dist: pd.DataFrame
    feature_names: np.array
    initial_features: np.array
    
    def __init__(self, scores:pd.Series):
        self.oneHot = OneHotEncoder(handle_unknown = 'ignore', categories = 'auto', sparse_output = False)
        self.scores = scores
        self.score_values = np.arange(start = 0.5, stop = 6.5, step = 0.5)
        self.initial_features = np.array(['ac_Input', 'ac_Move', 'ac_NonPro', 'ac_Paste', 'ac_RemCut', 'ac_Replace'])
        
    def fit(self,X, y=None):
        #Transform X labels first:
        #Transform all with move into a Move:
        X.activity = X.activity.apply(lambda x: "Move" if ("Move" in x) else x)
        #Encode then get the distribution
        self.oneHot.fit(X)
        a1 = self.oneHot.fit_transform(X.activity.values.reshape(-1,1))
        a2 = pd.DataFrame(data=a1, columns=self.initial_features)
        a2['id'] = X.id.copy()
        
        act = a2.groupby(by = "id").sum()
        act = act + 1 # to avoid expected value of zero
        #added to include counts:
        self.act = act
        
        # Get the distribution for each kind of score
        # act distribution:
        act_dist = pd.merge(act, scores, left_index = True, right_index = True)
        act_dist = act_dist.groupby('score').sum()
        
        row_total = np.sum(act_dist, axis = 1)
        self.act_dist = act_dist.apply(lambda x: x / row_total)
            
        return self


    def compute_score_by_chisquare(self, fo:pd.Series, distribution):
        fo = fo+1
        total = np.sum(fo)
        # print(total)
        # add 1 to avoid expected value of zero.
        expected_arrays = distribution * total 
        # print(expected_arrays)
        chi_stat = []
        for j in range(expected_arrays.shape[0]):
            results = st.chisquare(f_obs = fo, f_exp = expected_arrays.iloc[j])
            chi_stat.append(results[1])
    
        chi_stat = np.array(chi_stat)
        # get the maximum p-value (-1) 
        score_idx_1 = np.where(chi_stat == np.partition(chi_stat,-1)[-1])[0][0]
        score_idx_2 = np.where(chi_stat == np.partition(chi_stat,-2)[-2])[0][0]
        score_idx_3 = np.where(chi_stat == np.partition(chi_stat,-3)[-3])[0][0]

        #get the mean:
        mu_1 = np.mean([self.score_values[score_idx_1], 
                        self.score_values[score_idx_2],
                        self.score_values[score_idx_3],
                        ]) 
        
        return mu_1 


    def transform(self, X):
        print("Started Activity Feature Engineering")
        #Transform X labels first:
        #Transform all with move into a Move:
        X.activity = X.activity.apply(lambda x: "Move" if ("Move" in x) else x)
        
        pre_output = self.oneHot.transform(X['activity'].values.reshape(-1,1))
        a2 = pd.DataFrame(data = pre_output, columns = self.initial_features)
        a2['id'] = X.id 
        act = a2.groupby('id').sum()
        output = act.apply(lambda z: self.compute_score_by_chisquare(z, self.act_dist), axis = 1)
        output.name = "act_chisq"
        self.feature_names = output.name
        print("Done..")
        print("act shape in transform: {}".format(act.shape))
        # The below will return the chi-square and the counts but the counts has no added benefit as of now.
        output1 = pd.merge(output, act, left_index = True, right_index = True)
        return sparse.csr_matrix(output1)
        


# Eda combination and activity and action time
# Based on the above, only the Input action time, nonpro, remove/cut and replace will be used:


class ComboActivityActionTime(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        print("Started Combo Activity Action Time Feature Engineering")
        X = X.copy()
        ids = X.id.unique()
        ids_length = ids.shape[0]
        
        X.activity = X.activity.apply(lambda x: "Move" if ("Move" in x) else x)
        
        input_cols = pd.Series(data = np.ones(shape = (ids_length,), dtype = np.int64), index = ids)
        input_cols.name = "Input"
        data_container = X.loc[X.activity == 'Input'][['action_time','id']].groupby('id').agg("sum")
        for t in data_container.index.values:
            if (data_container.loc[t]>0).item():
                input_cols.at[t] = data_container.loc[t].item()  


        nonproduction_cols = pd.Series(data = np.ones(shape = (ids_length,), dtype = np.int64), index = ids)
        nonproduction_cols.name = "Nonproduction"
        data_container = X.loc[X.activity == 'Nonproduction'][['action_time','id']].groupby('id').agg("sum")
        for t in data_container.index.values: 
            if (data_container.loc[t]>0).item():
                nonproduction_cols.at[t] = data_container.loc[t].item()  

        
        move_cols = pd.Series(data = np.ones(shape = (ids_length,), dtype = np.int64), index = ids)
        move_cols.name = "Move"
        data_container = X.loc[X.activity == 'Move'][['action_time','id']].groupby('id').agg("sum")
        for t in data_container.index.values:
            if (data_container.loc[t]>0).item():
                move_cols.at[t] = data_container.loc[t].item()  
        

        paste_cols = pd.Series(data = np.ones(shape = (ids_length,), dtype = np.int64), index = ids)
        paste_cols.name = "Paste"
        data_container= X.loc[X.activity == 'Paste'][['action_time','id']].groupby('id').agg('sum')
        for t in data_container.index.values: 
            if (data_container.loc[t]>0).item():
                paste_cols.at[t] = data_container.loc[t].item()  

        
        remove_cols = pd.Series(data = np.ones(shape = (ids_length,), dtype = np.int64), index = ids)
        remove_cols.name = "Remove/Cut"
        data_container = X.loc[X.activity == 'Remove/Cut'][['action_time','id']].groupby('id').agg('sum')
        for t in data_container.index.values: 
            if (data_container.loc[t]>0).item():
                remove_cols.at[t] = data_container.loc[t].item()  

        
        replace_cols = pd.Series(data = np.ones(shape = (ids_length,), dtype = np.int64), index = ids)
        replace_cols.name = "Replace"
        data_container= X.loc[X.activity == 'Replace'][['action_time','id']].groupby('id').agg('sum')
        for t in data_container.index.values: 
            if (data_container.loc[t]>0).item():
                replace_cols.loc[t] = data_container.loc[t].item()  

        n = pd.merge(input_cols, move_cols, left_index = True, right_index = True)
        n.columns = ['Input', 'Move']
        n = pd.merge(n,nonproduction_cols, left_index = True, right_index = True)
        n.columns = ['Input', 'Move', 'Nonproduction']
        n = pd.merge(n, paste_cols, left_index = True, right_index = True)
        n.columns = ['Input', 'Move', 'Nonproduction', 'Paste']
        n = pd.merge(n, remove_cols, left_index = True, right_index = True)
        n.columns = ['Input', 'Move','Nonproduction', 'Paste', 'Remove/Cut']
        n = pd.merge(n, replace_cols, left_index = True, right_index = True)
        n.columns = ['Input', 'Move','Nonproduction','Paste', 'Remove/Cut', 'Replace']
        n = np.log(n)
        print("Done..")
        return sparse.csr_matrix(n.values)
        



''' 
Eda for Up_event and Down_event comparing if a certain sample plays a music or
has a differnt up and down event 
'''
class EventComboTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X, y = None):
        return self

    def transform(self, X):
        feat1 = X.groupby('id')['down_event'].aggregate(lambda x: 1 if("MediaPlayPause" in " ".join(x)) else 0)
        X['diff_up_down'] = X.up_event == X.down_event
        feat2 = X.groupby('id')['diff_up_down'].aggregate(lambda x: 1 if(len(x) == np.sum(x)) else 0)
        output = pd.merge(feat1,feat2, left_index = True, right_index = True)
                                                   
        return sparse.csr_matrix(output.values)
                                                                                        
class EventCountVectorizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.count_vectorizer = CountVectorizer()
    
    def fit(self,X, y = None):
        feat1 = X.groupby('id')['down_event'].aggregate(lambda x: " ".join(x)) 
        self.count_vectorizer.fit(feat1)
        return self

    def transform(self, X):
        print("Starting Down Event Vectorizer.")
        feat1 = X.groupby('id')['down_event'].aggregate(lambda x: " ".join(x)) 
        output = self.count_vectorizer.transform(feat1)
        print("Done..")
        return output 


''' Vectorizer and Clustering combined '''                                                                                        
class EventCountVectorizerTransformer2(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.count_vectorizer = CountVectorizer()
    
    def fit(self,X, y = None):
        feat1 = X.groupby('id')['up_event'].aggregate(lambda x: " ".join(x)) 
        self.count_vectorizer.fit(feat1)
        return self

    def transform(self, X):
        print("Starting Up Event Vectorizer.")
        feat1 = X.groupby('id')['up_event'].aggregate(lambda x: " ".join(x)) 
        output = self.count_vectorizer.transform(feat1)
        print("Done..")
        return output 

# Eda TextChange 4
class TextChangeVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.kcluster =KMeans(12)

    def fit(self, X, y=None):
        feature = X.groupby('id')['text_change'].agg(lambda z: " ".join(z))
        pre_out = self.vectorizer.fit_transform(feature)
        self.kcluster.fit(pre_out)
        return self

    def transform(self, X):
        print("Starting Text Change Vectorizer")
        feature = X.groupby('id')['text_change'].agg(lambda z: " ".join(z))
        pre_out = self.vectorizer.transform(feature)
        output = self.kcluster.transform(pre_out)
        print("Done..")
        return sparse.csr_array(output)


class DenseTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray()


                                                                                        

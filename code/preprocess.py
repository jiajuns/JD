import pandas as pd
import numpy as np 


class preprocess(object):
    
    def __init__(self):
        self.sample = None
        self.train = None
        self.test = None
    
    '''READ DATA'''
    def read_data(self):
        self.sample = pd.read_csv('sample_data.csv').sort_values('user_id', ascending = True)
    
    '''Add DATE INDEX RANGE'''
    def date_index(self, cut_index=30):
        
        # extract date info
        self.sample['date'] = self.sample['time'].apply(lambda x: x.split(" ")[0])
        
        # filter data before 03-01
        unique_date = sorted(pd.unique(self.sample['date'])) # sort by date ascending
        effective_date = unique_date[cut_index:] # get 03-02 to 04-15
        self.sample = self.sample[self.sample['date'].isin(effective_date)]
        
        '''ASSIGN DATE INDEX'''
        
        # build date index dictionary
        total_days = len(effective_date)
        date_index_dict = {}
        idx = 0
        date_index = 0
        while idx < total_days:
            date_range = effective_date[idx:idx+5]
            date_index_dict[tuple(date_range)] = date_index
            idx += 5
            date_index += 1
        
        # assign date index to corresponding date
        def calculate_date_index(group):
            for key in date_index_dict:
                if group in key:
                    return date_index_dict[key]
        self.sample['date_index'] = self.sample['date'].apply(lambda x: calculate_date_index(x))
        
    '''CREATE X DATAFRAME & Y FOR EACH USER'''
    def create_X_Y(self):
        
        Xlist = []
        unique_user = pd.unique(self.sample['user_id'])
        for user in tqdm(unique_user):
            user_data = self.sample[self.sample['user_id'] == user]
            bought_items = pd.unique(user_data['sku_id'])
            
            # create frequency (use count here)
            bought_items_df = user_data[['sku_id', 'date']].groupby(['sku_id']).agg(['count'])['date']['count']
            bought_items_df = pd.DataFrame(bought_items_df) # (index: sku_id, column: count of items)
            
            # create type*time feature by action recently
            user_data['(type, date_index)'] = list(zip(user_data['type'], user_data['date_index']))
            type_time_df = user_data[['sku_id', '(type, date_index)']]\
                .groupby(['sku_id']).agg(lambda x: list(set(x)))
            
            # target variable 1 means bought 0 mean not
            Y_df = user_data[user_data['type']==4]['sku_id']
            
            # inner join two df by index
            merge_df = pd.concat([bought_items_df, type_time_df], axis = 1)
            merge_df = merge_df.reset_index() # sku_id becomes a column
            
            # add user_id column
            merge_df['user_id'] = user
            
            # rename columns
            merge_df.columns = ['sku_id', 'frequent_count', '(type, date_index)', 'user_id']
            merge_df['bought'] = merge_df['sku_id'].apply(lambda x: 1 if x in Y_df else 0)
            Xlist.append(merge_df)
        
        all_df = pd.concat(Xlist, axis = 0)[['user_id', 'sku_id', 'frequent_count', \
                                             '(type, date_index)', 'bought']]
        return all_df

    '''TRAIN & TEST SPLIT (INPUT DATE RANGE FOR TESTING)'''
    def train_test_split(self, date):
        pass

p = preprocess()
p.read_data()
p.date_index()
print 'finish date index'
df = p.create_X_Y()

# save data
df.to_csv('sample_processed_data.csv', index = False)
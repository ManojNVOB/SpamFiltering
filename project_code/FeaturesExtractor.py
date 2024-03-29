

import os
import numpy as np
    import pandas as pd
    from collections import Counter
    from itertools import islice

    # yield the next word in a line of the file
    def words_generator(fileobj):
        for line in fileobj:
            for word in line.split():
                yield word

    # function to generate the dictionaries for words total frequency and word frequencies in a file 
    def generate_freq_dictionaries(root_dir):
        word_count_dict={}
        file_dict = {}
        emails_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]    
        for emails_dir in emails_dirs:
            dirs = [os.path.join(emails_dir,f) for f in os.listdir(emails_dir)]
            for d in dirs:
                emails = [os.path.join(d,f) for f in os.listdir(d)]            
                for mail in emails:
                    file_name = mail.rsplit('/', 1)[-1]
                    file_dict[file_name] = {}
                    # if email is spam, assign spam label
                    if mail.split(".")[-2] == 'spam':
                        file_dict[file_name]["is_spam"] = 1 # spam
                    else:
                        file_dict[file_name]["is_spam"] = 0 # ham                    
                    f = open(mail,"r")
                    words = words_generator(f)
                    for word in words:
                        # if word is not an alpha numeric, don't include it
                        if word.isalpha()==False:
                            continue                       
                        if word not in word_count_dict:
                              word_count_dict[word] = 0
                        word_count_dict[word]+= 1                      
                        if word not in file_dict[file_name]:
                            file_dict[file_name][word] = 0
                        file_dict[file_name][word]+=1

        return word_count_dict,file_dict


    # find most frequent 3000 words list
    def find_most_frequent_words(word_count_dict):
        df = pd.DataFrame(word_count_dict.items(), columns=['word', 'total_freq'])
        df = df.sort(['total_freq'],ascending=False).head(n=3000)
        return df['word'].tolist(), df['total_freq'].tolist()


    # get frequency of a word in word_frequency dictionary
    def get_freq(word,freq_dict):
        if word in freq_dict:
            return freq_dict[word]
        else:
            return 0

    # divide the data into chunks to extract features
    def chunks_generator(data, SIZE):
        it = iter(data)
        for i in xrange(0, len(data), SIZE):
            yield {k:data[k] for k in islice(it, SIZE)}

    # extract features and store them in features.csv file
    def extract_features(file_dict,freq_words,file_name): 
        features_df = pd.DataFrame(file_dict.items(), columns=['file_name', 'freq_values'])
        features_df['is_spam'] = features_df['freq_values'].apply(lambda x: x.get('is_spam'))
        for word in freq_words:
                features_df[word] = features_df['freq_values'].apply(lambda freq_dict: get_freq(word,freq_dict))
        columns = list(features_df.columns.values)
        columns.remove('freq_values')
        if os.path.isfile(file_name):
            with open(fname, 'a') as f:
                features_df.to_csv(f, columns =columns, header=False)
        else:
            features_df.to_csv(file_name, columns = columns,header=columns, index=None,sep='\t')
        del features_df


    # email data directory
    email_data_dir = "/home/datascience/Desktop/SpamFiltering/enronEmailData"
    # features data file location
    file_name = "/home/datascience/Desktop/SpamFiltering/features.csv"

    word_count_dict,file_dict = generate_freq_dictionaries(email_data_dir)
    freq_words,frequencies = find_most_frequent_words(word_count_dict)
    chunks_list = []
    # create chunks of file dictionary data and extract features preocessing each chunk iteratively
    for item in chunks_generator(file_dict, 30):
        extract_features(item,freq_words,file_name)


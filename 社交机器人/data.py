import pandas as pd
list1 = [ 'followers', 'membership', 'retweeted', 'like', 'followed', 'quoted', 'discuss', 'replied', 'mentioned', 'contain']
for i in list1:
    print(i)
    df = pd.read_csv('edge.csv', iterator=True)
    new_df = pd.DataFrame()

    # new_df = pd.concat(,new_df)
    while 1:
        try:
            a = df.get_chunk(1000000)
            new_df = pd.concat([new_df, a[a['relation'] == i]])
        except:
            try:
                df.get_chunk(10000)
                new_df = pd.concat([new_df, a[a['relation'] == i]])
            except:
                try:
                    df.get_chunk(100)
                    new_df = pd.concat([new_df, a[a['relation'] == i]])
                except:
                    try:
                        df.get_chunk(1)
                        new_df = pd.concat([new_df, a[a['relation'] == i]])
                    except:
                        break
    new_df.to_csv(i + '.csv', index=None)


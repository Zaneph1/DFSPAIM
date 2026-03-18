import pandas as pd


# 读取CSV文件
df = pd.read_csv('data\\split\\2\\2.csv')

# split比例，例如70%和30%
split_ratio = [0.8, 0.2]

# shuffle data (if needed)
df = df.sample(frac=1).reset_index(drop=True)

# split data
split_points = [int(split_ratio[0] * len(df)), int(split_ratio[1] * len(df))]
print(split_points)
train_df = df[:split_points[0]]
print(train_df)
test_df = df[split_points[0]:len(df)]
print(test_df)


train_df.to_csv('data\\split\\2\\train.csv', index=False)
test_df.to_csv('data\\split\\2\\test.csv', index=False)
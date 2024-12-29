import pandas as pd  
  
# 读取CSV文件  
df = pd.read_csv('dataset/test.csv')  
  
# 计算要选择的行数  
num_rows = len(df)  
half_rows = num_rows // 2  
  
# 使用随机抽样选择50%的数据  
# 注意：这里使用了`sample`方法，它会随机选择数据，所以每次运行的结果可能不同  
# 如果你需要可复现的结果，可以给`sample`方法提供一个`random_state`参数  
sampled_df = df.sample(n=half_rows)  
  
# 保存结果到新的CSV文件  
sampled_df.to_csv('dataset/half_test.csv', index=False)
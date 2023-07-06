import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 获取苹果公司的股票数据
symbol = "AAPL"
data = yf.download(symbol, start='2001-01-01', end='2023-07-05')

# 提取特征和目标变量
X = data[['Open', 'High', 'Low', 'Volume']]  # 特征为开盘价、最高价、最低价和成交量
y = data['Close']  # 目标变量为收盘价

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型并进行训练
model = LinearRegression()
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
print("均方误差（Mean Squared Error）:", mse)

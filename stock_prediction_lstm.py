import os
import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib
from datetime import datetime, timedelta
import time
import random
import json

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像负号'-'显示为方块的问题
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 使用双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # 添加批量归一化 (注意这里的大小是hidden_size * 2因为是双向LSTM)
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # 优化全连接层结构
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # LSTM层
        out, _ = self.lstm(x, (h0, c0))
        
        # 只使用最后一个时间步的输出
        out = out[:, -1, :]
        
        # 批量归一化
        out = self.batch_norm(out)
        
        # 全连接层
        out = self.fc_layers(out)
        
        return out

def format_feature(df):
    """格式化特征数据"""
    required_features = [
        'Open', 'High', 'Low', 'Close', 'Volume'
    ]
    
    # 检查必要的基础特征是否存在
    for feature in required_features:
        if feature not in df.columns:
            raise ValueError(f"缺少基础特征: {feature}")
    
    # 确保技术指标特征存在，如果不存在则计算
    technical_features = [
        'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
        'VWAP', 'ATR', 'ROC'
    ]
    
    # 检查是否缺少技术指标并计算
    if not all(feature in df.columns for feature in technical_features):
        print("数据中缺少某些技术指标，将进行计算...")
        
        # 移动平均线
        if 'MA5' not in df.columns:
            df['MA5'] = df['Close'].rolling(window=5).mean()
        if 'MA10' not in df.columns:
            df['MA10'] = df['Close'].rolling(window=10).mean()
        if 'MA20' not in df.columns:
            df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # RSI指标
        if 'RSI' not in df.columns:
            delta = df['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-10)  # 避免除以零
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD指标
        if 'MACD' not in df.columns:
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
        
        # VWAP指标
        if 'VWAP' not in df.columns:
            df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum().replace(0, 1)
        
        # ATR指标
        if 'ATR' not in df.columns:
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift(1))
            low_close = abs(df['Low'] - df['Close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(window=14).mean()
        
        # ROC指标
        if 'ROC' not in df.columns:
            df['ROC'] = df['Close'].pct_change(periods=1) * 100
    
    # 填充缺失值
    for col in df.columns:
        if df[col].isnull().any():
            # 使用前向填充
            df[col] = df[col].fillna(method='ffill')
            # 然后使用后向填充（处理开头的NaN）
            df[col] = df[col].fillna(method='bfill')
            # 最后使用0填充任何剩余的NaN
            df[col] = df[col].fillna(0)
    
    # 选择最终特征
    selected_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MA5', 'MA10', 'MA20', 'RSI', 'MACD',
        'VWAP', 'ATR', 'ROC'
    ]
    
    # 确保所有列都存在
    for feature in selected_features:
        if feature not in df.columns:
            print(f"警告: 无法找到特征 {feature}，使用默认值0")
            df[feature] = 0
    
    return df[selected_features]

def create_sequences(data, seq_length):
    """创建时间序列数据"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 3]  # 使用Close价格作为目标
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def predict(save_dir, ticker_name, stock_data, stock_features, epochs=500, batch_size=32, learning_rate=0.001, seq_length=60):
    """训练LSTM模型并进行预测（优化版，支持混合精度训练）"""
    try:
        # 检查数据是否为空
        if stock_data.empty or stock_features.empty:
            raise ValueError(f"股票 {ticker_name} 的数据为空")
        
        if len(stock_features) < seq_length + 10:
            raise ValueError(f"股票 {ticker_name} 的数据样本太少 ({len(stock_features)}行), 需要至少 {seq_length + 10} 行数据")
        
        # 显示数据信息
        print(f"数据形状: {stock_features.shape}")
        print(f"包含特征: {', '.join(stock_features.columns)}")
        
        # 检查数据中是否有NaN值
        if stock_features.isna().any().any():
            print("警告: 数据中存在NaN值，将进行填充")
            # 使用前向填充策略
            stock_features = stock_features.fillna(method='ffill')
            # 如果还有NaN（如开头就有NaN），使用后向填充
            stock_features = stock_features.fillna(method='bfill')
            # 如果还有NaN（极少数情况），用0填充
            stock_features = stock_features.fillna(0)
        
        # 创建保存目录
        os.makedirs(f"{save_dir}/pic/predictions", exist_ok=True)
        os.makedirs(f"{save_dir}/pic/loss", exist_ok=True)
        os.makedirs(f"{save_dir}/models", exist_ok=True)
        
        # 数据预处理
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(stock_features)
        
        # 创建序列
        X, y = create_sequences(scaled_data, seq_length)
        
        # 再次检查序列数据是否足够
        if len(X) < 10:
            raise ValueError(f"处理后的序列数据太少 ({len(X)}个样本), 无法进行有效训练")
        
        # 划分训练集和测试集
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"训练集大小: {len(X_train)}个样本, 测试集大小: {len(X_test)}个样本")
        
        # 为RTX 3060笔记本GPU（6GB显存）优化批次大小
        optimal_batch_size = min(batch_size, 64)
        
        # 如果测试集太小，调整batch_size
        if len(X_test) < optimal_batch_size:
            optimal_batch_size = max(1, len(X_test))
            print(f"测试集太小，调整batch_size为: {optimal_batch_size}")
        
        # 创建数据加载器（使用单线程模式，避免Windows多进程问题）
        train_dataset = StockDataset(X_train, y_train)
        test_dataset = StockDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=optimal_batch_size, shuffle=True, 
                                 num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=optimal_batch_size, 
                               num_workers=0, pin_memory=False)
        
        # 初始化模型（优化隐藏层大小）
        model = LSTMModel(
            input_size=X_train.shape[2],
            hidden_size=96,  # 适合6GB显存的隐藏层大小
            num_layers=2,
            output_size=1,
            dropout=0.3
        ).to(device)
        
        # 使用较小的初始学习率，搭配学习率调度器
        initial_lr = learning_rate * 0.5
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-5)
        
        # 使用更简单的学习率调度器避免内存问题
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        
        # 混合精度训练
        scaler = GradScaler()
        
        # 训练过程
        train_losses = []
        val_losses = []
        
        # 设置梯度累积步数
        accumulation_steps = 2
        
        # 使用tqdm进度条显示训练进度
        print(f"开始训练LSTM模型，将进行{epochs}轮训练...")
        progress_bar = tqdm(range(epochs), desc="LSTM训练")
        
        for epoch in progress_bar:
            model.train()
            train_loss = 0
            
            for i, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # 清除梯度
                optimizer.zero_grad()
                
                # 使用混合精度前向传播
                with autocast():
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss = loss / accumulation_steps  # 缩放损失以进行梯度累积
                
                # 使用混合精度反向传播
                scaler.scale(loss).backward()
                
                # 累积梯度
                if (i + 1) % accumulation_steps == 0:
                    # 更新参数
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()  # 重置梯度
                
                train_loss += loss.item() * accumulation_steps
            
            # 验证
            model.eval()
            val_loss = 0
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    # 混合精度推理
                    with autocast():
                        outputs = model(batch_X)
                        val_loss += criterion(outputs.squeeze(), batch_y).item()
                    
                    predictions.extend(outputs.squeeze().cpu().numpy())
                    actuals.extend(batch_y.cpu().numpy())
            
            epoch_train_loss = train_loss / len(train_loader)
            epoch_val_loss = val_loss / len(test_loader)
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
            
            # 更新进度条描述
            progress_bar.set_postfix({
                'train_loss': f'{epoch_train_loss:.4f}', 
                'val_loss': f'{epoch_val_loss:.4f}'
            })
            
            if (epoch + 1) % 50 == 0:
                # 使用tqdm.write替代print，避免干扰进度条
                tqdm.write(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
                # 主动清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 训练完成
        print("LSTM模型训练完成!")
        
        # 保存模型
        torch.save(model.state_dict(), f"{save_dir}/models/{ticker_name}_model.pth")
        
        # 计算预测指标
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        accuracy = 1 - (rmse / np.mean(actuals))
        
        # 绘制预测结果
        plt.figure(figsize=(12, 6))
        plt.plot(actuals, label='实际值')
        plt.plot(predictions, label='预测值')
        plt.title(f'{ticker_name} 股票价格预测')
        plt.xlabel('时间')
        plt.ylabel('价格')
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()  # 自动调整布局，确保所有内容都能显示
        plt.savefig(f"{save_dir}/pic/predictions/{ticker_name}_prediction.png", dpi=300)
        plt.close()
        
        # 绘制损失曲线
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(val_losses, label='验证损失')
        plt.title(f'{ticker_name} 训练过程')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()  # 自动调整布局
        plt.savefig(f"{save_dir}/pic/loss/{ticker_name}_loss.png", dpi=300)
        plt.close()
        
        return {
            'accuracy': accuracy,
            'rmse': rmse,
            'mae': mae
        }
    
    except Exception as e:
        print(f"训练过程出错: {str(e)}")
        # 出错时主动清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise e

def calculate_technical_indicators(data, start_date=None, end_date=None):
    """
    计算股票的技术指标
    
    参数:
        data: DataFrame, 包含OHLCV数据的DataFrame
        start_date: str, 开始日期 (可选，用于相对表现计算)
        end_date: str, 结束日期 (可选，用于相对表现计算)
    
    返回:
        DataFrame: 添加了技术指标的数据
    """
    # 添加日期特征
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['Day'] = data.index.day
    
    # 移动平均线
    data['MA5'] = data['Close'].shift(1).rolling(window=5).mean()
    data['MA10'] = data['Close'].shift(1).rolling(window=10).mean()
    data['MA20'] = data['Close'].shift(1).rolling(window=20).mean()
    
    # RSI指标
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD指标
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['Signal_Line']
    
    # VWAP指标
    data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    # 布林带
    period = 20
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['Std_dev'] = data['Close'].rolling(window=period).std()
    data['Upper_band'] = data['SMA'] + 2 * data['Std_dev']
    data['Lower_band'] = data['SMA'] - 2 * data['Std_dev']
    
    # 相对大盘表现
    if start_date and end_date:
        benchmark_data = yf.download('SPY', start=start_date, end=end_date)['Close']
        data['Relative_Performance'] = (data['Close'] / benchmark_data.values) * 100
    
    # ROC指标
    data['ROC'] = data['Close'].pct_change(periods=1) * 100
    
    # ATR指标
    high_low_range = data['High'] - data['Low']
    high_close_range = abs(data['High'] - data['Close'].shift(1))
    low_close_range = abs(data['Low'] - data['Close'].shift(1))
    true_range = pd.concat([high_low_range, high_close_range, low_close_range], axis=1).max(axis=1)
    data['ATR'] = true_range.rolling(window=14).mean()
    
    # 前一天数据
    data[['Close_yes', 'Open_yes', 'High_yes', 'Low_yes']] = data[['Close', 'Open', 'High', 'Low']].shift(1)
    
    # 删除缺失值
    data = data.dropna()
    
    return data

def get_stock_data(ticker, start_date, end_date):
    """
    获取并处理单个股票的数据
    
    参数:
        ticker: 股票代码
        start_date: 起始日期
        end_date: 结束日期
    返回:
        处理后的股票数据DataFrame
    """
    # 下载股票数据
    # data = yf.download(ticker, start=start_date, end=end_date)  # 无代理
    data = yf.download(ticker, start=start_date, end=end_date, proxy="http://127.0.0.1:7890")  # 有代理
    
    # 计算技术指标
    data = calculate_technical_indicators(data, start_date, end_date)
    
    return data

def clean_csv_files(file_path):

    df = pd.read_csv(file_path)
            
    # 删除第二行和第三行
    df = df.drop([0, 1]).reset_index(drop=True)
            
    # 重命名列
    df = df.rename(columns={'Price': 'Date'})
            
    # 保存修改后的文件
    df.to_csv(file_path, index=False)
    print("所有文件处理完成！")

def main():
    """主函数：执行数据收集和处理流程"""
    # 股票分类列表
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',       # 科技
        'JPM', 'BAC', 'C', 'WFC', 'GS',                # 金融
        'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',            # 医药
        'XOM', 'CVX', 'COP', 'SLB', 'BKR',             # 能源
        'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',         # 消费
        'CAT', 'DE', 'MMM', 'GE', 'HON'                # 工业
    ]

    # 设置参数
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    NUM_FEATURES_TO_KEEP = 11
    
    # 创建数据文件夹
    data_folder = 'data'
    os.makedirs(data_folder, exist_ok=True)
    
    # 获取并保存所有股票数据
    print("开始下载和处理股票数据...")
    for ticker in tickers:
        try:
            print(f"处理 {ticker} 中...")
            stock_data = get_stock_data(ticker, START_DATE, END_DATE)
            stock_data.to_csv(f'{data_folder}/{ticker}.csv')
            clean_csv_files(f'{data_folder}/{ticker}.csv')
            print(f"{ticker} 处理完成")
        except Exception as e:
            print(f"处理 {ticker} 时出错: {str(e)}")

if __name__ == "__main__":
    main()
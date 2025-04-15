import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import warnings
from tqdm import tqdm
import matplotlib
# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像负号'-'显示为方块的问题
warnings.filterwarnings('ignore')

# 使用CUDA如果可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQNModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class Agent:
    def __init__(self, state_size, action_size=3, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        
        # 超参数
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        
        # 创建模型
        self.model = DQNModel(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    def act(self, state):
        # 探索
        if not self.is_eval and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        
        # 利用
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 随机采样
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).to(device)
            next_state_tensor = torch.FloatTensor(next_state).to(device)
            
            with torch.no_grad():
                target = reward
                if not done:
                    target += self.gamma * torch.max(self.model(next_state_tensor))
            
            # 前向传播获取当前Q值
            current_q = self.model(state_tensor)
            target_q = current_q.clone()
            target_q[0, action] = target
            
            # 计算损失并更新模型
            self.optimizer.zero_grad()
            loss = self.criterion(current_q, target_q)
            loss.backward()
            self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def sigmoid(x):
    """缩放价格变化的sigmoid函数"""
    return 1 / (1 + np.exp(-x))

def normalize_sentiment(sentiment_score):
    """将情感得分归一化到[0,1]区间"""
    # 假设情感得分范围为[-1,1]
    return (sentiment_score + 1) / 2

def get_state(data, t, window_size, sentiment_data=None):
    """
    生成状态表示，包含价格变化和情感数据
    
    参数:
        data: 价格数据
        t: 当前时间步
        window_size: 窗口大小
        sentiment_data: 情感分析数据，默认为None（字典形式，键为日期，值为情感特征）
    """
    if t < window_size:
        # 如果时间步小于窗口大小，填充前面的数据
        d = t - window_size + 1
        block = data[0:t+1]
        if d < 0:
            block = np.pad(block, (abs(d), 0), 'constant', constant_values=(block[0]))
    else:
        block = data[t-window_size+1:t+1]
    
    # 生成价格变化特征
    price_features = []
    for i in range(window_size - 1):
        price_features.append(sigmoid(block[i+1] - block[i]))
    
    # 如果有情感数据，添加到特征中
    if sentiment_data is not None and t in sentiment_data:
        sentiment_features = sentiment_data[t]
        # 确保情感特征是列表或数组
        if not isinstance(sentiment_features, (list, np.ndarray)):
            sentiment_features = [sentiment_features]
        # 归一化并添加情感特征
        normalized_sentiment = [normalize_sentiment(score) for score in sentiment_features]
        combined_features = price_features + normalized_sentiment
    else:
        # 如果没有情感数据，只使用价格特征
        combined_features = price_features
    
    return np.array([combined_features])

def process_stock(ticker, save_dir, window_size=10, initial_money=10000, iterations=500, use_sentiment=True):
    """
    处理单只股票的交易模拟
    
    参数:
        ticker: 股票代码
        save_dir: 结果保存目录
        window_size: 价格窗口大小
        initial_money: 初始资金
        iterations: 训练迭代次数
        use_sentiment: 是否使用情感数据
    """
    try:
        # 创建必要的目录
        os.makedirs(f"{save_dir}/models", exist_ok=True)
        os.makedirs(f"{save_dir}/pic/trades", exist_ok=True)
        os.makedirs(f"{save_dir}/pic/earnings", exist_ok=True)
        os.makedirs(f"{save_dir}/transactions", exist_ok=True)
        
        # 确保ticker不包含扩展名
        ticker = ticker.replace('.csv', '')
        
        # 读取价格数据
        file_path = f"data/{ticker}.csv"
        print(f"正在读取价格文件: {file_path}")
        stock_data = pd.read_csv(file_path)
        
        if stock_data.empty:
            raise ValueError(f"股票 {ticker} 的数据为空")
        
        # 获取收盘价
        close_column = None
        if 'Close' in stock_data.columns:
            close_column = 'Close'
        elif 'close' in stock_data.columns:
            close_column = 'close'
        else:
            raise ValueError(f"股票数据中缺少Close/close列，可用列: {stock_data.columns.tolist()}")

        # 标准化日期列以便合并情感数据
        date_column = None
        if 'Date' in stock_data.columns:
            date_column = 'Date'
        elif 'date' in stock_data.columns:
            date_column = 'date'
        else:
            raise ValueError(f"股票数据中缺少Date/date列，可用列: {stock_data.columns.tolist()}")
            
        # 确保日期列是datetime类型
        stock_data[date_column] = pd.to_datetime(stock_data[date_column])
            
        # 显示数据信息
        print(f"价格数据形状: {stock_data.shape}")
        print(f"包含列: {', '.join(stock_data.columns)}")
        
        close_prices = stock_data[close_column].values
        
        # 检查数据是否足够
        if len(close_prices) < window_size * 3:
            raise ValueError(f"数据样本太少，至少需要 {window_size * 3} 行数据")
        
        # 加载情感数据（如果使用）
        sentiment_data = None
        sentiment_features_count = 0
        
        if use_sentiment:
            sentiment_file = f"./data/sentiment/{ticker}_with_sentiment_signals.csv"
            if os.path.exists(sentiment_file):
                print(f"正在读取情感数据文件: {sentiment_file}")
                sentiment_df = pd.read_csv(sentiment_file)
                
                # 标准化日期列
                if 'Date' in sentiment_df.columns:
                    sentiment_df.rename(columns={'Date': 'date'}, inplace=True)
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                
                # 检查必要的情感列
                required_columns = ['date', 'sentiment_score']
                if all(col in sentiment_df.columns for col in required_columns):
                    # 合并情感数据
                    print("合并价格和情感数据...")
                    
                    # 创建包含所有特征的情感数据字典
                    sentiment_features = ['sentiment_score']
                    # 如果有其他情感指标，也可以加入
                    if 'sentiment_volatility' in sentiment_df.columns:
                        sentiment_features.append('sentiment_volatility')
                    if 'sentiment_momentum' in sentiment_df.columns:
                        sentiment_features.append('sentiment_momentum')
                        
                    sentiment_features_count = len(sentiment_features)
                    
                    # 创建日期到情感特征的映射
                    sentiment_data = {}
                    
                    # 创建一个索引到日期的映射，以便在训练和测试中使用
                    date_to_index = {i: date for i, date in enumerate(stock_data[date_column])}
                    index_to_sentiment = {}
                    
                    # 遍历价格数据的每一行，找到对应的情感数据
                    for i, row in stock_data.iterrows():
                        current_date = row[date_column]
                        # 查找这个日期的情感数据
                        sentiment_row = sentiment_df[sentiment_df['date'] == current_date]
                        
                        if not sentiment_row.empty:
                            # 提取所有情感特征
                            features = [float(sentiment_row[feat].values[0]) for feat in sentiment_features]
                            index_to_sentiment[i] = features
                    
                    sentiment_data = index_to_sentiment
                    print(f"成功加载情感数据，特征数量: {sentiment_features_count}")
                else:
                    print(f"情感数据缺少必要的列，将不使用情感特征。可用列: {sentiment_df.columns.tolist()}")
                    use_sentiment = False
            else:
                print(f"找不到情感数据文件: {sentiment_file}，将不使用情感特征。")
                use_sentiment = False
        
        # 划分训练集和测试集
        split = int(0.7 * len(close_prices))
        train_data = close_prices[:split]
        test_data = close_prices[split:]
        
        print(f"训练集: {len(train_data)} 样本, 测试集: {len(test_data)} 样本")
        
        # 初始化智能体，状态大小为价格窗口加情感特征数量
        state_size = (window_size - 1) + (sentiment_features_count if use_sentiment else 0)
        print(f"状态维度: {state_size} (价格特征: {window_size-1}, 情感特征: {sentiment_features_count if use_sentiment else 0})")
        
        agent = Agent(state_size=state_size)
        
        # 训练模式
        print(f"开始训练智能体，迭代 {iterations} 次...")
        for e in tqdm(range(iterations), desc="训练进度"):
            state = get_state(train_data, 0, window_size, 
                             sentiment_data if use_sentiment else None)
            total_profit = 0
            agent.inventory = []
            
            for t in range(1, len(train_data) - 1):
                action = agent.act(state)
                next_state = get_state(train_data, t, window_size,
                                      sentiment_data if use_sentiment else None)
                reward = 0

                # 买入
                if action == 1:
                    agent.inventory.append(train_data[t])
                
                # 卖出
                elif action == 2 and len(agent.inventory) > 0:
                    bought_price = agent.inventory.pop(0)
                    reward = max(0, train_data[t] - bought_price)
                    total_profit += train_data[t] - bought_price
                
                done = t == len(train_data) - 2
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                
                # 每隔100步或训练结束时才调用replay
                if t % 100 == 0 or done:
                    agent.replay()
                
                if done and e % 10 == 0:
                    tqdm.write(f"回合: {e+1}/{iterations}, 训练利润: {total_profit:.2f}")
        
        # 保存模型
        if use_sentiment:
            model_filename = f"{ticker}_dqn_with_sentiment.pth"
        else:
            model_filename = f"{ticker}_dqn.pth"
            
        torch.save(agent.model.state_dict(), f"{save_dir}/models/{model_filename}")
        
        # 测试模式
        agent.is_eval = True
        agent.epsilon = 0  # 不再探索
        
        print("开始测试交易策略...")
        state = get_state(test_data, 0, window_size, 
                         sentiment_data if use_sentiment else None)
        total_profit = 0
        agent.inventory = []
        
        history = []
        balance = initial_money
        buy_dates = []
        sell_dates = []
        buy_prices = []
        sell_prices = []
        transactions = []
        
        # 测试索引偏移量（训练集长度）
        test_offset = len(train_data)
        
        for t in tqdm(range(1, len(test_data) - 1), desc="测试进度"):
            action = agent.act(state)
            next_state = get_state(test_data, t, window_size,
                                  sentiment_data if use_sentiment else None)
            
            # 计算当前持仓价值
            holding_value = len(agent.inventory) * test_data[t] if agent.inventory else 0
            current_value = balance + holding_value
            history.append(current_value)
            
            # 获取真实索引（用于记录）
            real_index = t + test_offset
            
            # 买入
            if action == 1 and balance > test_data[t]:
                agent.inventory.append(test_data[t])
                balance -= test_data[t]
                buy_dates.append(t)
                buy_prices.append(test_data[t])
                
                transactions.append({
                    'day': real_index,
                    'operate': '买入',
                    'price': test_data[t],
                    'investment': test_data[t],
                    'total_balance': balance + holding_value,
                    'sentiment': sentiment_data.get(real_index, [0])[0] if use_sentiment and sentiment_data else 0
                })
                
            # 卖出
            elif action == 2 and len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                balance += test_data[t]
                total_profit += test_data[t] - bought_price
                sell_dates.append(t)
                sell_prices.append(test_data[t])
                
                transactions.append({
                    'day': real_index,
                    'operate': '卖出',
                    'price': test_data[t],
                    'investment': test_data[t],
                    'total_balance': balance + holding_value,
                    'sentiment': sentiment_data.get(real_index, [0])[0] if use_sentiment and sentiment_data else 0
                })
            
            state = next_state
        
        # 计算最终价值和回报率
        final_value = balance + sum([test_data[-1]] * len(agent.inventory))
        invest_return = ((final_value - initial_money) / initial_money) * 100
        
        print(f"初始资金: ${initial_money:.2f}")
        print(f"最终资金: ${final_value:.2f}")
        print(f"总收益: ${final_value - initial_money:.2f}")
        print(f"投资回报率: {invest_return:.2f}%")
        print(f"买入次数: {len(buy_dates)}")
        print(f"卖出次数: {len(sell_dates)}")
        
        # 保存交易记录
        transactions_df = pd.DataFrame(transactions)
        if not transactions_df.empty:
            model_type = "with_sentiment" if use_sentiment else "no_sentiment"
            transactions_df.to_csv(f"{save_dir}/transactions/{ticker}_transactions_{model_type}.csv", index=False)
        else:
            # 创建一个空的交易记录文件
            pd.DataFrame(columns=['day', 'operate', 'price', 'investment', 'total_balance', 'sentiment']
                        ).to_csv(f"{save_dir}/transactions/{ticker}_transactions.csv", index=False)
        
        # 生成交易图
        plt.figure(figsize=(15, 5))
        plt.plot(test_data, label='股票价格', color='black', alpha=0.5)
        
        if buy_dates:
            plt.scatter(buy_dates, [test_data[i] for i in buy_dates], marker='^', c='green', alpha=1, s=100, label='买入')
        if sell_dates:
            plt.scatter(sell_dates, [test_data[i] for i in sell_dates], marker='v', c='red', alpha=1, s=100, label='卖出')
        
        model_type = "使用情感分析" if use_sentiment else "不使用情感分析"
        plt.title(f"{ticker} 交易记录 ({model_type})")
        plt.xlabel('时间')
        plt.ylabel('价格')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)
        plt.tight_layout()  # 自动调整布局，确保所有内容都能显示
        
        graph_suffix = "with_sentiment" if use_sentiment else "no_sentiment"
        plt.savefig(f"{save_dir}/pic/trades/{ticker}_trades_{graph_suffix}.png", dpi=300)
        plt.close()
        
        # 生成收益图
        plt.figure(figsize=(15, 5))
        plt.plot(history, label='投资组合价值', color='blue')
        plt.axhline(y=initial_money, color='r', linestyle='-', label='初始投资')
        plt.title(f"{ticker} 累计收益 ({model_type})")
        plt.xlabel('时间')
        plt.ylabel('投资组合价值 ($)')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True)
        plt.tight_layout()  # 自动调整布局
        plt.savefig(f"{save_dir}/pic/earnings/{ticker}_cumulative_{graph_suffix}.png", dpi=300)
        plt.close()
        
        return {
            'ticker': ticker,
            'model_type': model_type,
            'total_gains': final_value - initial_money,
            'investment_return': invest_return,
            'trades_buy': len(buy_dates),
            'trades_sell': len(sell_dates)
        }
    
    except Exception as e:
        print(f"处理股票 {ticker} 出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 确保创建空文件以避免后续处理错误
        os.makedirs(f"{save_dir}/transactions", exist_ok=True)
        os.makedirs(f"{save_dir}/pic/trades", exist_ok=True)
        os.makedirs(f"{save_dir}/pic/earnings", exist_ok=True)
        
        # 创建空白图像 - 确保错误信息也能正确显示
        plt.figure(figsize=(15, 5))
        plt.title(f"{ticker} - 处理出错")
        plt.text(0.5, 0.5, f"错误: {str(e)}", horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/pic/trades/{ticker}_trades.png", dpi=300)
        plt.close()
        
        plt.figure(figsize=(15, 5))
        plt.title(f"{ticker} - 处理出错")
        plt.text(0.5, 0.5, f"错误: {str(e)}", horizontalalignment='center', verticalalignment='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/pic/earnings/{ticker}_cumulative.png", dpi=300)
        plt.close()
        
        # 创建空的交易记录
        pd.DataFrame(columns=['day', 'operate', 'price', 'investment', 'total_balance', 'sentiment']
                   ).to_csv(f"{save_dir}/transactions/{ticker}_transactions.csv", index=False)
        
        # 返回空结果
        return {
            'ticker': ticker,
            'model_type': 'error',
            'total_gains': 0,
            'investment_return': 0,
            'trades_buy': 0,
            'trades_sell': 0
        }

def main():
    """主函数：执行所有股票的交易策略，同时比较使用和不使用情感分析的结果"""
    import argparse
    
    parser = argparse.ArgumentParser(description='基于强化学习的股票交易智能体')
    parser.add_argument('--ticker', type=str, help='要处理的单个股票代码 (例如 AAPL)')
    parser.add_argument('--save_dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--window_size', type=int, default=10, help='历史价格窗口大小')
    parser.add_argument('--initial_money', type=int, default=10000, help='初始投资金额')
    parser.add_argument('--iterations', type=int, default=200, help='训练迭代次数')
    parser.add_argument('--no_sentiment', action='store_true', help='设置此项以不使用情感分析')
    parser.add_argument('--compare', action='store_true', help='同时运行使用和不使用情感分析的模型并比较结果')
    
    args = parser.parse_args()
    
    # 处理单只股票
    if args.ticker:
        tickers = [args.ticker]
    else:
        # 股票列表
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',       # 科技
            'JPM', 'BAC', 'C', 'WFC', 'GS',                # 金融
            'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',            # 医药
            'XOM', 'CVX', 'COP', 'SLB', 'BKR',             # 能源
            'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',         # 消费
            'CAT', 'DE', 'MMM', 'GE', 'HON'                # 工业
        ]
    
    save_dir = args.save_dir
    all_results = []
    
    # 处理每只股票
    for ticker in tickers:
        print(f"\n开始处理股票: {ticker}")
        
        if args.compare:
            # 使用情感分析运行模型
            print("使用情感分析运行模型...")
            with_sentiment_results = process_stock(
                ticker, save_dir, args.window_size, args.initial_money, args.iterations, use_sentiment=True
            )
            all_results.append(with_sentiment_results)
            
            # 不使用情感分析运行模型
            print("不使用情感分析运行模型...")
            no_sentiment_results = process_stock(
                ticker, save_dir, args.window_size, args.initial_money, args.iterations, use_sentiment=False
            )
            all_results.append(no_sentiment_results)
            
            # 比较结果
            print("\n---- 对比结果 ----")
            print(f"{'模型类型':<20} {'总收益':<15} {'回报率(%)':<15} {'买入次数':<10} {'卖出次数':<10}")
            print(f"{'使用情感分析':<20} {with_sentiment_results['total_gains']:<15.2f} {with_sentiment_results['investment_return']:<15.2f} {with_sentiment_results['trades_buy']:<10} {with_sentiment_results['trades_sell']:<10}")
            print(f"{'不使用情感分析':<20} {no_sentiment_results['total_gains']:<15.2f} {no_sentiment_results['investment_return']:<15.2f} {no_sentiment_results['trades_buy']:<10} {no_sentiment_results['trades_sell']:<10}")
            
        else:
            # 根据参数决定是否使用情感分析
            use_sentiment = not args.no_sentiment
            result = process_stock(
                ticker, save_dir, args.window_size, args.initial_money, args.iterations, use_sentiment=use_sentiment
            )
            all_results.append(result)
    
    # 生成汇总报告
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df.to_csv(f"{save_dir}/summary_results.csv", index=False)
        print(f"\n结果汇总已保存到 {save_dir}/summary_results.csv")
    
    print("\n所有股票处理完成!")

if __name__ == "__main__":
    main()
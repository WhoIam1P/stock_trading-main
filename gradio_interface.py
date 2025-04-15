import gradio as gr
import pandas as pd
import torch
import os
from PIL import Image, ImageDraw, ImageFont
import warnings
import yfinance as yf
from stock_prediction_lstm import predict, format_feature
from RLagent import process_stock
from datetime import datetime
from process_stock_data import get_stock_data, clean_csv_files
from stock_sentiment_analyzer import StockSentimentAnalyzer
from process_stock_sentiment import StockNewsProcessor

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'tmp/gradio'

# 创建所有必要的目录
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs('tmp/gradio/pic', exist_ok=True)
os.makedirs('tmp/gradio/pic/predictions', exist_ok=True)
os.makedirs('tmp/gradio/pic/loss', exist_ok=True)
os.makedirs('tmp/gradio/pic/earnings', exist_ok=True)
os.makedirs('tmp/gradio/pic/trades', exist_ok=True)
os.makedirs('tmp/gradio/models', exist_ok=True)
os.makedirs('tmp/gradio/transactions', exist_ok=True)
os.makedirs('tmp/gradio/ticker', exist_ok=True)
os.makedirs('tmp/gradio/sentiment', exist_ok=True)  # 情感分析结果目录
os.makedirs('tmp/gradio/news', exist_ok=True)  # 新闻数据目录

def get_data(ticker, start_date, end_date, progress=gr.Progress()):
    data_folder = 'tmp/gradio/ticker'
    temp_path = f'{data_folder}/{ticker}.csv'
    try:        
        # 获取并保存所有股票数据
        progress(0, desc="开始获取股票数据...")
        stock_data = get_stock_data(ticker, start_date, end_date)
        progress(0.4, desc="计算技术指标...")
        stock_data.to_csv(temp_path)
        progress(0.7, desc="处理数据格式...")
        clean_csv_files(temp_path)
        progress(1.0, desc="数据获取完成")
        return temp_path, "数据获取成功"
    except Exception as e:
        return None, f"获取数据出错: {str(e)}"

def validate_and_fix_data(file_path):
    """验证并修复股票数据文件，确保其符合预期格式"""
    try:
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 检查必要的列是否存在
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"警告: 文件 {file_path} 缺少必要的列: {', '.join(missing_cols)}")
            # 如果缺少必要列，创建默认数据
            dates = pd.date_range(start='2020-01-01', periods=len(df) if not df.empty else 100)
            default_df = pd.DataFrame({
                'Date': dates,
                'Open': [100] * len(dates),
                'High': [110] * len(dates),
                'Low': [90] * len(dates),
                'Close': [105] * len(dates),
                'Volume': [1000000] * len(dates)
            })
            default_df.to_csv(file_path, index=False)
            print(f"已为 {file_path} 创建默认数据")
            return False
        
        # 检查是否有NaN值
        if df[required_cols].isna().any().any():
            print(f"警告: 文件 {file_path} 包含NaN值，将进行填充")
            # 填充NaN值
            for col in required_cols:
                if df[col].isna().any():
                    # 使用前向填充
                    df[col] = df[col].fillna(method='ffill')
                    # 然后使用后向填充（处理开头的NaN）
                    df[col] = df[col].fillna(method='bfill')
                    # 最后使用列平均值填充任何剩余的NaN
                    df[col] = df[col].fillna(df[col].mean() if not df[col].empty else 0)
            
            # 保存修复后的数据
            df.to_csv(file_path, index=False)
            print(f"已修复 {file_path} 中的NaN值")
        
        # 检查数据长度是否足够
        if len(df) < 100:  # 假设至少需要100行数据
            print(f"警告: 文件 {file_path} 的数据量不足 ({len(df)} 行)")
            # 可以选择通过复制已有数据来扩充数据集
            
        return True
    
    except Exception as e:
        print(f"验证和修复文件 {file_path} 时出错: {str(e)}")
        # 创建默认数据
        dates = pd.date_range(start='2020-01-01', periods=100)
        default_df = pd.DataFrame({
            'Date': dates,
            'Open': [100] * 100,
            'High': [110] * 100,
            'Low': [90] * 100,
            'Close': [105] * 100,
            'Volume': [1000000] * 100
        })
        default_df.to_csv(file_path, index=False)
        print(f"已为 {file_path} 创建默认数据")
        return False

# 新增情感分析处理函数
def analyze_sentiment(temp_csv_path, uploaded_news_file=None, use_uploaded_news=False, 
                      positive_threshold=0.2, negative_threshold=-0.2, 
                      use_custom_thresholds=False, save_dir=SAVE_DIR, backtest=True, 
                      news_file_path=None, progress=gr.Progress()):
    """
    执行情感分析并生成交易信号
    
    参数:
        temp_csv_path: 股票价格数据文件路径
        uploaded_news_file: 用户上传的自定义新闻数据文件路径
        use_uploaded_news: 是否使用用户上传的新闻数据
        positive_threshold: 正面情感阈值（高于此值触发买入信号）
        negative_threshold: 负面情感阈值（低于此值触发卖出信号）
        use_custom_thresholds: 是否使用自定义阈值（否则使用默认设置）
        save_dir: 保存结果的目录
        backtest: 是否执行回测
        news_file_path: 通过获取新闻数据按钮生成的新闻文件路径
        progress: Gradio进度条对象
    """
    if not temp_csv_path:
        return [None] * 8, "请先获取股票价格数据"

    try:
        # 从文件路径中提取股票代码（去掉.csv后缀）
        ticker = os.path.basename(temp_csv_path).split('.')[0]
        
        progress(0.1, desc="初始化情感分析器...")
        
        # 确保sentiment目录存在
        os.makedirs(f"{save_dir}/sentiment", exist_ok=True)
        os.makedirs(f"{save_dir}/pic", exist_ok=True)
        
        # 初始化情感分析器
        analyzer = StockSentimentAnalyzer(data_dir=f"{save_dir}/sentiment")
        
        # 读取价格数据
        progress(0.2, desc="读取股票价格数据...")
        price_data = pd.read_csv(temp_csv_path)
        
        # 标准化列名
        if 'Date' in price_data.columns:
            price_data.rename(columns={'Date': 'date'}, inplace=True)
        if 'Close' in price_data.columns:
            price_data.rename(columns={'Close': 'close'}, inplace=True)
            
        price_data['date'] = pd.to_datetime(price_data['date'])
        
        # 确定使用哪个新闻文件
        news_file = None
        if use_uploaded_news and uploaded_news_file:
            news_file = uploaded_news_file
            progress(0.25, desc="使用上传的自定义新闻数据...")
        elif news_file_path:
            news_file = news_file_path
            progress(0.25, desc="使用自动获取的新闻数据...")
        else:
            # 尝试在不同可能的位置查找新闻数据
            possible_paths = [
                f"./data/{ticker}_news.csv",
                f"./data/news/{ticker}_news.csv",
                f"{save_dir}/news/{ticker}_news.csv",
                f"{save_dir}/sentiment/{ticker}_news.csv"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    news_file = path
                    progress(0.25, desc=f"找到现有新闻数据: {os.path.basename(path)}...")
                    break
            
            if not news_file:
                return [None] * 8, f"找不到{ticker}的新闻数据文件，请先获取新闻数据"
                
        progress(0.3, desc="分析新闻情感...")
        # 分析新闻情感
        news_with_sentiment = analyzer.analyze_news_file(news_file, ticker)
        
        if news_with_sentiment.empty:
            return [None] * 8, f"无法从{news_file}分析情感"
            
        progress(0.4, desc="聚合每日情感...")
        # 聚合每日情感
        daily_sentiment = analyzer.aggregate_daily_sentiment(news_with_sentiment)
        
        # 保存初步的每日情感数据
        daily_sentiment_filename = f"{ticker}_daily_sentiment.csv"
        analyzer.save_sentiment_data(daily_sentiment, ticker, file_name=daily_sentiment_filename)
        
        progress(0.5, desc="合并情感与价格数据...")
        # 合并情感和价格数据
        merged_data = analyzer.merge_with_price_data(daily_sentiment, price_data)
        
        progress(0.6, desc="生成可视化...")
        # 生成基础可视化
        analyzer.plot_sentiment_vs_price(merged_data, ticker, output_dir=f"{save_dir}/pic")
        analyzer.visualize_sentiment_metrics(merged_data, ticker, output_dir=f"{save_dir}/pic")
        
        # 分析情感效应
        effect_results = analyzer.analyze_sentiment_effect(merged_data)
        
        progress(0.7, desc="生成交易信号...")
        # 生成交易信号（使用自定义阈值或默认阈值）
        if use_custom_thresholds:
            data_with_signals = analyzer.generate_trading_signals(
                merged_data, 
                positive_threshold=positive_threshold,
                negative_threshold=negative_threshold
            )
        else:
            data_with_signals = analyzer.generate_trading_signals(merged_data)
        
        # 绘制交易信号
        analyzer.plot_trading_signals(data_with_signals, ticker, output_dir=f"{save_dir}/pic")
        
        # 保存带有信号的数据
        merged_signals_filename = f"{ticker}_with_sentiment_signals.csv"
        analyzer.save_sentiment_data(data_with_signals, ticker, file_name=merged_signals_filename)
        
        # 回测交易策略（如果启用）
        backtest_results = {}
        if backtest:
            progress(0.8, desc="执行交易策略回测...")
            backtest_results = analyzer.backtest_trading_strategy(data_with_signals)
            
            if backtest_results and 'summary' in backtest_results:
                analyzer.plot_backtest_results(backtest_results, ticker, output_dir=f"{save_dir}/pic")
                
                # 保存回测摘要
                backtest_summary_df = pd.DataFrame([backtest_results['summary']])
                backtest_summary_df.to_csv(f"{save_dir}/sentiment/{ticker}_backtest_summary.csv", index=False)
                
                # 保存交易记录
                if backtest_results.get('positions'):
                    trades_df = pd.DataFrame(backtest_results['positions'])
                    trades_df.to_csv(f"{save_dir}/sentiment/{ticker}_trades.csv", index=False)
        
        progress(0.9, desc="加载图像结果...")
        # 加载图像结果
        images = []
        try:
            sentiment_price_path = f"{save_dir}/pic/{ticker}_sentiment_vs_price.png"
            if os.path.exists(sentiment_price_path):
                sentiment_price_img = Image.open(sentiment_price_path)
                images.append(sentiment_price_img)
            else:
                blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
                images.append(blank_img)
                
            sentiment_metrics_path = f"{save_dir}/pic/{ticker}_sentiment_metrics.png"
            if os.path.exists(sentiment_metrics_path):
                sentiment_metrics_img = Image.open(sentiment_metrics_path)
                images.append(sentiment_metrics_img)
            else:
                blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
                images.append(blank_img)
                
            trading_signals_path = f"{save_dir}/pic/{ticker}_trading_signals.png"
            if os.path.exists(trading_signals_path):
                trading_signals_img = Image.open(trading_signals_path)
                images.append(trading_signals_img)
            else:
                blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
                images.append(blank_img)
                
            if backtest and backtest_results:
                backtest_path = f"{save_dir}/pic/{ticker}_backtest_results.png"
                if os.path.exists(backtest_path):
                    backtest_img = Image.open(backtest_path)
                    images.append(backtest_img)
                else:
                    blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
                    images.append(blank_img)
            else:
                blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
                images.append(blank_img)
        except Exception as e:
            print(f"加载图像出错: {str(e)}")
            blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
            images = [blank_img, blank_img, blank_img, blank_img]
        
        # 获取交易记录
        trades_df = pd.DataFrame()
        try:
            trades_path = f"{save_dir}/sentiment/{ticker}_trades.csv"
            if os.path.exists(trades_path):
                trades_df = pd.read_csv(trades_path)
        except Exception as e:
            print(f"加载交易记录出错: {str(e)}")
        
        # 准备情感分析摘要
        sentiment_summary = {
            'average_sentiment': merged_data['sentiment_score'].mean() if not merged_data.empty else 0,
            'positive_days': len(merged_data[merged_data['sentiment_score'] > 0]) if not merged_data.empty else 0,
            'negative_days': len(merged_data[merged_data['sentiment_score'] < 0]) if not merged_data.empty else 0,
            'neutral_days': len(merged_data[merged_data['sentiment_score'] == 0]) if not merged_data.empty else 0,
            'correlation_with_price': effect_results.get('correlation', 0) if effect_results else 0,
            'avg_price_change_after_positive': effect_results.get('avg_change_after_positive', 0) if effect_results else 0,
            'avg_price_change_after_negative': effect_results.get('avg_change_after_negative', 0) if effect_results else 0
        }
        
        # 准备回测结果摘要
        backtest_summary = {}
        if backtest_results and 'summary' in backtest_results:
            backtest_summary = {
                'total_return': backtest_results['summary'].get('total_return', 0),
                'annualized_return': backtest_results['summary'].get('annualized_return', 0),
                'total_trades': backtest_results['summary'].get('total_trades', 0),
                'win_rate': backtest_results['summary'].get('win_rate', 0),
                'profit_factor': backtest_results['summary'].get('profit_factor', 0),
                'max_drawdown': backtest_results['summary'].get('max_drawdown', 0)
            }
        
        progress(1.0, desc="情感分析完成!")
        
        return [
            images,
            sentiment_summary['average_sentiment'],
            sentiment_summary['positive_days'],
            sentiment_summary['negative_days'],
            sentiment_summary['correlation_with_price'],
            backtest_summary.get('total_return', 0),
            backtest_summary.get('win_rate', 0),
            trades_df,
            "情感分析完成"
        ]
    
    except Exception as e:
        print(f"情感分析处理出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
        draw = ImageDraw.Draw(blank_img)
        draw.text((50, 240), f"处理出错: {str(e)}", fill=(0, 0, 0))
        
        return [
            [blank_img, blank_img, blank_img, blank_img],
            0, 0, 0, 0, 0, 0,
            pd.DataFrame(),
            f"情感分析出错: {str(e)}"
        ]

def process_and_predict(temp_csv_path, epochs, batch_size, learning_rate, 
                       window_size, initial_money, agent_iterations, save_dir, progress=gr.Progress()):
    if not temp_csv_path:
        return [None] * 9  # 返回空结果
        
    try:
        # 从文件路径中提取股票代码（去掉.csv后缀）
        ticker = os.path.basename(temp_csv_path).split('.')[0]
        
        progress(0.05, desc="正在加载并验证股票数据...")
        # 验证并修复数据
        data_valid = validate_and_fix_data(temp_csv_path)
        if not data_valid:
            progress(0.1, desc="使用替代数据...")
        
        # 读取数据
        stock_data = pd.read_csv(temp_csv_path)
        
        # 确保处理特征前检查数据结构
        try:
            stock_features = format_feature(stock_data)
        except Exception as e:
            print(f"格式化特征时出错: {str(e)}")
            # 创建一个具有默认格式的提示
            return [None] * 9
        
        progress(0.1, desc="开始LSTM预测训练...")
        # 使用纯股票代码而非文件名
        try:
            metrics = predict(
                save_dir=save_dir,
                ticker_name=ticker,
                stock_data=stock_data,
                stock_features=stock_features,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
        except Exception as e:
            print(f"LSTM预测训练出错: {str(e)}")
            metrics = {'accuracy': 0, 'rmse': 0, 'mae': 0}
        
        progress(0.5, desc="开始交易代理训练...")
        # 使用纯股票代码而非文件名
        try:
            trading_results = process_stock(
                ticker,
                save_dir,
                window_size=window_size,
                initial_money=initial_money,
                iterations=agent_iterations,
                use_sentiment=True  # 尝试使用情感分析数据
            )
        except Exception as e:
            print(f"交易代理训练出错: {str(e)}")
            trading_results = {'total_gains': 0, 'investment_return': 0, 'trades_buy': 0, 'trades_sell': 0}
        
        progress(0.9, desc="生成结果可视化...")
        # 使用安全的图像加载方式
        images = []
        try:
            prediction_path = f"{save_dir}/pic/predictions/{ticker}_prediction.png"
            if os.path.exists(prediction_path):
                prediction_plot = Image.open(prediction_path)
                images.append(prediction_plot)
            else:
                print(f"无法找到预测图片: {prediction_path}")
                # 创建一个空白图像作为替代
                blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
                images.append(blank_img)
        except Exception as e:
            print(f"加载预测图片出错: {str(e)}")
            blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
            images.append(blank_img)
        
        try:
            loss_path = f"{save_dir}/pic/loss/{ticker}_loss.png"
            if os.path.exists(loss_path):
                loss_plot = Image.open(loss_path)
                images.append(loss_plot)
            else:
                print(f"无法找到损失图片: {loss_path}")
                blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
                images.append(blank_img)
        except Exception as e:
            print(f"加载损失图片出错: {str(e)}")
            blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
            images.append(blank_img)
            
        try:
            earnings_path = f"{save_dir}/pic/earnings/{ticker}_cumulative.png"
            if os.path.exists(earnings_path):
                earnings_plot = Image.open(earnings_path)
                images.append(earnings_plot)
            else:
                print(f"无法找到收益图片: {earnings_path}")
                blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
                images.append(blank_img)
        except Exception as e:
            print(f"加载收益图片出错: {str(e)}")
            blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
            images.append(blank_img)
            
        try:
            trades_path = f"{save_dir}/pic/trades/{ticker}_trades.png"
            if os.path.exists(trades_path):
                trades_plot = Image.open(trades_path)
                images.append(trades_plot)
            else:
                print(f"无法找到交易图片: {trades_path}")
                blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
                images.append(blank_img)
        except Exception as e:
            print(f"加载交易图片出错: {str(e)}")
            blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
            images.append(blank_img)
        
        try:
            transactions_path = f"{save_dir}/transactions/{ticker}_transactions.csv"
            if os.path.exists(transactions_path):
                transactions_df = pd.read_csv(transactions_path)
            else:
                print(f"无法找到交易记录: {transactions_path}")
                # 创建空的交易记录
                transactions_df = pd.DataFrame(columns=['day', 'operate', 'price', 'investment', 'total_balance'])
        except Exception as e:
            print(f"加载交易记录出错: {str(e)}")
            transactions_df = pd.DataFrame(columns=['day', 'operate', 'price', 'investment', 'total_balance'])
        
        progress(1.0, desc="完成!")
        return [
            images,
            metrics['accuracy'] * 100 if 'accuracy' in metrics else 0,
            metrics['rmse'] if 'rmse' in metrics else 0,
            metrics['mae'] if 'mae' in metrics else 0,
            trading_results['total_gains'] if 'total_gains' in trading_results else 0,
            trading_results['investment_return'] if 'investment_return' in trading_results else 0,
            trading_results['trades_buy'] if 'trades_buy' in trading_results else 0,
            trading_results['trades_sell'] if 'trades_sell' in trading_results else 0,
            transactions_df
        ]
    except Exception as e:
        print(f"处理错误: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细错误堆栈
        
        # 返回空结果但带有有意义的错误信息
        blank_img = Image.new('RGB', (640, 480), color=(255, 255, 255))
        # 在空白图像上添加错误信息
        draw = ImageDraw.Draw(blank_img)
        draw.text((50, 240), f"处理出错: {str(e)}", fill=(0, 0, 0))
        
        return [
            [blank_img, blank_img, blank_img, blank_img],
            0, 0, 0, 0, 0, 0, 0,
            pd.DataFrame(columns=['day', 'operate', 'price', 'investment', 'total_balance'])
        ]

def get_news_data(ticker, start_date, end_date, max_articles=500, progress=gr.Progress()):
    """
    获取指定股票的新闻数据
    
    参数:
        ticker: 股票代码
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        max_articles: 最大获取文章数量
        progress: Gradio进度条对象
    
    返回:
        新闻数据文件路径, 状态信息
    """
    news_folder = 'tmp/gradio/news'
    os.makedirs(news_folder, exist_ok=True)
    
    try:
        # 初始化新闻处理器
        progress(0.1, desc="初始化新闻处理器...")
        processor = StockNewsProcessor(output_dir=news_folder)
        
        # 计算日期范围天数
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        date_range_days = (end_date_dt - start_date_dt).days
        
        # 调整max_articles
        adjusted_max = max(max_articles, min(date_range_days * 2, 1000))
        progress(0.2, desc=f"开始获取{ticker}的新闻数据 (最多{adjusted_max}篇)...")
        
        # 获取新闻数据
        news_df = processor.process_stock_news(
            ticker,
            start_date=start_date,
            end_date=end_date,
            max_articles=adjusted_max
        )
        
        if news_df.empty:
            return None, f"未找到{ticker}在指定日期范围内的新闻数据"
        
        progress(0.7, desc="预处理新闻数据...")
        # 预处理数据
        from process_stock_sentiment import preprocess_news_data
        news_df = preprocess_news_data(news_df)
        
        # 保存文件
        progress(0.9, desc="保存新闻数据...")
        file_name = f"{ticker}_news_from_{start_date}_to_{end_date.replace('-', '')}.csv"
        file_path = os.path.join(news_folder, file_name)
        news_df.to_csv(file_path, index=False)
        
        progress(1.0, desc="新闻数据获取完成")
        return file_path, f"成功获取{len(news_df)}条{ticker}的新闻数据"
        
    except Exception as e:
        progress(1.0, desc="获取新闻数据出错")
        import traceback
        traceback.print_exc()
        return None, f"获取新闻数据出错: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# 智能股票预测与交易Agent")
    
    save_dir_state = gr.State(value='tmp/gradio')
    temp_csv_state = gr.State(value=None)
    news_file_state = gr.State(value=None)
    
    with gr.Row():
        with gr.Column(scale=2):
            ticker_input = gr.Textbox(label="股票代码 (例如: AAPL)")
        with gr.Column(scale=2):
            start_date = gr.Textbox(
                label="开始日期 (YYYY-MM-DD)", 
                value=(datetime.now().replace(year=datetime.now().year-4).strftime('%Y-%m-%d'))
            )
        with gr.Column(scale=2):
            end_date = gr.Textbox(
                label="结束日期 (YYYY-MM-DD)", 
                value=datetime.now().strftime('%Y-%m-%d')
            )
        with gr.Column(scale=1):
            fetch_button = gr.Button("获取价格数据")
    
    with gr.Row():
        status_output = gr.Textbox(label="状态信息", interactive=False)
    
    with gr.Row():
        data_file = gr.File(label="下载股票价格数据", visible=True, interactive=False)
        
    with gr.Row():
        max_articles = gr.Slider(minimum=100, maximum=1000, value=500, step=50, 
                               label="最大获取新闻文章数量")
        fetch_news_button = gr.Button("获取新闻数据", interactive=False)
        
    with gr.Row():
        news_status_output = gr.Textbox(label="新闻数据状态", interactive=False)
        
    with gr.Row():
        news_file = gr.File(label="下载股票新闻数据", visible=True, interactive=False)
    
    with gr.Tabs() as tabs:
        with gr.TabItem("LSTM预测参数"):
            with gr.Column():
                lstm_epochs = gr.Slider(minimum=100, maximum=1000, value=500, step=10, 
                                      label="LSTM训练轮数")
                lstm_batch = gr.Slider(minimum=16, maximum=128, value=32, step=16, 
                                     label="LSTM批次大小")
                learning_rate = gr.Slider(minimum=0.0001, maximum=0.01, value=0.001, 
                                        step=0.0001, label="LSTM训练学习率")
        
        with gr.TabItem("交易代理参数"):
            with gr.Column():
                window_size = gr.Slider(minimum=10, maximum=100, value=30, step=5,
                                      label="时间窗口大小")
                initial_money = gr.Number(value=10000, label="初始投资金额 ($)")
                agent_iterations = gr.Slider(minimum=100, maximum=1000, value=500, 
                                          step=50, label="代理训练迭代次数")
        
        # 新增情感分析参数选项卡
        with gr.TabItem("情感分析参数"):
            with gr.Column():
                uploaded_news_file = gr.File(label="上传自定义新闻数据文件（可选）", visible=True)
                use_uploaded_news = gr.Checkbox(label="使用上传的自定义新闻数据", value=False)
                positive_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                                            label="正面情感阈值（触发买入信号）")
                negative_threshold = gr.Slider(minimum=-1.0, maximum=0.0, value=-0.2, step=0.05,
                                            label="负面情感阈值（触发卖出信号）")
                use_custom_thresholds = gr.Checkbox(label="使用自定义阈值", value=False)
                run_backtest = gr.Checkbox(label="执行回测分析", value=True)
    
    with gr.Row():
        train_button = gr.Button("开始LSTM与交易代理训练", interactive=False)
        sentiment_button = gr.Button("执行情感分析", interactive=False)
    
    # 创建结果显示区域的标签页
    with gr.Tabs() as result_tabs:
        # LSTM及交易代理结果标签页
        with gr.TabItem("预测与交易结果"):
            with gr.Row():
                output_gallery = gr.Gallery(label="预测与交易结果可视化", show_label=True,
                                          elem_id="gallery", columns=4, rows=1,
                                          height="auto", object_fit="contain")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 预测指标")
                    accuracy_output = gr.Number(label="预测准确率 (%)")
                    rmse_output = gr.Number(label="RMSE (均方根误差)")
                    mae_output = gr.Number(label="MAE (平均绝对误差)")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 交易指标")
                    gains_output = gr.Number(label="总收益 ($)")
                    return_output = gr.Number(label="投资回报率 (%)")
                    trades_buy_output = gr.Number(label="买入次数")
                    trades_sell_output = gr.Number(label="卖出次数")
            
            with gr.Row():
                gr.Markdown("### 交易记录")
                transactions_df = gr.DataFrame(
                    headers=["day", "operate", "price", "investment", "total_balance"],
                    label="交易详细记录"
                )
        
        # 情感分析结果标签页
        with gr.TabItem("情感分析结果"):
            with gr.Row():
                sentiment_gallery = gr.Gallery(label="情感分析可视化", show_label=True,
                                             elem_id="sentiment_gallery", columns=4, rows=1,
                                             height="auto", object_fit="contain")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 情感指标")
                    avg_sentiment = gr.Number(label="平均情感得分")
                    positive_days = gr.Number(label="正面情感天数")
                    negative_days = gr.Number(label="负面情感天数")
                    sentiment_correlation = gr.Number(label="情感与价格相关性")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 回测指标")
                    backtest_return = gr.Number(label="回测总回报率 (%)")
                    backtest_win_rate = gr.Number(label="胜率 (%)")
            
            with gr.Row():
                gr.Markdown("### 情感交易记录")
                sentiment_trades_df = gr.DataFrame(
                    headers=["date", "action", "price", "shares", "portfolio_value"],
                    label="情感交易详细记录"
                )
                
            with gr.Row():
                sentiment_status = gr.Textbox(label="情感分析状态", interactive=False)
    
    def update_interface(csv_path):
        return (
            csv_path if csv_path else None,  # 更新文件下载
            gr.update(interactive=bool(csv_path)),  # 更新训练按钮
            gr.update(interactive=bool(csv_path))   # 更新新闻获取按钮
        )
    
    # 获取数据按钮事件
    fetch_result = fetch_button.click(
        fn=get_data,
        inputs=[ticker_input, start_date, end_date],
        outputs=[temp_csv_state, status_output]
    )
    
    # 更新界面状态
    fetch_result.then(
        update_interface,
        inputs=[temp_csv_state],
        outputs=[data_file, train_button, fetch_news_button]
    )
    
    # 获取新闻数据按钮事件
    fetch_news_result = fetch_news_button.click(
        fn=get_news_data,
        inputs=[ticker_input, start_date, end_date, max_articles],
        outputs=[news_file_state, news_status_output]
    )
    
    # 更新新闻文件下载状态和情感分析按钮
    def update_news_interface(news_path):
        return (
            news_path if news_path else None,  # 更新新闻文件下载
            gr.update(interactive=bool(news_path))  # 更新情感分析按钮
        )
    
    fetch_news_result.then(
        update_news_interface,
        inputs=[news_file_state],
        outputs=[news_file, sentiment_button]
    )
    
    # 训练按钮事件
    train_button.click(
        fn=process_and_predict,
        inputs=[
            temp_csv_state,
            lstm_epochs,
            lstm_batch,
            learning_rate,
            window_size,
            initial_money,
            agent_iterations,
            save_dir_state
        ],
        outputs=[
            output_gallery,
            accuracy_output,
            rmse_output,
            mae_output,
            gains_output,
            return_output,
            trades_buy_output,
            trades_sell_output,
            transactions_df
        ]
    )
    
    # 情感分析按钮事件
    sentiment_button.click(
        fn=analyze_sentiment,
        inputs=[
            temp_csv_state,
            uploaded_news_file,
            use_uploaded_news,
            positive_threshold,
            negative_threshold,
            use_custom_thresholds,
            save_dir_state,
            run_backtest,
            news_file_state  # 添加新闻文件路径
        ],
        outputs=[
            sentiment_gallery,
            avg_sentiment,
            positive_days,
            negative_days,
            sentiment_correlation,
            backtest_return,
            backtest_win_rate,
            sentiment_trades_df,
            sentiment_status
        ]
    )

demo.launch(server_port=7860, share=True)
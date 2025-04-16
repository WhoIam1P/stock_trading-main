#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用yahoo_fin库获取股票新闻数据，并保存到指定目录。
这个模块可以独立运行，也可以作为库被导入到其他脚本中。
功能与process_stock_data.py相似，专注于情感数据获取和处理。
"""

import os
import pandas as pd
import numpy as np
import datetime
import time
import logging
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from yahoo_fin import stock_info as si
from yahoo_fin.news import get_yf_rss

# 设置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockNewsProcessor')

class StockNewsProcessor:
    """获取股票新闻数据并处理保存的类"""
    
    def __init__(self, output_dir='./data/news'):
        """
        初始化新闻处理器。
        
        参数:
            output_dir: 新闻数据保存目录
        """
        self.output_dir = output_dir
        
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
    
    def get_stock_news(self, ticker, days_back=1500, max_articles=500):
        """
        获取指定股票的新闻数据。
        
        参数:
            ticker: 股票代码
            days_back: 获取多少天前的新闻 (默认1500天，约4年，覆盖2020年至今)
            max_articles: 最大获取文章数 (默认500，增加获取量)
            
        返回:
            包含新闻数据的DataFrame
        """
        logger.info(f"开始获取 {ticker} 的新闻数据（最多{days_back}天前，最多{max_articles}篇文章）...")
        
        try:
            # 使用yahoo_fin获取RSS新闻源
            news_data = get_yf_rss(ticker)
            
            # 如果没有获取到新闻
            if not news_data or len(news_data) == 0:
                logger.warning(f"未找到 {ticker} 的新闻数据")
                return pd.DataFrame()
                
            # 转换为DataFrame
            news_df = pd.DataFrame(news_data)
            
            # 重命名列以匹配我们的格式
            news_df = news_df.rename(columns={
                'title': 'title',
                'summary': 'summary',
                'link': 'url',
                'published': 'date'
            })
            
            # 确保日期列为datetime类型
            news_df['date'] = pd.to_datetime(news_df['date'])
            
            # 将 Pandas 日期时间列转换为 timezone-naive UTC
            # 这样可以与 timezone-naive 的 start_date 进行比较
            if pd.api.types.is_datetime64_any_dtype(news_df['date']) and news_df['date'].dt.tz is not None:
                news_df['date'] = news_df['date'].dt.tz_localize(None)
            
            # 计算起始日期 (timezone-naive 本地时间，或者可以改为 UTC)
            start_date = datetime.datetime.now() - datetime.timedelta(days=days_back)
            
            # 筛选日期范围内的新闻 (现在两者都是 timezone-naive)
            news_df = news_df[news_df['date'] >= start_date]
            
            # 限制文章数
            if len(news_df) > max_articles:
                news_df = news_df.head(max_articles)
                
            # 添加股票代码列
            news_df['ticker'] = ticker
            
            # 添加来源列
            news_df['source'] = 'Yahoo Finance'
            
            # 组合标题和摘要作为文本分析内容
            news_df['text'] = news_df['title'] + ' ' + news_df['summary']
            
            logger.info(f"成功获取 {len(news_df)} 条 {ticker} 的新闻")
            return news_df
            
        except Exception as e:
            logger.error(f"获取 {ticker} 新闻时出错: {str(e)}")
            return pd.DataFrame()
    
    def calculate_sentiment_features(self, news_df):
        """
        计算新闻数据的情感特征（待情感分析模型实现）
        
        参数:
            news_df: 包含新闻数据的DataFrame
            
        返回:
            添加了情感特征的DataFrame
        """
        # 这里只是一个占位函数，实际情感分析由stock_sentiment_analyzer.py实现
        # 添加一些基本字段以保持一致性
        result_df = news_df.copy()
        
        # 添加日期特征（类似process_stock_data.py）
        if 'date' in result_df.columns:
            result_df['Year'] = result_df['date'].dt.year
            result_df['Month'] = result_df['date'].dt.month
            result_df['Day'] = result_df['date'].dt.day
            result_df['Weekday'] = result_df['date'].dt.weekday
        
        # 添加文本长度特征
        if 'text' in result_df.columns:
            result_df['text_length'] = result_df['text'].apply(lambda x: len(str(x)))
            result_df['word_count'] = result_df['text'].apply(lambda x: len(str(x).split()))
        
        # 添加标题长度特征
        if 'title' in result_df.columns:
            result_df['title_length'] = result_df['title'].apply(lambda x: len(str(x)))
            result_df['title_word_count'] = result_df['title'].apply(lambda x: len(str(x).split()))
        
        return result_df
    
    def get_multi_stock_news(self, tickers, days_back=30, max_articles_per_ticker=50):
        """
        并行获取多支股票的新闻数据。
        
        参数:
            tickers: 股票代码列表
            days_back: 获取多少天前的新闻
            max_articles_per_ticker: 每个股票最大获取文章数
            
        返回:
            包含多支股票新闻数据的DataFrame
        """
        all_news = []
        
        # 使用线程池并行获取新闻
        with ThreadPoolExecutor(max_workers=min(10, len(tickers))) as executor:
            future_to_ticker = {
                executor.submit(self.get_stock_news, ticker, days_back, max_articles_per_ticker): ticker 
                for ticker in tickers
            }
            
            for future in future_to_ticker:
                ticker = future_to_ticker[future]
                try:
                    news_df = future.result()
                    if not news_df.empty:
                        all_news.append(news_df)
                except Exception as e:
                    logger.error(f"处理 {ticker} 新闻时出错: {str(e)}")
        
        # 如果没有任何新闻数据
        if not all_news:
            logger.warning(f"未找到任何股票的新闻数据")
            return pd.DataFrame()
            
        # 合并所有新闻数据
        combined_news = pd.concat(all_news, ignore_index=True)
        
        # 按日期排序（最新的在前）
        combined_news = combined_news.sort_values('date', ascending=False)
        
        return combined_news
    
    def save_news_data(self, news_df, filename=None):
        """
        保存新闻数据到CSV文件。
        
        参数:
            news_df: 包含新闻数据的DataFrame
            filename: 文件名（如果为None，则根据内容自动生成）
            
        返回:
            保存的文件路径
        """
        if news_df.empty:
            logger.warning("没有新闻数据可保存")
            return None
            
        # 如果没有提供文件名，自动生成
        if not filename:
            # 检查是否多个股票
            if 'ticker' in news_df.columns:
                tickers = news_df['ticker'].unique()
                if len(tickers) == 1:
                    ticker_str = tickers[0]
                else:
                    ticker_str = "multi_stocks"
            else:
                ticker_str = "unknown"
                
            # 获取当前日期作为文件名一部分
            date_str = datetime.datetime.now().strftime("%Y%m%d")
            filename = f"{ticker_str}_news_{date_str}.csv"
            
        # 构建完整的文件路径
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            # 保存到CSV
            news_df.to_csv(file_path, index=False)
            logger.info(f"新闻数据已保存到 {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"保存新闻数据时出错: {str(e)}")
            return None
    
    def process_stock_news(self, ticker, start_date=None, end_date=None, max_articles=500):
        """
        处理单个股票的新闻数据。
        
        参数:
            ticker: 股票代码
            start_date: 开始日期（如果为None，则默认为2020-01-01）
            end_date: 结束日期（如果为None，则默认为当前日期）
            max_articles: 最大获取文章数
            
        返回:
            处理后的新闻数据DataFrame
        """
        # 确定日期范围
        if end_date is None:
            end_date = datetime.datetime.now()
        else:
            if isinstance(end_date, str):
                end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_date is None:
            start_date = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
        else:
            if isinstance(start_date, str):
                start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        
        # 计算时间范围
        days_back = (end_date - start_date).days
        logger.info(f"获取 {ticker} 的新闻数据，时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')} ({days_back}天)")
        
        # 获取新闻数据
        news_df = self.get_stock_news(ticker, days_back, max_articles)
        
        # 确保日期范围筛选
        if not news_df.empty:
            # 将日期转换为datetime以便比较
            if not pd.api.types.is_datetime64_any_dtype(news_df['date']):
                news_df['date'] = pd.to_datetime(news_df['date'])
            
            # 确保在指定的日期范围内
            news_df = news_df[(news_df['date'] >= start_date) & (news_df['date'] <= end_date)]
            logger.info(f"筛选日期范围内的新闻，保留 {len(news_df)} 条")
            
            # 计算情感特征
            if not news_df.empty:
                news_df = self.calculate_sentiment_features(news_df)
        
        return news_df

def preprocess_news_data(news_df):
    """
    对新闻数据进行预处理，以便于情感分析。
    
    参数:
        news_df: 包含新闻数据的DataFrame
        
    返回:
        预处理后的DataFrame
    """
    # 确保DataFrame不为空
    if news_df.empty:
        return news_df
    
    # 确保日期列为datetime类型
    if 'date' in news_df.columns and not pd.api.types.is_datetime64_any_dtype(news_df['date']):
        news_df['date'] = pd.to_datetime(news_df['date'])
    
    # 删除重复的新闻（基于URL或标题）
    if 'url' in news_df.columns:
        news_df = news_df.drop_duplicates(subset=['url'])
    elif 'title' in news_df.columns:
        news_df = news_df.drop_duplicates(subset=['title'])
    
    # 确保文本字段存在
    if 'text' not in news_df.columns:
        if 'title' in news_df.columns and 'summary' in news_df.columns:
            news_df['text'] = news_df['title'] + ' ' + news_df['summary'].fillna('')
        elif 'title' in news_df.columns:
            news_df['text'] = news_df['title']
        elif 'summary' in news_df.columns:
            news_df['text'] = news_df['summary']
    
    # 删除文本为空的行
    if 'text' in news_df.columns:
        news_df = news_df[news_df['text'].notna() & (news_df['text'] != '')]
    
    # 按日期排序
    if 'date' in news_df.columns:
        news_df = news_df.sort_values('date', ascending=False)
    
    return news_df

def main():
    """主函数：执行数据收集和处理流程"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='获取并处理股票新闻数据')
    parser.add_argument('--ticker', type=str, help='单个股票代码')
    parser.add_argument('--tickers', type=str, help='多个股票代码，用逗号分隔')
    parser.add_argument('--file', type=str, help='包含股票代码的文件（每行一个代码）')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='开始日期 (YYYY-MM-DD), 默认2020-01-01')
    parser.add_argument('--end_date', type=str, default=None, help='结束日期 (YYYY-MM-DD), 默认当前日期')
    parser.add_argument('--output_dir', type=str, default='./data/news', help='输出目录')
    parser.add_argument('--max_articles', type=int, default=500, help='每个股票最大获取的文章数')
    parser.add_argument('--predefined', action='store_true', help='使用预定义的股票组合')
    
    args = parser.parse_args()
    
    # 创建新闻处理器
    output_dir = args.output_dir
    processor = StockNewsProcessor(output_dir=output_dir)
    
    # 设置日期参数
    start_date = args.start_date
    # 确保设置了合理的默认开始日期(2020-01-01)
    if not start_date:
        start_date = '2020-01-01'
        
    if args.end_date:
        end_date = args.end_date
    else:
        end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
    # 计算日期间隔天数
    start_date_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    date_range_days = (end_date_dt - start_date_dt).days
    
    # 动态调整max_articles，确保获取足够多的文章
    max_articles = max(args.max_articles, min(date_range_days * 2, 1000))
    logger.info(f"设置最大获取文章数为 {max_articles}，日期范围为 {date_range_days} 天")
    
    # 确定要处理的股票代码列表
    tickers = []
    
    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    elif args.file and os.path.exists(args.file):
        with open(args.file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    elif args.predefined or not (args.ticker or args.tickers or args.file):
        # 默认使用预定义的股票列表，与process_stock_data.py相同
        # 如果没有指定任何股票代码选项，也使用这个默认列表
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',       # 科技
            'JPM', 'BAC', 'C', 'WFC', 'GS',                # 金融
            'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',            # 医药
            'XOM', 'CVX', 'COP', 'SLB', 'BKR',             # 能源
            'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',         # 消费
            'CAT', 'DE', 'MMM', 'GE', 'HON'                # 工业
        ]
        logger.info("使用默认预定义的股票列表")
    
    # 如果没有指定任何股票，显示帮助信息（此处已修改逻辑，现在会使用默认列表）
    if not tickers:
        parser.print_help()
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"将保存新闻数据到目录: {output_dir}")
    
    # 处理每个股票的新闻数据
    logger.info(f"开始获取和处理 {len(tickers)} 支股票的新闻数据...")
    logger.info(f"日期范围: {start_date} 至 {end_date} (约 {date_range_days} 天)")
    
    for ticker in tickers:
        try:
            logger.info(f"处理 {ticker} 中...")
            # 获取并处理新闻数据
            news_df = processor.process_stock_news(
                ticker, 
                start_date=start_date, 
                end_date=end_date, 
                max_articles=max_articles
            )
            
            # 预处理数据
            news_df = preprocess_news_data(news_df)
            
            # 保存数据
            if not news_df.empty:
                file_path = processor.save_news_data(news_df, f"{ticker}_news_from_{start_date}_to_{end_date.replace('-', '')}.csv")
                logger.info(f"已保存 {ticker} 的新闻数据到 {file_path}，获取到 {len(news_df)} 条新闻")
            else:
                logger.warning(f"没有找到 {ticker} 的新闻数据")
                
        except Exception as e:
            logger.error(f"处理 {ticker} 新闻数据时出错: {str(e)}")
    
    logger.info("所有股票新闻数据处理完成")

if __name__ == "__main__":
    main() 
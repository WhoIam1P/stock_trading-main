#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用VADER模型进行股票情感分析。
VADER (Valence Aware Dictionary and sEntiment Reasoner) 是一个基于词典和规则的情感分析工具，
特别适合社交媒体和金融新闻文本的情感分析。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import re
import json
import requests
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor
import glob # Import glob for file searching

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StockSentimentAnalyzer')

# 确保VADER词典已下载
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logger.info("下载VADER词典...")
    nltk.download('vader_lexicon')

class StockSentimentAnalyzer:
    """股票情感分析类，使用VADER模型分析金融新闻和社交媒体评论的情感。"""
    
    def __init__(self, data_dir='./data/sentiment'):
        """初始化情感分析器。"""
        self.sia = SentimentIntensityAnalyzer()
        self.data_dir = data_dir
        
        # 创建数据目录（如果不存在）
        os.makedirs(data_dir, exist_ok=True)
        
        # 增强VADER词典，添加金融领域特定词汇
        self._enhance_vader_lexicon()
        
    def _enhance_vader_lexicon(self):
        """增强VADER词典，添加金融领域特定词汇。"""
        # 股票市场常用词汇及其情感得分
        financial_lexicon = {
            'beat': 3.0,              # 超出预期
            'miss': -3.0,             # 未达预期
            'upgrade': 2.5,           # 上调评级
            'downgrade': -2.5,        # 下调评级
            'bullish': 2.0,           # 看涨
            'bearish': -2.0,          # 看跌
            'outperform': 1.5,        # 表现优于
            'underperform': -1.5,     # 表现不及
            'rally': 1.8,             # 反弹
            'crash': -2.2,            # 崩盘
            'profit': 1.6,            # 利润
            'loss': -1.6,             # 亏损
            'growth': 1.5,            # 增长
            'decline': -1.5,          # 下降
            'surge': 1.7,             # 激增
            'plunge': -2.1,           # 暴跌
            'soar': 1.8,              # 飙升
            'tumble': -1.7,           # 暴跌
            'jump': 1.4,              # 跳涨
            'sink': -1.5,             # 下沉
            'bankruptcy': -3.0,       # 破产
            'investigation': -1.8,    # 调查
            'scandal': -2.0,          # 丑闻
            'sue': -1.7,              # 起诉
            'layoff': -1.6,           # 裁员
            'acquisition': 0.5,       # 收购（中性略偏正面）
            'merger': 0.5,            # 合并（中性略偏正面）
            'dividend': 1.2,          # 分红
            'lawsuit': -1.5,          # 诉讼
            'default': -2.5,          # 违约
            'sanctions': -1.8,        # 制裁
            'fine': -1.5,             # 罚款
            'settlement': 0.3,        # 和解（中性略偏正面）
            
            # 新增基本面词汇
            'revenue': 1.2,           # 营收
            'sales': 1.0,             # 销售额
            'earnings': 1.5,          # 盈利
            'eps': 1.5,               # 每股收益
            'guidance': 0.5,          # 业绩指引
            'outlook': 0.5,           # 前景展望
            'forecast': 0.5,          # 预测
            'strongest': 2.5,         # 最强劲
            'weakest': -2.5,          # 最疲软
            'robust': 2.0,            # 强劲
            'sluggish': -2.0,         # 疲软
            'exceeds': 2.3,           # 超出
            'falls short': -2.3,      # 不及
            'record high': 2.5,       # 创新高
            'record low': -2.5,       # 创新低
            'better than expected': 2.3, # 好于预期
            'worse than expected': -2.3, # 差于预期
            'surprise': 1.0,          # 惊喜（财报）
            
            # 新增市场情绪词汇
            'confidence': 1.8,        # 信心
            'fear': -1.8,             # 恐惧
            'optimism': 1.5,          # 乐观
            'pessimism': -1.5,        # 悲观
            'enthusiasm': 1.7,        # 热情
            'anxiety': -1.7,          # 焦虑
            'euphoria': 2.0,          # 狂热
            'panic': -2.5,            # 恐慌
            'greed': -1.0,            # 贪婪
            'doubt': -1.2,            # 怀疑
            'uncertainty': -1.5,      # 不确定性
            'volatile': -1.0,         # 波动
            'stability': 1.0,         # 稳定
            'momentum': 1.0,          # 动能
            
            # 新增技术面词汇
            'overbought': -1.0,       # 超买
            'oversold': 1.0,          # 超卖
            'breakout': 1.5,          # 突破
            'breakdown': -1.5,        # 跌破
            'resistance': -0.5,       # 阻力
            'support': 0.5,           # 支撑
            'rebound': 1.5,           # 反弹
            'correction': -1.2,       # 调整
            'consolidation': 0.0,     # 盘整（中性）
            'uptrend': 1.5,           # 上升趋势
            'downtrend': -1.5,        # 下降趋势
            'reversal': 0.0,          # 反转（中性）
            'volume spike': 0.5,      # 成交量激增
            
            # 新增宏观经济词汇
            'inflation': -1.0,        # 通胀
            'deflation': -1.5,        # 通缩
            'recession': -2.5,        # 衰退
            'recovery': 2.0,          # 复苏
            'expansion': 1.5,         # 扩张
            'contraction': -1.5,      # 收缩
            'gdp growth': 1.5,        # GDP增长
            'gdp decline': -1.5,      # GDP下降
            'unemployment': -1.5,     # 失业率
            'stimulus': 1.0,          # 刺激措施
            'rate hike': -1.0,        # 加息
            'rate cut': 1.0,          # 降息
            'trade war': -2.0,        # 贸易战
            'economic crisis': -2.5,  # 经济危机
            
            # 新增中文金融词汇
            '涨停': 2.5,               # 涨停板
            '跌停': -2.5,              # 跌停板
            '牛市': 2.0,               # 牛市
            '熊市': -2.0,              # 熊市
            '利好': 1.8,               # 利好消息
            '利空': -1.8,              # 利空消息
            '增长': 1.5,               # 增长
            '下滑': -1.5,              # 下滑
            '盈利': 1.7,               # 盈利
            '亏损': -1.7,              # 亏损
            '高于预期': 2.0,            # 高于预期
            '低于预期': -2.0,           # 低于预期
            '业绩预警': -1.8,           # 业绩预警
            '扩张': 1.6,               # 扩张
            '收缩': -1.6,              # 收缩
            '重组': 0.5,               # 重组
            '并购': 0.7,               # 并购
            '破产': -3.0,              # 破产
            '违约': -2.5,              # 违约
            '盈喜': 2.0,               # 盈利预喜
            '盈警': -2.0,              # 盈利预警
            '减持': -1.0,              # 减持
            '增持': 1.0,               # 增持
            '回购': 1.5,               # 回购
            '分红': 1.2,               # 分红
            '停牌': -0.5,              # 停牌
            '复牌': 0.5,               # 复牌
            
            # 行业特定词汇
            'breakthrough': 2.0,      # 突破（药物/技术）
            'patent': 1.5,            # 专利
            'fda approval': 2.5,      # FDA批准
            'clinical trial': 0.5,    # 临床试验
            'recall': -2.0,           # 召回
            'glitch': -1.5,           # 故障
            'hack': -2.0,             # 黑客攻击
            'security breach': -2.0,  # 安全漏洞
            'disruption': -1.5,       # 业务中断
            'innovation': 1.8,        # 创新
            'sustainable': 1.0,       # 可持续
            'renewable': 1.0,         # 可再生
            'tariff': -1.0,           # 关税
            'sanction': -1.5,         # 制裁
            'regulation': -0.5,       # 监管
            'deregulation': 0.5,      # 放松监管
            'subsidy': 1.0,           # 补贴

            # 公司事件
            'ipo': 1.5,               # 首次公开募股
            'delisting': -2.0,        # 退市
            'spinoff': 0.5,           # 分拆
            'restructuring': 0.0,     # 重组（中性）
            'insider trading': -2.0,  # 内幕交易
            'ceo departure': -1.0,    # CEO离职
            'ceo appointment': 1.0,   # CEO任命
            'layoffs': -1.5,          # 裁员
            'hiring': 1.2,            # 招聘
            'strike': -1.5,           # 罢工
            'labor dispute': -1.2,    # 劳资纠纷
            'dividend cut': -2.0,     # 削减股息
            'dividend increase': 2.0, # 增加股息
            'stock split': 0.5,       # 股票分割
            'reverse split': -0.5,    # 反向分割
        }
        
        # 更新VADER词典
        self.sia.lexicon.update(financial_lexicon)
        logger.info("已增强VADER词典，添加了金融领域特定词汇")
        
    def analyze_text(self, text):
        """分析单条文本的情感得分。"""
        if not text or not isinstance(text, str):
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 0.0,
                'neg': 0.0
            }
            
        # 获取情感得分
        sentiment_scores = self.sia.polarity_scores(text)
        return sentiment_scores
        
    def analyze_news_dataframe(self, news_df, text_column='text', date_column='date'):
        """
        分析包含新闻的DataFrame的情感得分。
        
        参数:
            news_df: 包含新闻的DataFrame
            text_column: 文本列的名称
            date_column: 日期列的名称
            
        返回:
            添加了情感得分的DataFrame
        """
        if news_df.empty:
            logger.warning("输入的新闻DataFrame为空")
            return news_df
            
        # 检查必要的列是否存在
        if text_column not in news_df.columns:
            logger.error(f"输入的DataFrame中没有列 '{text_column}'")
            return news_df
            
        # 复制DataFrame以避免修改原始数据
        result_df = news_df.copy()
        
        # 使用apply函数分析每条新闻的情感
        logger.info(f"开始分析 {len(result_df)} 条新闻的情感...")
        
        # 定义处理函数
        def process_row(row):
            text = row[text_column]
            sentiment = self.analyze_text(text)
            return pd.Series({
                'sentiment_compound': sentiment['compound'],
                'sentiment_positive': sentiment['pos'],
                'sentiment_neutral': sentiment['neu'],
                'sentiment_negative': sentiment['neg']
            })
        
        # 并行处理以提高效率
        with ThreadPoolExecutor() as executor:
            sentiment_results = list(executor.map(
                lambda row: process_row(row),
                [row for _, row in result_df.iterrows()]
            ))
        
        # 将结果添加到DataFrame
        sentiment_df = pd.DataFrame(sentiment_results)
        result_df = pd.concat([result_df, sentiment_df], axis=1)
        
        logger.info("情感分析完成")
        return result_df
        
    def aggregate_daily_sentiment(self, news_df, date_column='date', weight_column=None):
        """
        将新闻情感聚合为每日情感得分。
        
        参数:
            news_df: 包含新闻和情感得分的DataFrame
            date_column: 日期列的名称
            weight_column: 权重列的名称（如果有）
            
        返回:
            包含每日聚合情感得分的DataFrame
        """
        if news_df.empty:
            logger.warning("输入的新闻DataFrame为空")
            return pd.DataFrame()
            
        # 检查必要的列是否存在
        required_columns = [date_column, 'sentiment_compound']
        if not all(col in news_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in news_df.columns]
            logger.error(f"输入的DataFrame中缺少必要的列: {missing}")
            return pd.DataFrame()
            
        # 确保日期列为datetime类型
        news_df = news_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(news_df[date_column]):
            try:
                news_df[date_column] = pd.to_datetime(news_df[date_column])
            except Exception as e:
                logger.error(f"无法将 {date_column} 列转换为日期时间格式: {str(e)}")
                return pd.DataFrame()
                
        # 提取日期部分（不包括时间）
        news_df['date_only'] = news_df[date_column].dt.date
        
        # 计算每日聚合情感
        logger.info("计算每日聚合情感...")
        
        # 如果提供了权重列，使用加权平均
        if weight_column and weight_column in news_df.columns:
            daily_sentiment = news_df.groupby('date_only').apply(
                lambda x: np.average(
                    x['sentiment_compound'], 
                    weights=x[weight_column] if x[weight_column].sum() > 0 else None
                )
            ).reset_index()
            daily_sentiment.columns = ['date', 'sentiment_score']
        else:
            # 否则使用简单平均
            daily_sentiment = news_df.groupby('date_only')['sentiment_compound'].mean().reset_index()
            daily_sentiment.columns = ['date', 'sentiment_score']
            
        # 计算情感波动性和情感趋势
        if len(daily_sentiment) > 1:
            daily_sentiment['sentiment_volatility'] = daily_sentiment['sentiment_score'].rolling(window=3, min_periods=1).std()
            daily_sentiment['sentiment_momentum'] = daily_sentiment['sentiment_score'].diff()
        else:
            daily_sentiment['sentiment_volatility'] = 0
            daily_sentiment['sentiment_momentum'] = 0
            
        logger.info(f"每日情感聚合完成，共 {len(daily_sentiment)} 天的数据")
        return daily_sentiment
        
    def merge_with_price_data(self, sentiment_df, price_df, date_column='date'):
        """
        将情感数据与价格数据合并。
        
        参数:
            sentiment_df: 包含情感得分的DataFrame
            price_df: 包含价格数据的DataFrame
            date_column: 日期列的名称
            
        返回:
            合并后的DataFrame
        """
        if sentiment_df.empty or price_df.empty:
            logger.warning("情感DataFrame或价格DataFrame为空")
            return price_df
            
        # 确保两个DataFrame的日期列都是datetime类型
        sentiment_df = sentiment_df.copy()
        price_df = price_df.copy()
        
        for df in [sentiment_df, price_df]:
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                try:
                    df[date_column] = pd.to_datetime(df[date_column])
                except Exception as e:
                    logger.error(f"无法将DataFrame的 {date_column} 列转换为日期时间格式: {str(e)}")
                    return price_df
                    
        # 合并DataFrames
        logger.info("将情感数据与价格数据合并...")
        merged_df = pd.merge(price_df, sentiment_df, on=date_column, how='left')
        
        # 用前一个有效值填充空缺的情感分数
        sentiment_columns = ['sentiment_score', 'sentiment_volatility', 'sentiment_momentum']
        for col in sentiment_columns:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(method='ffill')
                
        # 如果仍有缺失值，填充为0（对于没有情感数据的最早几天）
        merged_df = merged_df.fillna({col: 0 for col in sentiment_columns if col in merged_df.columns})
        
        logger.info(f"数据合并完成，最终DataFrame包含 {len(merged_df)} 行")
        return merged_df
        
    def analyze_news_file(self, file_path, ticker=None, text_column='text', date_column='date'):
        """
        分析包含新闻的文件（CSV或JSON）。
        
        参数:
            file_path: 文件路径
            ticker: 股票代码
            text_column: 文本列的名称
            date_column: 日期列的名称
            
        返回:
            包含情感分析结果的DataFrame
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return pd.DataFrame()
            
        # 根据文件扩展名读取文件
        file_ext = os.path.splitext(file_path)[1].lower()
        try:
            if file_ext == '.csv':
                news_df = pd.read_csv(file_path)
            elif file_ext == '.json':
                news_df = pd.read_json(file_path)
            else:
                logger.error(f"不支持的文件格式: {file_ext}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"读取文件 {file_path} 时出错: {str(e)}")
            return pd.DataFrame()
            
        # 如果没有文本列，尝试组合标题和摘要
        if text_column not in news_df.columns:
            if 'title' in news_df.columns and 'summary' in news_df.columns:
                news_df[text_column] = news_df['title'] + ' ' + news_df['summary']
            elif 'title' in news_df.columns:
                news_df[text_column] = news_df['title']
            elif 'summary' in news_df.columns:
                news_df[text_column] = news_df['summary']
            else:
                logger.error(f"找不到可用于分析的文本列")
                return pd.DataFrame()
                
        # 如果提供了股票代码，添加到DataFrame
        if ticker:
            news_df['ticker'] = ticker
            
        # 分析情感
        result_df = self.analyze_news_dataframe(news_df, text_column, date_column)
        
        return result_df
        
    def save_sentiment_data(self, sentiment_df, ticker, file_name=None):
        """保存情感分析数据到文件。"""
        if sentiment_df.empty:
            logger.warning("没有数据可保存")
            return
            
        # 如果没有提供文件名，使用默认格式
        if not file_name:
            file_name = f"{ticker}_sentiment_data.csv"
            
        # 构建完整路径
        file_path = os.path.join(self.data_dir, file_name)
        
        # 保存到CSV
        try:
            sentiment_df.to_csv(file_path, index=False)
            logger.info(f"情感数据已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存情感数据时出错: {str(e)}")
            
    def plot_sentiment_vs_price(self, data, ticker=None, output_dir='./plots'):
        """
        绘制情感得分与股价的对比图。
        
        参数:
            data: 包含价格和情感数据的DataFrame
            ticker: 股票代码
            output_dir: 图表输出目录
        """
        if data.empty:
            logger.warning("没有数据可绘图")
            return
            
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查必要的列是否存在
        required_columns = ['date', 'close', 'sentiment_score']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"输入的DataFrame中缺少必要的列: {missing}")
            return
            
        # 设置图表风格
        plt.style.use('fivethirtyeight')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 绘制股价图
        ax1.plot(data['date'], data['close'], 'b-', label='收盘价')
        ax1.set_ylabel('价格')
        if ticker:
            ax1.set_title(f'{ticker} 股价与情感得分对比')
        else:
            ax1.set_title('股价与情感得分对比')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # 绘制情感得分图，用颜色表示正负
        sentiment = data['sentiment_score']
        colors = ['red' if score < 0 else 'green' for score in sentiment]
        ax2.bar(data['date'], sentiment, color=colors, alpha=0.7, label='情感得分')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_ylabel('情感得分')
        ax2.set_xlabel('日期')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # 设置x轴日期格式
        fig.autofmt_xdate()
        
        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图表
        if ticker:
            file_name = f"{ticker}_sentiment_vs_price.png"
        else:
            file_name = "sentiment_vs_price.png"
        file_path = os.path.join(output_dir, file_name)
        
        try:
            plt.savefig(file_path)
            logger.info(f"图表已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存图表时出错: {str(e)}")
            
        plt.close()
        
    def visualize_sentiment_metrics(self, data, ticker=None, output_dir='./plots'):
        """
        可视化多种情感指标。
        
        参数:
            data: 包含价格和情感数据的DataFrame
            ticker: 股票代码
            output_dir: 图表输出目录
        """
        if data.empty:
            logger.warning("没有数据可绘图")
            return
            
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查必要的列是否存在
        required_columns = ['date', 'close', 'sentiment_score']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"输入的DataFrame中缺少必要的列: {missing}")
            return
            
        # 计算价格变化百分比
        data['price_change'] = data['close'].pct_change() * 100
        
        # 设置图表风格
        sns.set(style="whitegrid")
        
        # 创建多个子图
        fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
        
        # 1. 价格变化与情感得分
        ax1 = axes[0]
        ax1.plot(data['date'], data['price_change'], 'b-', label='价格变化%')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(data['date'], data['sentiment_score'], 'r-', label='情感得分')
        ax1.set_ylabel('价格变化%')
        ax1_twin.set_ylabel('情感得分')
        ax1.set_title('价格变化与情感得分')
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 2. 情感波动性
        if 'sentiment_volatility' in data.columns:
            ax2 = axes[1]
            ax2.plot(data['date'], data['sentiment_volatility'], 'g-', label='情感波动性')
            ax2.set_ylabel('情感波动性')
            ax2.set_title('情感波动性')
            ax2.legend(loc='upper left')
        
        # 3. 情感动量（变化趋势）
        if 'sentiment_momentum' in data.columns:
            ax3 = axes[2]
            colors = ['red' if mom < 0 else 'green' for mom in data['sentiment_momentum']]
            ax3.bar(data['date'], data['sentiment_momentum'], color=colors, alpha=0.7, label='情感动量')
            ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax3.set_ylabel('情感动量')
            ax3.set_title('情感动量（变化趋势）')
            ax3.legend(loc='upper left')
        
        # 设置主标题
        if ticker:
            plt.suptitle(f'{ticker} 股票情感指标分析', fontsize=16)
        else:
            plt.suptitle('股票情感指标分析', fontsize=16)
        
        # 设置x轴日期格式
        fig.autofmt_xdate()
        
        # 调整子图之间的间距
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # 保存图表
        if ticker:
            file_name = f"{ticker}_sentiment_metrics.png"
        else:
            file_name = "sentiment_metrics.png"
        file_path = os.path.join(output_dir, file_name)
        
        try:
            plt.savefig(file_path)
            logger.info(f"情感指标图表已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存图表时出错: {str(e)}")
            
        plt.close()
        
    def analyze_sentiment_effect(self, data, price_column='close', lag_days=1):
        """
        分析情感得分对未来股价的影响。
        
        参数:
            data: 包含价格和情感数据的DataFrame
            price_column: 价格列的名称
            lag_days: 滞后天数
            
        返回:
            分析结果字典
        """
        if data.empty:
            logger.warning("没有数据可分析")
            return {}
            
        # 检查必要的列是否存在
        required_columns = ['sentiment_score', price_column]
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"输入的DataFrame中缺少必要的列: {missing}")
            return {}
            
        # 计算价格变化百分比
        data['price_change'] = data[price_column].pct_change() * 100
        
        # 创建滞后特征
        for i in range(1, lag_days + 1):
            data[f'sentiment_lag_{i}'] = data['sentiment_score'].shift(i)
            
        # 删除有NaN的行
        data_clean = data.dropna()
        
        if len(data_clean) < 10:
            logger.warning("清洗后的数据不足，无法进行有意义的分析")
            return {}
            
        # 计算相关性
        correlation = data_clean['sentiment_score'].corr(data_clean['price_change'])
        
        # 分析情感极性与价格变化的关系
        positive_sentiment = data_clean[data_clean['sentiment_score'] > 0]
        negative_sentiment = data_clean[data_clean['sentiment_score'] < 0]
        neutral_sentiment = data_clean[data_clean['sentiment_score'] == 0]
        
        avg_change_after_positive = positive_sentiment['price_change'].mean()
        avg_change_after_negative = negative_sentiment['price_change'].mean()
        avg_change_after_neutral = neutral_sentiment['price_change'].mean()
        
        # 分析滞后特征的影响
        lag_correlations = {}
        for i in range(1, lag_days + 1):
            lag_correlations[i] = data_clean[f'sentiment_lag_{i}'].corr(data_clean['price_change'])
            
        results = {
            'correlation': correlation,
            'avg_change_after_positive': avg_change_after_positive,
            'avg_change_after_negative': avg_change_after_negative,
            'avg_change_after_neutral': avg_change_after_neutral,
            'lag_correlations': lag_correlations,
            'data_points': len(data_clean)
        }
        
        return results
        
    def generate_trading_signals(self, data, sentiment_threshold_buy=0.3, sentiment_threshold_sell=-0.3,
                                 momentum_threshold_buy=0.1, momentum_threshold_sell=-0.1,
                                 volatility_threshold=0.15, price_column='close'):
        """
        根据情感分析结果生成交易信号。
        
        参数:
            data: 包含价格和情感数据的DataFrame
            sentiment_threshold_buy: 买入信号的情感阈值
            sentiment_threshold_sell: 卖出信号的情感阈值
            momentum_threshold_buy: 买入信号的情感动量阈值
            momentum_threshold_sell: 卖出信号的情感动量阈值
            volatility_threshold: 情感波动性阈值
            price_column: 价格列的名称
            
        返回:
            添加了交易信号的DataFrame
        """
        if data.empty:
            logger.warning("输入数据为空，无法生成交易信号")
            return data
            
        # 检查必要的列是否存在
        required_columns = ['sentiment_score', price_column]
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"输入的DataFrame中缺少必要的列: {missing}")
            return data
            
        # 复制DataFrame以避免修改原始数据
        result = data.copy()
        
        # 计算技术指标
        # 1. 简单移动平均线 (5日和20日)
        result['ma5'] = result[price_column].rolling(window=5).mean()
        result['ma20'] = result[price_column].rolling(window=20).mean()
        
        # 2. 价格相对强弱指标
        result['price_rsi'] = 0  # 默认值
        # 计算价格变化
        delta = result[price_column].diff()
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        # 计算平均增益和平均损失
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        # 计算相对强弱
        rs = avg_gain / avg_loss.replace(0, 1e-10)  # 避免除以零
        # 计算RSI
        result['price_rsi'] = 100 - (100 / (1 + rs))
        
        # 3. 情感RSI (可选，如果有足够的情感数据)
        if 'sentiment_score' in result.columns:
            result['sent_delta'] = result['sentiment_score'].diff()
            sent_gain = result['sent_delta'].where(result['sent_delta'] > 0, 0)
            sent_loss = -result['sent_delta'].where(result['sent_delta'] < 0, 0)
            sent_avg_gain = sent_gain.rolling(window=5).mean()
            sent_avg_loss = sent_loss.rolling(window=5).mean()
            sent_rs = sent_avg_gain / sent_avg_loss.replace(0, 1e-10)
            result['sentiment_rsi'] = 100 - (100 / (1 + sent_rs))
        
        # 初始化信号列
        result['signal'] = 0  # 0: 持有, 1: 买入, -1: 卖出
        
        # 生成信号逻辑
        for i in range(1, len(result)):
            # 获取当前行的值
            sentiment = result.iloc[i]['sentiment_score']
            
            # 获取情感动量(如果存在)
            momentum = result.iloc[i].get('sentiment_momentum', 0)
            
            # 获取情感波动性(如果存在)
            volatility = result.iloc[i].get('sentiment_volatility', 0)
            
            # 获取价格技术指标
            ma5 = result.iloc[i].get('ma5', 0)
            ma20 = result.iloc[i].get('ma20', 0)
            price_rsi = result.iloc[i].get('price_rsi', 50)
            
            # 综合买入信号条件:
            # 1. 情感分数高于买入阈值
            # 2. 情感动量为正且高于动量阈值
            # 3. 价格技术指标支持买入(例如MA5 > MA20或RSI < 30)
            # 4. 情感波动性在可接受范围内
            buy_signal = (
                (sentiment > sentiment_threshold_buy) and 
                (momentum > momentum_threshold_buy if 'sentiment_momentum' in result.columns else True) and
                (ma5 > ma20 if not pd.isna(ma5) and not pd.isna(ma20) else False) and
                (volatility < volatility_threshold if 'sentiment_volatility' in result.columns else True)
            )
            
            # 综合卖出信号条件:
            # 1. 情感分数低于卖出阈值
            # 2. 情感动量为负且低于动量阈值
            # 3. 价格技术指标支持卖出(例如MA5 < MA20或RSI > 70)
            sell_signal = (
                (sentiment < sentiment_threshold_sell) and 
                (momentum < momentum_threshold_sell if 'sentiment_momentum' in result.columns else True) and
                (ma5 < ma20 if not pd.isna(ma5) and not pd.isna(ma20) else False)
            )
            
            # 根据条件设置信号
            if buy_signal:
                result.iloc[i, result.columns.get_loc('signal')] = 1
            elif sell_signal:
                result.iloc[i, result.columns.get_loc('signal')] = -1
        
        logger.info(f"交易信号生成完成，买入信号: {sum(result['signal'] == 1)}个，卖出信号: {sum(result['signal'] == -1)}个")
        return result
        
    def plot_trading_signals(self, data, ticker=None, output_dir='./plots'):
        """
        绘制带有交易信号的价格和情感图表。
        
        参数:
            data: 包含价格、情感和交易信号的DataFrame
            ticker: 股票代码
            output_dir: 图表输出目录
        """
        if data.empty:
            logger.warning("没有数据可绘图")
            return
            
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查必要的列是否存在
        required_columns = ['date', 'close', 'sentiment_score', 'signal']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"输入的DataFrame中缺少必要的列: {missing}")
            return
            
        # 设置图表风格
        plt.style.use('fivethirtyeight')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 绘制股价图
        ax1.plot(data['date'], data['close'], 'b-', label='收盘价')
        
        # 添加买入和卖出信号
        buy_signals = data[data['signal'] == 1]
        sell_signals = data[data['signal'] == -1]
        
        if not buy_signals.empty:
            ax1.scatter(buy_signals['date'], buy_signals['close'], marker='^', color='g', s=100, label='买入信号')
        if not sell_signals.empty:
            ax1.scatter(sell_signals['date'], sell_signals['close'], marker='v', color='r', s=100, label='卖出信号')
            
        ax1.set_ylabel('价格')
        if ticker:
            ax1.set_title(f'{ticker} 股价与情感交易信号')
        else:
            ax1.set_title('股价与情感交易信号')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # 绘制情感得分图，用颜色表示正负
        sentiment = data['sentiment_score']
        colors = ['red' if score < 0 else 'green' for score in sentiment]
        ax2.bar(data['date'], sentiment, color=colors, alpha=0.7, label='情感得分')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # 添加情感阈值参考线（如果存在）
        if 'signal' in data.columns:
            # 尝试找出信号生成中使用的阈值
            positive_threshold = data[data['signal'] == 1]['sentiment_score'].min()
            negative_threshold = data[data['signal'] == -1]['sentiment_score'].max()
            
            if not pd.isna(positive_threshold):
                ax2.axhline(y=positive_threshold, color='green', linestyle='--', alpha=0.5, label='买入阈值')
            if not pd.isna(negative_threshold):
                ax2.axhline(y=negative_threshold, color='red', linestyle='--', alpha=0.5, label='卖出阈值')
        
        ax2.set_ylabel('情感得分')
        ax2.set_xlabel('日期')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # 设置x轴日期格式
        fig.autofmt_xdate()
        
        # 调整子图之间的间距
        plt.tight_layout()
        
        # 保存图表
        if ticker:
            file_name = f"{ticker}_trading_signals.png"
        else:
            file_name = "trading_signals.png"
        file_path = os.path.join(output_dir, file_name)
        
        try:
            plt.savefig(file_path)
            logger.info(f"交易信号图表已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存图表时出错: {str(e)}")
            
        plt.close()
        
    def backtest_trading_strategy(self, data, initial_capital=10000, commission_rate=0.001, price_column='close'):
        """
        回测基于情感分析的交易策略。
        
        参数:
            data: 包含价格、情感和交易信号的DataFrame
            initial_capital: 初始资金
            commission_rate: 交易佣金率
            price_column: 价格列的名称
            
        返回:
            回测结果字典
        """
        if data.empty:
            logger.warning("没有数据可回测")
            return {}
            
        # 检查必要的列是否存在
        required_columns = ['date', price_column, 'signal']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            logger.error(f"输入的DataFrame中缺少必要的列: {missing}")
            return {}
            
        # 初始化结果列表
        positions = []  # 记录每次交易
        portfolio_value = []  # 每日投资组合价值
        cash = initial_capital  # 可用现金
        shares = 0  # 持有的股份数
        
        # 循环每一行数据
        for i, row in data.iterrows():
            date = row['date']
            price = row[price_column]
            signal = row['signal']
            
            # 计算当前投资组合价值
            current_value = cash + shares * price
            portfolio_value.append({
                'date': date,
                'value': current_value,
                'cash': cash,
                'shares': shares,
                'price': price
            })
            
            # 处理买入信号
            if signal == 1 and cash > price:
                # 计算可购买的股数（考虑佣金）
                max_shares = int(cash / price / (1 + commission_rate))
                
                if max_shares > 0:
                    # 计算实际成本
                    cost = max_shares * price * (1 + commission_rate)
                    # 更新持仓
                    shares += max_shares
                    cash -= cost
                    
                    # 记录交易
                    positions.append({
                        'date': date,
                        'action': 'buy',
                        'price': price,
                        'shares': max_shares,
                        'cost': cost,
                        'commission': max_shares * price * commission_rate,
                        'cash_after': cash,
                        'portfolio_value': cash + shares * price
                    })
            
            # 处理卖出信号
            elif signal == -1 and shares > 0:
                # 计算实际收益
                revenue = shares * price * (1 - commission_rate)
                # 更新持仓
                old_shares = shares
                shares = 0
                cash += revenue
                
                # 记录交易
                positions.append({
                    'date': date,
                    'action': 'sell',
                    'price': price,
                    'shares': old_shares,
                    'revenue': revenue,
                    'commission': old_shares * price * commission_rate,
                    'cash_after': cash,
                    'portfolio_value': cash
                })
        
        # 如果结束时仍有股票，执行最后一次卖出（平仓）
        if shares > 0:
            last_price = data.iloc[-1][price_column]
            revenue = shares * last_price * (1 - commission_rate)
            
            positions.append({
                'date': data.iloc[-1]['date'],
                'action': 'final_sell',
                'price': last_price,
                'shares': shares,
                'revenue': revenue,
                'commission': shares * last_price * commission_rate,
                'cash_after': cash + revenue,
                'portfolio_value': cash + revenue
            })
            
            cash += revenue
            shares = 0
        
        # 计算最终投资组合价值
        final_value = cash
        
        # 计算投资回报率
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # 计算策略年化收益率
        days = (data.iloc[-1]['date'] - data.iloc[0]['date']).days
        annual_return = (final_value / initial_capital) ** (365 / max(days, 1)) - 1
        
        # 计算买入并持有策略的回报
        buy_hold_return = (data.iloc[-1][price_column] - data.iloc[0][price_column]) / data.iloc[0][price_column] * 100
        
        # 计算最大回撤
        max_drawdown = 0
        peak_value = 0
        
        for p in portfolio_value:
            if p['value'] > peak_value:
                peak_value = p['value']
            
            drawdown = (peak_value - p['value']) / peak_value
            max_drawdown = max(max_drawdown, drawdown)
        
        # 计算夏普比率
        daily_returns = []
        for i in range(1, len(portfolio_value)):
            ret = (portfolio_value[i]['value'] - portfolio_value[i-1]['value']) / portfolio_value[i-1]['value']
            daily_returns.append(ret)
            
        risk_free_rate = 0.02 / 365  # 假设年化无风险利率为2%
        avg_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        
        if std_return > 0:
            sharpe_ratio = (avg_return - risk_free_rate) / std_return * np.sqrt(252)  # 年化夏普比率
        else:
            sharpe_ratio = 0
        
        # 计算胜率
        win_trades = sum(1 for p in positions if p['action'] == 'sell' and p['revenue'] > p.get('cost', 0))
        total_trades = sum(1 for p in positions if p['action'] == 'sell')
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        
        # 返回回测结果
        results = {
            'summary': {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return_pct': total_return,
                'annual_return_pct': annual_return * 100,
                'buy_hold_return_pct': buy_hold_return,
                'max_drawdown_pct': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'total_trades': len(positions),
                'buy_trades': sum(1 for p in positions if p['action'] == 'buy'),
                'sell_trades': sum(1 for p in positions if p['action'] in ['sell', 'final_sell']),
                'commission_paid': sum(p.get('commission', 0) for p in positions)
            },
            'positions': positions,
            'portfolio_values': portfolio_value
        }
        
        return results
        
    def plot_backtest_results(self, backtest_results, ticker=None, output_dir='./plots'):
        """
        绘制回测结果图表。
        
        参数:
            backtest_results: 回测结果字典
            ticker: 股票代码
            output_dir: 图表输出目录
        """
        if not backtest_results or 'portfolio_values' not in backtest_results:
            logger.warning("没有有效的回测结果可绘图")
            return
            
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建投资组合价值时间序列
        portfolio_values = backtest_results['portfolio_values']
        dates = [p['date'] for p in portfolio_values]
        values = [p['value'] for p in portfolio_values]
        prices = [p['price'] for p in portfolio_values]
        
        # 创建交易点位
        positions = backtest_results['positions']
        buy_dates = [p['date'] for p in positions if p['action'] == 'buy']
        buy_values = [p['portfolio_value'] for p in positions if p['action'] == 'buy']
        sell_dates = [p['date'] for p in positions if p['action'] in ['sell', 'final_sell']]
        sell_values = [p['portfolio_value'] for p in positions if p['action'] in ['sell', 'final_sell']]
        
        # 设置图表风格
        plt.style.use('fivethirtyeight')
        
        # 绘制投资组合价值图和基准价格对比图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 投资组合价值
        ax1.plot(dates, values, 'b-', label='投资组合价值')
        # 添加买入和卖出点
        if buy_dates:
            ax1.scatter(buy_dates, buy_values, marker='^', color='g', s=100, label='买入')
        if sell_dates:
            ax1.scatter(sell_dates, sell_values, marker='v', color='r', s=100, label='卖出')
        
        # 添加初始资金参考线
        initial_capital = backtest_results['summary']['initial_capital']
        ax1.axhline(y=initial_capital, color='gray', linestyle='-', alpha=0.3, label='初始资金')
        
        ax1.set_ylabel('投资组合价值')
        if ticker:
            ax1.set_title(f'{ticker} 情感交易策略回测结果')
        else:
            ax1.set_title('情感交易策略回测结果')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # 基准价格走势
        scaled_prices = np.array(prices) * (initial_capital / prices[0])  # 缩放价格以便比较
        ax2.plot(dates, scaled_prices, 'g-', label='买入持有策略')
        ax2.plot(dates, values, 'b-', label='情感交易策略')
        ax2.set_ylabel('价值')
        ax2.set_xlabel('日期')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # 添加回测结果摘要文本框
        summary = backtest_results['summary']
        summary_text = (
            f"初始资金: ${summary['initial_capital']:.2f}\n"
            f"最终价值: ${summary['final_value']:.2f}\n"
            f"总收益率: {summary['total_return_pct']:.2f}%\n"
            f"年化收益率: {summary['annual_return_pct']:.2f}%\n"
            f"买入持有收益率: {summary['buy_hold_return_pct']:.2f}%\n"
            f"最大回撤: {summary['max_drawdown_pct']:.2f}%\n"
            f"夏普比率: {summary['sharpe_ratio']:.2f}\n"
            f"胜率: {summary['win_rate']*100:.2f}%\n"
            f"交易次数: {summary['total_trades']}"
        )
        
        fig.text(0.15, 0.02, summary_text, bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        
        # 设置x轴日期格式
        fig.autofmt_xdate()
        
        # 调整子图之间的间距
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.20)
        
        # 保存图表
        if ticker:
            file_name = f"{ticker}_backtest_results.png"
        else:
            file_name = "backtest_results.png"
        file_path = os.path.join(output_dir, file_name)
        
        try:
            plt.savefig(file_path)
            logger.info(f"回测结果图表已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存图表时出错: {str(e)}")
            
        plt.close()

def find_latest_news_file(ticker, news_dir='./data/news'):
    """查找指定ticker的最新新闻文件。"""
    search_pattern = os.path.join(news_dir, f"{ticker}_news_*.csv")
    files = glob.glob(search_pattern)
    
    if not files:
        return None
        
    # 解析文件名中的日期并找到最新的文件
    latest_file = None
    latest_date = None
    date_pattern = re.compile(r'_news_(\d{8}).csv$') # Matches YYYYMMDD
    
    for f in files:
        match = date_pattern.search(os.path.basename(f))
        if match:
            date_str = match.group(1)
            try:
                current_date = datetime.strptime(date_str, '%Y%m%d')
                if latest_date is None or current_date > latest_date:
                    latest_date = current_date
                    latest_file = f
            except ValueError:
                continue # Skip files with invalid date format
                
    return latest_file

def main(ticker=None, news_file=None, price_file=None, start_date=None, end_date=None, backtest=True, predefined=False):
    """
    主函数，执行情感分析流程。
    支持直接运行处理预定义列表，或指定单个ticker。
    
    参数:
        ticker: 单个股票代码 (如果提供，优先于predefined)
        news_file: 特定新闻文件路径 (如果提供，覆盖自动查找)
        price_file: 特定价格文件路径 (如果提供，覆盖默认路径)
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        backtest: 是否进行回测分析
        predefined: 是否强制使用预定义列表 (即使提供了ticker)
    """
    
    # 确定要处理的股票列表
    tickers_to_process = []
    run_default = False
    
    if ticker and not predefined:
        tickers_to_process = [ticker]
        logger.info(f"指定处理单个股票: {ticker}")
    else:
        # 使用预定义的股票列表（与process_stock_data.py相同）
        tickers_to_process = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',       # 科技
            'JPM', 'BAC', 'C', 'WFC', 'GS',                # 金融
            'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',            # 医药
            'XOM', 'CVX', 'COP', 'SLB', 'BKR',             # 能源
            'DIS', 'NFLX', 'CMCSA', 'NKE', 'SBUX',         # 消费
            'CAT', 'DE', 'MMM', 'GE', 'HON'                # 工业
        ]
        logger.info("使用默认预定义的股票列表进行处理")
        run_default = True

    # 循环处理每个ticker
    for current_ticker in tickers_to_process:
        logger.info(f"-- 开始处理股票: {current_ticker} --")
        
        # 确定要使用的文件路径
        current_news_file = news_file
        current_price_file = price_file
        
        if run_default or not current_news_file:
            # 自动查找最新的新闻文件
            found_news_file = find_latest_news_file(current_ticker)
            if found_news_file:
                current_news_file = found_news_file
                logger.info(f"找到最新的新闻文件: {current_news_file}")
            else:
                logger.warning(f"未找到 {current_ticker} 的新闻文件，跳过此股票。")
                continue # 跳到下一个ticker
                
        if run_default or not current_price_file:
            # 使用默认的价格文件路径
            default_price_path = f"./data/{current_ticker}.csv"
            if os.path.exists(default_price_path):
                current_price_file = default_price_path
                logger.info(f"使用默认价格文件: {current_price_file}")
            else:
                logger.warning(f"未找到 {current_ticker} 的价格文件 ({default_price_path})，跳过此股票。")
                continue # 跳到下一个ticker

        # --- 执行核心分析流程 (与之前类似，但使用current_ticker, current_news_file, current_price_file) ---
        logger.info(f"为 {current_ticker} 执行情感分析...")
        
        # --- 1. 参数设置和初始化 ---
        analyzer = StockSentimentAnalyzer()
        output_dir = analyzer.data_dir
        plot_dir = './plots'
        os.makedirs(plot_dir, exist_ok=True)
        
        # 检查文件是否存在 (虽然上面已检查，双重保险)
        if not current_news_file or not os.path.exists(current_news_file):
            logger.error(f"错误：新闻文件不存在: {current_news_file}")
            continue
        if not current_price_file or not os.path.exists(current_price_file):
            logger.error(f"错误：价格文件不存在: {current_price_file}")
            continue
            
        # --- 2. 加载和分析新闻数据 ---
        logger.info(f"加载并分析新闻文件: {current_news_file}")
        news_with_sentiment = analyzer.analyze_news_file(current_news_file, current_ticker)
        
        if news_with_sentiment.empty:
            logger.error(f"无法从 {current_news_file} 分析情感。")
            continue
            
        # --- 3. 聚合每日情感 ---
        logger.info("聚合每日情感得分...")
        daily_sentiment = analyzer.aggregate_daily_sentiment(news_with_sentiment)
        
        if daily_sentiment.empty:
            logger.error("无法聚合每日情感得分。")
            continue
            
        # 保存初步的每日情感数据
        daily_sentiment_filename = f"{current_ticker}_daily_sentiment.csv"
        analyzer.save_sentiment_data(daily_sentiment, current_ticker, file_name=daily_sentiment_filename)
        
        # --- 4. 加载价格数据 ---
        logger.info(f"加载价格文件: {current_price_file}")
        try:
            # 读取价格文件
            price_data = pd.read_csv(current_price_file)
            
            # 检查日期列名 - 处理可能的大小写不同情况
            date_column = 'date'
            if date_column not in price_data.columns:
                # 检查是否有'Date'列
                if 'Date' in price_data.columns:
                    # 重命名为小写，标准化列名
                    price_data.rename(columns={'Date': 'date'}, inplace=True)
                    logger.info("将'Date'列重命名为'date'以标准化")
                else:
                    # 查找可能的其他日期列名
                    possible_date_columns = [col for col in price_data.columns if col.lower() == 'date']
                    if possible_date_columns:
                        # 使用找到的第一个匹配的列名
                        price_data.rename(columns={possible_date_columns[0]: 'date'}, inplace=True)
                        logger.info(f"将'{possible_date_columns[0]}'列重命名为'date'以标准化")
                    else:
                        logger.error(f"价格文件中找不到日期列 (date/Date)")
                        continue
            
            # 确认日期列存在后，转换为datetime
            if not pd.api.types.is_datetime64_any_dtype(price_data['date']):
                price_data['date'] = pd.to_datetime(price_data['date'])
                
            # 检查close列是否存在，处理大小写问题  
            if 'close' not in price_data.columns:
                # 检查是否有'Close'列
                if 'Close' in price_data.columns:
                    # 重命名为小写，标准化列名
                    price_data.rename(columns={'Close': 'close'}, inplace=True)
                    logger.info("将'Close'列重命名为'close'以标准化")
                else:
                    # 查找可能的其他收盘价列名
                    possible_close_columns = [col for col in price_data.columns if col.lower() == 'close']
                    if possible_close_columns:
                        # 使用找到的第一个匹配的列名
                        price_data.rename(columns={possible_close_columns[0]: 'close'}, inplace=True)
                        logger.info(f"将'{possible_close_columns[0]}'列重命名为'close'以标准化")
                    else:
                        logger.error(f"价格文件中找不到收盘价列 (close/Close)")
                        continue
                    
        except Exception as e:
            logger.error(f"读取价格文件 {current_price_file} 时出错: {str(e)}")
            continue
            
        # --- 5. 筛选日期范围 (应用于价格数据) ---
        if start_date:
            try:
                start_date_dt = pd.to_datetime(start_date)
                price_data = price_data[price_data['date'] >= start_date_dt]
            except Exception as e:
                logger.error(f"处理开始日期 {start_date} 时出错: {str(e)}")
                continue
        if end_date:
            try:
                end_date_dt = pd.to_datetime(end_date)
                price_data = price_data[price_data['date'] <= end_date_dt]
            except Exception as e:
                logger.error(f"处理结束日期 {end_date} 时出错: {str(e)}")
                continue
                
        if price_data.empty:
            logger.error("筛选日期后价格数据为空。")
            continue
            
        # --- 6. 合并情感和价格数据 ---
        logger.info("合并情感数据与价格数据...")
        if not pd.api.types.is_datetime64_any_dtype(daily_sentiment['date']):
            daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
            
        merged_data = analyzer.merge_with_price_data(daily_sentiment, price_data)
        
        if merged_data.empty or 'sentiment_score' not in merged_data.columns:
            logger.error("合并数据失败或缺少情感得分列。")
            continue
            
        # --- 7. 可视化基础指标 ---
        logger.info("生成基础可视化图表...")
        analyzer.plot_sentiment_vs_price(merged_data, current_ticker, output_dir=plot_dir)
        analyzer.visualize_sentiment_metrics(merged_data, current_ticker, output_dir=plot_dir)
        
        # --- 8. 分析情感效应 ---
        logger.info("分析情感对价格的潜在影响...")
        effect_results = analyzer.analyze_sentiment_effect(merged_data)
        if effect_results:
            logger.info(f"情感分析效应结果: {json.dumps(effect_results, indent=2)}")
        else:
            logger.warning("未能计算情感效应分析结果。")
            
        # --- 9. 生成交易信号 ---
        logger.info("生成交易信号...")
        data_with_signals = analyzer.generate_trading_signals(merged_data)
        
        analyzer.plot_trading_signals(data_with_signals, current_ticker, output_dir=plot_dir)
        
        merged_signals_filename = f"{current_ticker}_with_sentiment_signals.csv"
        analyzer.save_sentiment_data(data_with_signals, current_ticker, file_name=merged_signals_filename)
        
        # --- 10. 回测交易策略 (如果启用) ---
        if backtest:
            logger.info("执行交易策略回测...")
            backtest_results = analyzer.backtest_trading_strategy(data_with_signals)
            
            if backtest_results and 'summary' in backtest_results:
                logger.info("回测结果摘要:")
                print(json.dumps(backtest_results['summary'], indent=2))
                
                analyzer.plot_backtest_results(backtest_results, current_ticker, output_dir=plot_dir)
                
                backtest_summary_filename = f"{current_ticker}_backtest_summary.csv"
                backtest_summary_df = pd.DataFrame([backtest_results['summary']])
                try:
                    backtest_summary_df.to_csv(os.path.join(output_dir, backtest_summary_filename), index=False)
                    logger.info(f"回测摘要已保存到 {os.path.join(output_dir, backtest_summary_filename)}")
                except Exception as e:
                    logger.error(f"保存回测摘要时出错: {str(e)}")
                
                if backtest_results.get('positions'):
                    trades_filename = f"{current_ticker}_trades.csv"
                    trades_df = pd.DataFrame(backtest_results['positions'])
                    try:
                        trades_df.to_csv(os.path.join(output_dir, trades_filename), index=False)
                        logger.info(f"交易记录已保存到 {os.path.join(output_dir, trades_filename)}")
                    except Exception as e:
                        logger.error(f"保存交易记录时出错: {str(e)}")
            else:
                logger.warning("未能成功执行回测。")
        else:
            logger.info("回测分析已禁用。")
            
        logger.info(f"-- 完成处理股票: {current_ticker} --\n")
        
    logger.info("所有指定股票的情感分析流程完成。")
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='使用VADER模型进行股票情感分析和策略回测')
    # 修改参数为非必需
    parser.add_argument('--ticker', type=str, help='要处理的单个股票代码 (例如 AAPL)')
    parser.add_argument('--news_file', type=str, help='特定新闻数据文件路径 (如果提供，覆盖自动查找)')
    parser.add_argument('--price_file', type=str, help='特定价格数据文件路径 (如果提供，覆盖默认路径)')
    parser.add_argument('--start_date', type=str, help='分析开始日期 (格式: YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='分析结束日期 (格式: YYYY-MM-DD)')
    parser.add_argument('--no_backtest', action='store_true', help='如果设置此项，则不执行回测分析')
    parser.add_argument('--predefined', action='store_true', help='强制使用预定义列表，即使提供了--ticker')
    
    args = parser.parse_args()
    
    # 调用主函数执行流程
    main(
        ticker=args.ticker, 
        news_file=args.news_file, 
        price_file=args.price_file, 
        start_date=args.start_date, 
        end_date=args.end_date, 
        backtest=not args.no_backtest,
        predefined=args.predefined
    ) 
# 股票预测情感舆情分析系统
# 特朗普行为预测系统

这是一个使用VADER(Valence Aware Dictionary and sEntiment Reasoner)模型进行股票情感分析的工具。VADER是一个专为社交媒体和金融文本设计的情感分析工具，可以有效分析股票相关新闻和评论的情感倾向，并根据股票预测正确率反向推导特朗普行为预测

## 功能特点

- 使用VADER模型分析金融新闻和社交媒体内容的情感倾向
- 增强了VADER词典，添加了金融领域特定词汇
- 支持批量分析新闻文本并计算情感得分
- 聚合每日情感数据并计算情感指标(情感得分、波动性、动量)
- 将情感数据与股票价格数据合并分析
- 生成可视化图表展示情感与股价的关系
- 分析情感数据对未来股价变动的影响
- **根据情感得分生成交易信号**
- **回测基于情感的交易策略**
- **自动从Yahoo Finance获取股票新闻数据**

## 安装依赖

```bash
pip install nltk pandas numpy matplotlib seaborn yahoo_fin requests-html
```

确保安装完NLTK后下载VADER词典：

```python
import nltk
nltk.download('vader_lexicon')
```

## 使用方法

### 基本使用

```python
from stock_sentiment_analyzer import StockSentimentAnalyzer

# 初始化分析器
analyzer = StockSentimentAnalyzer()

# 分析单条文本
text = "公司宣布季度盈利超出市场预期，并提高全年业绩展望。"
sentiment = analyzer.analyze_text(text)
print(sentiment)  # 输出: {'compound': 0.7, 'pos': 0.5, 'neu': 0.5, 'neg': 0.0}
```

### 使用命令行

您也可以通过命令行直接运行：

```bash
python stock_sentiment_analyzer.py --ticker AAPL --news_file data/news/apple_news.csv --price_file data/price/AAPL.csv --start_date 2023-01-01 --end_date 2023-12-31
```

参数说明：
- `--ticker`: 股票代码（必需）
- `--news_file`: 新闻数据文件路径（CSV或JSON格式）
- `--price_file`: 股票价格数据文件路径（CSV格式）
- `--start_date`: 分析起始日期（可选）
- `--end_date`: 分析结束日期（可选）
- `--no_backtest`: 不进行回测分析（可选）

## 获取股票新闻数据

系统提供了自动从Yahoo Finance获取股票新闻数据的功能，使用`process_stock_sentiment.py`脚本：

```bash
python process_stock_sentiment.py --ticker AAPL --days 30 --max 100
```

参数说明：
- `--ticker`: 单个股票代码
- `--tickers`: 多个股票代码，用逗号分隔（如 "AAPL,MSFT,GOOG"）
- `--market`: 获取市场整体新闻（S&P 500指数）
- `--sp500`: 获取标普500成分股新闻
- `--top`: 获取标普500前N大公司的新闻（默认10家）
- `--sector`: 获取特定行业板块的新闻，如Technology, Healthcare等
- `--days`: 获取最近多少天的新闻（默认30天）
- `--max`: 每个股票最大获取的文章数（默认100）
- `--output`: 输出文件名（默认自动生成）
- `--output-dir`: 输出目录（默认./data/news）

获取的新闻数据将保存在`data/news`目录下，可以直接用于情感分析。

## 数据格式要求

### 新闻数据

新闻数据文件（CSV或JSON）应包含以下字段：
- `date`: 新闻发布日期（YYYY-MM-DD或YYYY-MM-DD HH:MM:SS）
- `text`或`title`+`summary`: 新闻内容或标题和摘要

### 价格数据

价格数据文件（CSV）应包含以下字段：
- `date`: 交易日期（YYYY-MM-DD）
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量

## 输出结果

程序将生成以下文件：
- `{ticker}_sentiment_data.csv`: 每日情感得分数据
- `{ticker}_with_sentiment_signals.csv`: 价格数据与情感数据和交易信号合并结果
- `{ticker}_sentiment_vs_price.png`: 情感得分与股价对比图
- `{ticker}_sentiment_metrics.png`: 情感指标分析图
- `{ticker}_trading_signals.png`: 交易信号可视化图
- `{ticker}_backtest_results.png`: 策略回测结果图
- `{ticker}_backtest_summary.csv`: 回测汇总指标
- `{ticker}_trades.csv`: 详细交易记录

## 交易信号规则

系统根据情感得分生成交易信号，主要采用以下规则：

1. **正向情绪强度**：当情感得分高于阈值时，生成买入信号。
2. **负向情绪警报**：当情感得分低于阈值时，生成卖出信号。
3. **情感动量突变**：当情感得分快速从正转负或从负转正时，分别生成卖出或买入信号。
4. **异常强烈情感**：当情感得分远高于或远低于平均水平（超过2个标准差）时，分别生成买入或卖出信号。

交易信号以1（买入）、-1（卖出）和0（无操作）三种形式表示。

## 回测策略

系统使用生成的交易信号进行回测，回测策略基于以下原则：

- 仅在信号变化时执行交易（避免连续的相同信号重复交易）
- 买入信号时，使用90%可用资金进行买入（考虑交易佣金）
- 卖出信号时，清空全部持仓
- 计算每笔交易的收益和总体收益率
- 分析胜率（盈利交易占比）

## 完整分析流程

使用该系统进行完整的股票情感分析流程如下：

1. 获取股票新闻数据：
   ```bash
   python process_stock_sentiment.py --ticker AAPL --days 60
   ```

2. 下载股票价格数据（可以使用任何来源，如Yahoo Finance）

3. 运行情感分析和交易信号生成：
   ```bash
   python stock_sentiment_analyzer.py --ticker AAPL --news_file data/news/AAPL_news_20230601.csv --price_file data/price/AAPL.csv
   ```

4. 查看生成的结果和图表，分析情感与股价的关系以及交易策略的表现

## 案例分析

以苹果公司(AAPL)为例，我们可以分析2023年的新闻情感与股价的关系：

1. 收集苹果公司的相关新闻，保存为CSV格式
2. 获取苹果股票价格数据
3. 运行情感分析器分析新闻情感并与股价数据结合
4. 生成交易信号并回测交易策略
5. 观察情感变化是否能够有效指导交易决策

## 高级定制

您可以通过调整以下方面来定制模型：

1. 金融词典：在`_enhance_vader_lexicon`方法中添加更多金融专业词汇
2. 情感聚合方式：修改`aggregate_daily_sentiment`方法中的权重计算
3. 滞后效应分析：调整`analyze_sentiment_effect`中的`lag_days`参数
4. 交易信号阈值：修改`generate_trading_signals`方法中的各个阈值参数
5. 回测参数：调整`backtest_trading_strategy`方法中的初始资金和佣金率
6. 新闻获取：修改`process_stock_sentiment.py`中的参数来定制新闻获取方式

## 注意事项

- 情感分析结果仅供参考，不应作为投资决策的唯一依据
- 模型效果受数据质量和数量的影响，建议使用大量高质量的新闻数据
- 不同市场和不同行业的情感影响可能有所不同
- 回测结果不代表未来表现，实际交易中需考虑更多市场因素
- 交易策略较为简单，实际应用中可能需要与其他技术指标结合使用
- Yahoo Finance的API可能会有请求限制，频繁获取数据可能会导致暂时性封禁

# -*- coding: utf-8 -*-
"""
akshare 入门实例（优化版）
==========================
环境：Python 3.9+ + akshare + pandas + requests
安装：pip install akshare --upgrade pandas requests
运行：python akshare_quickstart_optimized.py
     可选参数：--code 600519 --start 2024-01-01 --end 2024-12-31
"""

import os
os.environ.setdefault("TQDM_DISABLE", "1")
#
import argparse
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import requests
import akshare as ak

# ----------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------
CONFIG = {
    "code": "600519",
    "start": "2025-01-01",
    "end": "2026-10-09",
    "adjust": "qfq",
    "output_dir": Path(__file__).resolve().parent,
}

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.unicode.east_asian_width", True)

# ----------------------------------------------------------------------
# 日志配置
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------
def title(text: str) -> None:
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def show(df, n: int = 3, note: str = "") -> None:
    """统一展示取数结果：形状 + 列名 + 前 n 行。"""
    if note:
        print(note)
    print(f"类型={type(df).__name__}  形状={getattr(df, 'shape', None)}")
    if isinstance(df, pd.DataFrame):
        print("列名:", list(df.columns)[:12])
        print(df.head(n).to_string())
    else:
        print(df)


def to_sina_symbol(code: str) -> str:
    """A 股代码 -> 新浪/腾讯风格代码：6/9 开头为沪市 sh，其余为深市 sz。"""
    return ("sh" if code.startswith(("6", "9")) else "sz") + code


def to_em_date(date_str: str) -> str:
    """将 YYYY-MM-DD 转为 YYYYMMDD"""
    return date_str.replace("-", "")


def to_sina_date(date_str: str) -> str:
    """将 YYYY-MM-DD 转为 YYYY-MM-DD（新浪要求带横线，此处仅作统一语义）"""
    # 如果已经是 YYYY-MM-DD 则直接返回；如果是 YYYYMMDD 则转换
    if "-" in date_str:
        return date_str
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """把不同数据源的中英文列名统一成 date/open/high/low/close/volume。"""
    rename = {
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount",
        "涨跌幅": "pct_chg", "换手率": "turnover",
    }
    df = df.rename(columns=rename).copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# 数据源降级获取
# ----------------------------------------------------------------------
# 定义网络相关异常，用于判断是否降级
NETWORK_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.RequestException,
)

def fetch_with_fallback(
    sources: list[tuple[str, Callable]],
    *args,
    **kwargs
) -> pd.DataFrame:
    """
    按顺序尝试多个数据源，返回第一个成功的结果。
    sources: [(源名称, 可调用对象), ...]
    """
    last_exc = None
    for name, func in sources:
        try:
            logger.info(f"尝试数据源：{name}")
            df = func(*args, **kwargs)
            if df is not None and not df.empty:
                logger.info(f"{name} 成功，共 {len(df)} 条")
                return df
            else:
                logger.warning(f"{name} 返回空数据")
        except Exception as exc:
            # 代码错误直接抛出，不降级
            if isinstance(exc, (KeyError, ValueError, TypeError)):
                raise
            logger.warning(f"{name} 失败：{type(exc).__name__}: {exc}")
            last_exc = exc
            time.sleep(1)  # 稍作等待再试下一个源
    raise RuntimeError(f"所有数据源均失败，最后错误：{last_exc}")


def fetch_daily(
    code: str = None,
    start: str = None,
    end: str = None,
    adjust: str = None,
) -> pd.DataFrame:
    """
    取 A 股日线，返回统一列名的 DataFrame。
    优先东方财富，失败降级到新浪。
    """
    code = code or CONFIG["code"]
    start = start or CONFIG["start"]
    end = end or CONFIG["end"]
    adjust = adjust or CONFIG["adjust"]

    sources = [
        (
            "东方财富 stock_zh_a_hist",
            lambda: ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=to_em_date(start),
                end_date=to_em_date(end),
                adjust=adjust,
            ),
        ),
        (
            "新浪 stock_zh_a_daily",
            lambda: ak.stock_zh_a_daily(
                symbol=to_sina_symbol(code),
                start_date=to_sina_date(start),
                end_date=to_sina_date(end),
                adjust=adjust,
            ),
        ),
    ]

    df = fetch_with_fallback(sources)
    return normalize(df)


# ----------------------------------------------------------------------
# 缓存获取（交易日历、股票列表）
# ----------------------------------------------------------------------
@lru_cache(maxsize=None)
def get_trade_cal() -> pd.DataFrame:
    logger.info("获取交易日历（缓存）")
    return ak.tool_trade_date_hist_sina()


@lru_cache(maxsize=None)
def get_a_code_name() -> pd.DataFrame:
    logger.info("获取 A 股代码表（缓存）")
    return ak.stock_info_a_code_name()


# ----------------------------------------------------------------------
# 第 1 章：环境与第一个例子
# ----------------------------------------------------------------------
def ch01_hello() -> None:
    title("第 1 章  环境检查与第一个例子")
    print("akshare 版本 :", ak.__version__)
    print("pandas  版本 :", pd.__version__)

    cal = get_trade_cal()
    print("\nA 股交易日历（最后 5 个交易日）:")
    print(cal.tail(5).to_string(index=False))


# ----------------------------------------------------------------------
# 第 2 章：拿到「股票池」——A 股代码表
# ----------------------------------------------------------------------
def ch02_stock_list() -> None:
    title("第 2 章  股票池：全部 A 股代码与名称")
    codes = get_a_code_name()
    show(codes, note="stock_info_a_code_name()")
    print(f"合计 {len(codes)} 只 A 股。")
    print("小技巧：先取代码表，再循环取行情，是做全市场扫描的第一步。")


# ----------------------------------------------------------------------
# 第 3 章：个股历史行情（最常用）
# ----------------------------------------------------------------------
def ch03_daily() -> None:
    title("第 3 章  个股日线行情（默认标的）")
    df = fetch_daily()
    show(df, note=f"前复权日线（qfq） {CONFIG['code']} {CONFIG['start']} ~ {CONFIG['end']}")

    print("\n复权口径说明：")
    for adjust, desc in [("", "不复权：真实成交价，除权日会跳空"),
                         ("qfq", "前复权：以最新价为基准向前调整，看历史涨跌幅最准"),
                         ("hfq", "后复权：以上市首日为基准，看长期真实收益")]:
        print(f"  adjust='{adjust}'  {desc}")

    df["pct_chg"] = df["close"].pct_change() * 100
    print("\n区间统计：")
    print(f"  区间涨跌幅 : {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.2f}%")
    print(f"  期间最高价 : {df['high'].max():.2f}")
    print(f"  期间最低价 : {df['low'].min():.2f}")
    print(f"  日均成交量 : {df['volume'].mean():,.0f} 股")


# ----------------------------------------------------------------------
# 第 4 章：指数行情
# ----------------------------------------------------------------------
def ch04_index() -> None:
    title("第 4 章  指数行情")
    idx = ak.stock_zh_index_daily(symbol="sh000001")     # 上证指数
    print("上证指数日线（新浪源，全历史）:")
    print(f"  共 {len(idx)} 条，最新日期 {idx['date'].iloc[-1]}")
    print(idx.tail(3).to_string(index=False))

    spot = ak.stock_zh_index_spot_sina()
    print("\n指数实时快照（部分）:")
    key = spot[spot["代码"].isin(["sh000001", "sz399001", "sz399006"])]
    print(key[["代码", "名称", "最新价", "涨跌幅"]].to_string(index=False))


# ----------------------------------------------------------------------
# 第 5 章：基本面 / 财务数据
# ----------------------------------------------------------------------
def ch05_fundamental() -> None:
    title("第 5 章  基本面：财务摘要与财务指标")
    abs_df = ak.stock_financial_abstract(symbol=CONFIG["code"])
    print("财务摘要 stock_financial_abstract() —— 宽表，列是报告期：")
    print(f"  形状 {abs_df.shape}，报告期列（前 6 个）: {list(abs_df.columns[2:8])}")
    uniq = abs_df.drop_duplicates(subset="指标")
    key_rows = [k for k in ["营业总收入", "归母净利润", "净资产收益率", "销售毛利率"]
                if k in set(uniq["指标"])]
    print(uniq[uniq["指标"].isin(key_rows)].iloc[:, :6].to_string(index=False))

    ind = ak.stock_financial_analysis_indicator(symbol=CONFIG["code"], start_year="2023")
    cols = ["日期", "摊薄每股收益(元)", "销售净利率(%)", "净资产收益率(%)"]
    cols = [c for c in cols if c in ind.columns]
    print("\n财务指标 stock_financial_analysis_indicator() —— 按报告期：")
    print(ind[cols].tail(6).to_string(index=False))


# ----------------------------------------------------------------------
# 第 6 章：ETF / 基金、港股、美股
# ----------------------------------------------------------------------
def ch06_multi_market() -> None:
    title("第 6 章  跨品种：ETF、港股、美股")
    etf = ak.fund_etf_hist_sina(symbol="sh510300")       # 沪深300ETF
    print(f"沪深300ETF(sh510300) 日线：{etf.shape[0]} 条，最后一日 {etf['date'].iloc[-1]}")
    print(etf.tail(2).to_string(index=False))

    hk = ak.stock_hk_daily(symbol="00700")               # 腾讯控股
    print(f"\n腾讯控股(00700) 日线：{hk.shape[0]} 条，最新收盘 {hk['close'].iloc[-1]}")
    print(hk.tail(2).to_string(index=False))

    us = ak.stock_us_daily(symbol="AAPL")                # 苹果
    print(f"\n苹果(AAPL) 日线：{us.shape[0]} 条，最新收盘 {us['close'].iloc[-1]}")


# ----------------------------------------------------------------------
# 第 7 章：宏观数据
# ----------------------------------------------------------------------
def ch07_macro() -> None:
    title("第 7 章  宏观数据")
    cpi = ak.macro_china_cpi_monthly()
    print("中国 CPI 月率（最后 3 期）:")
    print(cpi.tail(3).to_string(index=False))

    gdp = ak.macro_china_gdp_yearly()
    print("\n中国 GDP 年率（最后 3 期）:")
    print(gdp.tail(3).to_string(index=False))

    rate = ak.bond_zh_us_rate(start_date="20250901")
    cols = ["日期", "中国国债收益率10年", "美国国债收益率10年"]
    print("\n中美国债收益率（10 年期，最后 3 个交易日）:")
    print(rate[cols].tail(3).to_string(index=False))


# ----------------------------------------------------------------------
# 第 8 章：小实战——均线 + 区间指标 + 出图
# ----------------------------------------------------------------------
def ch08_practice() -> None:
    title("第 8 章  小实战：均线、区间指标与出图")
    df = fetch_daily()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ret"] = df["close"].pct_change()

    print("最近 3 个交易日：")
    print(df[["date", "close", "ma5", "ma20"]].tail(3).to_string(index=False))

    total_ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    cummax = df["close"].cummax()
    max_dd = (df["close"] / cummax - 1).min()
    ann_vol = df["ret"].std() * (250 ** 0.5)
    print(f"\n区间累计收益 : {total_ret * 100:6.2f}%")
    print(f"最大回撤     : {max_dd * 100:6.2f}%")
    print(f"年化波动率   : {ann_vol * 100:6.2f}%")

    # ---- 画图（可选，需要 matplotlib）----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # 更健壮的字体选择
        preferred_fonts = [
            "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei",
            "Source Han Sans CN", "Microsoft YaHei", "PingFang SC",
            "Hiragino Sans GB", "Droid Sans Fallback",
        ]
        available = {f.name for f in font_manager.fontManager.ttflist}
        chosen = None
        for name in preferred_fonts:
            if name in available:
                chosen = name
                break
        if chosen:
            plt.rcParams["font.sans-serif"] = [chosen]
            plt.rcParams["axes.unicode_minus"] = False
            logger.info(f"使用中文字体：{chosen}")
        else:
            logger.warning("未找到中文字体，图表中文可能显示为方块")

        fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
        ax.plot(df["date"], df["close"], lw=1.2, label="收盘价")
        ax.plot(df["date"], df["ma5"], lw=1.0, label="MA5")
        ax.plot(df["date"], df["ma20"], lw=1.0, label="MA20")
        ax.set_title(f"{CONFIG['code']} 前复权收盘价与均线")
        ax.set_ylabel("价格（元）")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.autofmt_xdate()
        CONFIG["output_dir"].mkdir(exist_ok=True)
        png = CONFIG["output_dir"] / "ma_line.png"
        fig.savefig(png, bbox_inches="tight")
        plt.close(fig)
        print(f"\n图表已保存：{png}")
    except Exception as exc:
        print(f"\n[提示] 跳过画图（{type(exc).__name__}: {exc}）")


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
CHAPTERS = [
    ch01_hello, ch02_stock_list, ch03_daily, ch04_index,
    ch05_fundamental, ch06_multi_market, ch07_macro, ch08_practice,
]


def parse_args():
    parser = argparse.ArgumentParser(description="akshare 入门实例")
    parser.add_argument("--code", default=CONFIG["code"], help="股票代码，如 600519")
    parser.add_argument("--start", default=CONFIG["start"], help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=CONFIG["end"], help="结束日期 YYYY-MM-DD")
    parser.add_argument("--adjust", default=CONFIG["adjust"], choices=["", "qfq", "hfq"],
                        help="复权方式：'' 不复权，qfq 前复权，hfq 后复权")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    CONFIG.update({
        "code": args.code,
        "start": args.start,
        "end": args.end,
        "adjust": args.adjust,
    })
    logger.info(f"运行配置：{CONFIG['code']} {CONFIG['start']} ~ {CONFIG['end']} 复权={CONFIG['adjust']}")

    print("akshare 入门实例开始运行 —— 共", len(CHAPTERS), "章")
    for fn in CHAPTERS:
        try:
            fn()
        except Exception as exc:
            logger.exception(f"章节 {fn.__name__} 执行失败")
    print("\n全部章节执行结束。")


if __name__ == "__main__":
    main()
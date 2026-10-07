"""
株テクニカル・アラート（チェック本体）
config.json の条件で銘柄をチェックし、成立したら LINE に通知します。
GitHub Actions から定期実行される想定です。
"""
import json
import os
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

JST = timezone(timedelta(hours=9))
CONFIG_PATH = "config.json"
STATE_PATH = "state.json"


# ---------------------------------------------------------------- 指標計算
def sma(s, n):
    return s.rolling(n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close, n=14):
    diff = close.diff()
    up = wilder(diff.clip(lower=0), n)
    down = wilder(-diff.clip(upper=0), n)
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def macd(close, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    return line, ema(line, signal)


def dmi(high, low, close, n=14):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = wilder(tr, n)
    plus_di = 100 * wilder(pd.Series(plus_dm, index=close.index), n) / atr
    minus_di = 100 * wilder(pd.Series(minus_dm, index=close.index), n) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = wilder(dx.fillna(0), n)
    return plus_di, minus_di, adx


def crossed(a, b, direction, within):
    """直近 within 本以内に a が b を上抜け(golden)/下抜け(dead)したか"""
    above = a > b
    if direction == "golden":
        ev = above & ~above.shift(1, fill_value=True)
    else:
        ev = ~above & above.shift(1, fill_value=False)
    return bool(ev.iloc[-within:].any())


# ---------------------------------------------------------------- 条件判定
def compare(x, op, v):
    return x > v if op == ">" else x < v


def check_condition(df, c):
    """1つの条件を判定し (成立したか, 説明文) を返す"""
    close, high, low = df["Close"], df["High"], df["Low"]
    t = c["type"]
    within = int(c.get("within", 1))

    if t == "ma_cross":
        s, l = sma(close, c["short"]), sma(close, c["long"])
        ok = crossed(s, l, c["direction"], within)
        label = "ゴールデンクロス" if c["direction"] == "golden" else "デッドクロス"
        return ok, f"MA{c['short']}/{c['long']} {label}"

    if t == "ma_position":  # 短期線が長期線の上/下にあるか
        s, l = sma(close, c["short"]).iloc[-1], sma(close, c["long"]).iloc[-1]
        ok = s > l if c["position"] == "above" else s < l
        return ok, f"MA{c['short']}が{c['long']}の{'上' if c['position']=='above' else '下'}"

    if t == "rsi":
        v = rsi(close, c.get("period", 14)).iloc[-1]
        return compare(v, c["op"], c["value"]), f"RSI {v:.1f} ({c['op']}{c['value']})"

    if t == "macd":
        line, sig = macd(close, c.get("fast", 12), c.get("slow", 26), c.get("signal", 9))
        ok = crossed(line, sig, c["direction"], within)
        label = "シグナル上抜け" if c["direction"] == "golden" else "シグナル下抜け"
        return ok, f"MACD {label}"

    if t == "dmi":
        p, m, adx = dmi(high, low, close, c.get("period", 14))
        p, m, a = p.iloc[-1], m.iloc[-1], adx.iloc[-1]
        trend_ok = p > m if c.get("trend", "plus") == "plus" else m > p
        ok = trend_ok and a >= c.get("adx_min", 25)
        return ok, f"DMI +DI {p:.1f} / -DI {m:.1f} / ADX {a:.1f}"

    raise ValueError(f"未知の条件タイプ: {t}")


def evaluate_rule(df, rule):
    results = [check_condition(df, c) for c in rule["conditions"]]
    flags = [ok for ok, _ in results]
    mode = rule.get("mode", "all")
    if mode == "all":
        hit = all(flags)
    elif mode == "none":
        hit = not any(flags)
    else:  # "any"
        hit = any(flags)
    return hit, results


# ---------------------------------------------------------------- 通知
def notify_line(text):
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    user = os.environ.get("LINE_USER_ID")
    if not token or not user:
        print("[LINE未設定] 送信内容:\n" + text)
        return
    r = requests.post(
        "https://api.line.me/v2/bot/message/push",
        headers={"Authorization": f"Bearer {token}"},
        json={"to": user, "messages": [{"type": "text", "text": text[:4900]}]},
        timeout=20,
    )
    print("LINE送信:", r.status_code, r.text[:200])


# ---------------------------------------------------------------- メイン
def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def fetch(ticker, interval, period):
    import yfinance as yf

    df = yf.download(ticker, interval=interval, period=period,
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def main():
    config = load_json(CONFIG_PATH, {})
    state = load_json(STATE_PATH, {})
    interval = config.get("interval", "1d")
    period = config.get("period", "1y")
    now = datetime.now(JST).strftime("%Y-%m-%d %H:%M")
    messages = []

    for ticker in config.get("tickers", []):
        try:
            df = fetch(ticker, interval, period)
            if len(df) < 60:
                print(f"{ticker}: データ不足")
                continue
        except Exception as e:
            print(f"{ticker}: 取得失敗 {e}")
            continue

        for rule in config.get("rules", []):
            if not rule.get("enabled", True):
                continue
            key = f"{ticker}|{rule['name']}"
            hit, results = evaluate_rule(df, rule)
            print(f"{key}: {'成立' if hit else '不成立'}")

            # 不成立→成立に変わった時だけ通知（同じシグナルを何度も送らない）
            if hit and not state.get(key):
                lines = [f"🔔 {ticker}【{rule['name']}】",
                         f"終値 {df['Close'].iloc[-1]:,.2f}"]
                lines += [f"{'✅' if ok else '❌'} {d}" for ok, d in results]
                messages.append("\n".join(lines))
            state[key] = hit

    if messages:
        notify_line(f"株アラート {now}\n\n" + "\n\n".join(messages))

    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())

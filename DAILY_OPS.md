# 每日操作手册

## 你需要准备的东西

- Binance 永续合约数据源（OHLCV + OI + funding_rate，至少最近 48 根 15m bar）
- 81 个币种：BTC,ETH,BNB,SOL,XRP,ADA,SUI,DOT,LINK,AVAX,FIL + rank 20-100 的中小币
- Python 环境，能调你的交易所 API（但不允许下单）

## 每 15 分钟你要做的事

### 1. 拉取最新数据

取 81 个币的最新一根 15m bar，包括：
- open, high, low, close, volume
- open_interest
- funding_rate

### 2. 计算基础特征（每币）

用最近 48 根 bar 算这些：

| 特征 | 算法 | 用途 |
|---|---|---|
| return_1h | close 比 4 根前的涨跌幅 | A1,A3 |
| oi_delta_z | OI 6 根变化 / 48 根标准差 | A1,A2,A3,A4 |
| volume_z | volume / 48 根均值标准差 | A1,A2,A3 |
| close_location | (close-low)/(high-low) | A1,A2,A3 |
| funding_z | funding_rate / 24 根均值标准差 | A1,A3,A4 |
| relative_return | symbol_ret_1h - btc_ret_1h | A3 |
| VWAP_4h | 16 根量价加权均价 | A1 延持判断 |

### 3. 判 BTC regime

用 BTC 的价格、OI、波动率判断当前市场状态：
- trend_up：BTC EMA 向上 + 多数币涨
- trend_down：BTC EMA 向下 + 多数币跌
- range：横盘
- panic_down：BTC 急跌 >2 倍标准差 + 高波动 + OI 崩塌
- chop：高波动但无方向

### 4. 跑 4 个 alpha 的条件判断

**Alpha 1 — Deleveraging Reversal**

如果下面全部满足 → 做多，持仓 30 分钟：
- return_1h 在 -2.5% 到 -12% 之间
- oi_delta_z 在 -3.0 到 -2.0 之间
- volume_z > 1.0
- close_location > 0.35
- BTC regime 不是 panic_down 也不是 chop
- |funding_z| < 2.0

出场：
- 30 分钟后平仓
- 如果 30 分钟时 close > VWAP_4h 且 oi_delta_z > -1.0 且 volume_z < 2.0 → 延长到 60 分钟
- 如果入场后 15 分钟价格逆向 >1% → 立即平仓

**Alpha 2 — OI Shock Absorption**

如果下面全部满足 → 做多（禁止做空），持仓 30 分钟：
- oi_delta_z > 2.5
- volume_z > 1.5
- close_location < 0.4
- BTC regime 不是 panic_down 也不是 chop

出场：30 分钟后平仓

**Alpha 3 — RS Shock**

如果下面全部满足 → 按 regime 定方向：
- relative_return > 5%（币比 BTC 强 5%以上）
- volume_z > 2.0
- oi_delta_z > 0.5
- close_location > 0.50
- |funding_z| < 2.5
- btc_ret_1h > -1%

方向规则：
- trend_down → 做空
- range → 做多（需 regime 已持续 4 根 bar 以上）
- trend_up / panic_down / chop → 不交易

出场：30 分钟后平仓

**Alpha 4 — Funding Carry EU**

只在 UTC 7:00-16:59 运行。不在这个时段跳过。

如果下面全部满足 → 做多，持仓 3 小时：
- funding_z < -2.5（资金费率极低 = 空头拥挤）
- oi_delta_z > -2.0（OI 不能正在崩塌）
- BTC regime 不是 panic_down

出场：
- 3 小时后平仓
- 如 BTC 进入 panic_down → 立即平仓

注意：这个 alpha 的收益主要来自资金费率（做多收钱），不是价格波动。funding_rate 为负值时做多方收钱。

### 5. 冲突处理

如果同一币种同一时间触发多个 alpha：

| 情况 | 处理 |
|---|---|
| 方向相同 | 只做优先级最高的（Deleveraging > OI Shock > RS Shock > FC） |
| 方向相反 | 全部放弃 |
| FC + 任何价格事件同向 | 不叠加仓位，只记录 |

### 6. 记录日志

每条信号记到 JSONL：
```json
{"timestamp":"...","symbol":"...","event_type":"...","direction":"...","hold_bars":...,"status":"SHADOW_SIGNAL","reject_reason":""}
```

被拒绝的也记：
```json
{"timestamp":"...","symbol":"...","event_type":"...","status":"REJECTED","reject_reason":"oi_delta_z > -1.5"}
```

## 每天收盘后你要做的事

生成 `portfolio_report_YYYYMMDD.json`，包含：

```
当日扫描次数
各 alpha 的信号数
模拟交易明细（入场价、出场价、盈亏 bps）
扣 9/12/15 bps 成本后的净盈亏
各 alpha 的 PF（盈利总额/亏损总额）
胜率
top3 单笔贡献（去掉后还剩多少）
top5 币种贡献
Funding Carry EU 的 funding_pnl vs price_pnl 拆分
重叠/冲突统计
```

## 验证方法

第一次跑的时候，用时间戳 `2026-06-04 01:00:00 UTC` 验证：
- 你应该只输出 2 个做多信号：RAVEUSDT 和 SPKUSDT
- 都是 Deleveraging Reversal
- 0 个 Funding Carry EU（因为 01:00 不在 EU 时段）
- 如果结果不同 → 你的条件判断有问题，对照上面重新检查

## 绝对不能做的事

1. 不下单 — 所有信号只记录不执行
2. 不临时改阈值 — 比如"volume_z > 1.0 太严了改 0.8"
3. OI Shock 做空 — 这个 alpha 写死了只做多
4. Funding Carry 在非 EU 时段触发 — session gate 必须生效
5. 让 AI 替你决定要不要交易 — 所有判断是写死的条件

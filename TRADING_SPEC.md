# Alpha Portfolio Shadow — 交易端规则说明书

## 总则
- 所有信号 shadow-only，不下单
- live_allowed = false
- 不允许 AI 或程序自动修改规则
- 每日输出组合级 shadow report

---

## Universe: rank 20-100 (81个USDT永续)
静态列表见 `registry/live/event_universe.json`

---

## Alpha 1: Deleveraging Reversal V1.2

### 触发条件（所有条件必须同时满足）
- return_1h < -2.5% (4根15m bar)
- return_1h > -12% (排除黑天鹅)
- oi_delta_z 在 (-3.0, -2.0] 区间 (钟形评分, 中心=-2.5)
- volume_z > 1.0 且 < 8.0
- BTC regime != panic_down 且 != chop
- |funding_z| < 2.0 (per-symbol)
- 同币种: marketwide_oi_collapse_count <= 5

### 入场
- direction: long
- close_location > 0.35 (价格不在bar最低点)
- regime阈值: trend_down=0.50, range=0.60, trend_up=0.75

### 出场
- 标准: hold=2 bars (30分钟)
- 延持条件(hold→4): 出场时 close>VWAP_4h 且 oi_delta_z > -1.0 且 volume_z < 2.0
- 提前止损(bar+1): 入场后1bar价格逆向>1%

### cooldown
- 16 bars (4小时) per symbol

---

## Alpha 2: OI Shock Absorption V1.0

### 触发条件
- oi_delta_z > 2.5 (OI暴增)
- volume_z > 1.5
- close_loc < 0.4 (价格在bar低位 = 空头被套)
- abs(ret_15m) < 1% 或 abs(ret_1h) < 2% (价格未继续跌)
- BTC regime != panic_down 且 != chop

### 入场
- direction: long ONLY
- symmetric_trading 禁止, allow_short_side = false

### 出场
- hold=2 bars

### cooldown
- 8 bars per symbol

---

## Alpha 3: RS Shock V1.0

### 触发条件
- relative_return_vs_BTC > 5% (symbol_ret_1h - btc_ret_1h)
- volume_z > 2.0
- oi_delta_z > 0.5
- close_location > 0.50
- |funding_z| < 2.5
- event_score > 0.60
- btc_ret_1h > -1%

### 方向规则(regime-dependent)
- trend_down: short (弱势延续)
- range: long (regime_confidence >= 1 才允许)
- trend_up: skip
- panic_down: skip
- chop: skip
- unknown: skip

### 出场
- hold=2 bars

### cooldown
- 8 bars per symbol

---

## Alpha 4: Funding Carry EU V1.0

### 触发条件
- funding_z < -2.5 (极端负funding = 空头拥挤)
- UTC小时 >= 7 且 < 17 (EU session)
- BTC regime != panic_down
- oi_delta_z > -2.0 (OI不能正在崩塌)

### 入场
- direction: long (收funding费)
- 每4根bar扫描一次 (1小时间隔)，避免重复入场

### 出场
- hold=12 bars (3小时)
- 或BTC进入panic_down 立即出场

### 收益计算
- funding_pnl = -avg(funding_rate) × 10000 × (持仓小时/8)
- price_pnl = (exit_price / entry_price - 1) × 10000
- total_pnl = price_pnl + funding_pnl
- funding_rate为负时long方收钱，上面公式已处理符号

### cooldown
- 16 bars per symbol

---

## 冲突解决规则（写死,不可自动改）

优先级: Deleveraging > OI Shock > RS Shock > Funding Carry

| 情况 | 规则 |
|---|---|
| 同币同时, 同方向, 多个alpha | 保留最高优先级, 其余标记MERGED |
| 同币同时, 反方向 | 全部REJECT, 标记DIRECTION_CONFLICT |
| Funding Carry + 价格事件 同向 | 保留优先级高的, 记录overlap, 不加仓 |
| Funding Carry + 价格事件 反向 | 全部拒绝 |

---

## 每日组合报告

输出 `portfolio_report_YYYYMMDD.json`, 包含:

```json
{
  "scan_ts": "扫描时间",
  "regime": "当前BTC regime",
  "portfolio": {
    "total_raw": "冲突前信号总数",
    "after_conflict": "冲突后信号数",
    "conflicts": "冲突数",
    "rejected": "拒绝数"
  },
  "by_alpha": { "DeleveragingReversal": N, "OIShockAbsorption": N, "RelativeStrengthShock": N, "FundingCarryEU": N },
  "overlap_summary": { "重叠对": 次数 },
  "cumulative": {
    "total_trades": "累计交易数",
    "net_bps_9": "累计净收益@9bps",
    "net_bps_12": "累计净收益@12bps",
    "net_bps_15": "累计净收益@15bps",
    "pf": "组合PF",
    "by_alpha_contribution": { "各alpha贡献" },
    "top3_trades_pct": "top3占比",
    "top5_symbols_pct": "top5币种占比",
    "max_daily_loss": "最大单日亏损"
  }
}
```

---

## 文件依赖

| 文件 | 说明 |
|---|---|
| registry/factor_registry.json | 4个alpha完整定义 |
| registry/live/live_config.json | 交易端只读配置 |
| registry/live/event_universe.json | 79币 + 11 regime锚 |
| shadow_config.json | shadow模式配置 |

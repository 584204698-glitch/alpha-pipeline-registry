# Registry Handoff — 交易端 Hermes 对接指南

## 核心文件

```
/mnt/e/alpha_pipeline/registry/
├── factor_registry.json          ← 因子全生命周期 (dev→paper→shadow→live)
├── pending_review.json           ← 研究端提审，等你批准
├── live/live_config.json         ← 交易端 ONLY 读取这个
├── live/trade_feedback.jsonl     ← 交易端写回交易结果
├── approvals/approvals.log       ← 人类审批记录
└── deprecated/                   ← 已杀因子存档
```

## 交易端 Hermes 规则（硬编码，不允许 AI 改写）

```python
# 1. 只读 live_config.json
live_config = read_json("registry/live/live_config.json")

# 2. 安全检查 — 必须全通过
assert live_config["allow_ai_override"] == False     # 不允许 AI 绕过
assert live_config["block_unapproved_factors"] == True  # 未审批因子禁止交易

# 3. 只交易 enabled_factors 里的因子
for factor_id in live_config["enabled_factors"]:
    # 允许交易
    pass

# 4. 任何不在 enabled_factors 的信号 → 直接拒绝
if factor_id not in live_config["enabled_factors"]:
    print(f"FACTOR_NOT_APPROVED:{factor_id}")
    return  # DO NOT TRADE
```

## 审批流程（只有你能做）

```
研究 Hermes: 写 pending_review.json → "FPR_DirStrengthSplit: promote_to_shadow"
      ↓
你审核: 看 factor_report.json + paper_battery.json
      ↓
你批准: 手动改 live_config.json → enabled_factors += "FPR_DirStrengthSplit"
      ↓
交易 Hermes: 下次 git pull → 自动开始 shadow 模式
```

## 当前状态

```
enabled_factors: []    ← 空！所有因子被阻止交易
mode: "disabled"       ← 初始模式
```

## 你需要做的

1. 拿到 paper_trading_battery.json 报告
2. 选出你想 shadow 测试的因子
3. 改 live_config.json 的 enabled_factors
4. 交易端 Hermes 拉代码即可开始

# 新因子假设卡片 — 基于7个优先研究方向
# 所有因子默认判定: idea_only，需经完整晋级流程

hypotheses = [
    {
        "factor_name": "OIFailureToAdvance",
        "factor_family": "OI_accumulation_failure",
        "role": "idea_only",
        "hypothesis": "OI持续累积但价格无法同向推进 → 新仓位正在被吸收/反向力量在积累 → 价格可能反转",
        "market_mechanism": "当OI连续增加但价格位移递减，说明新资金推动力在衰减，对手方正逐步吸收",
        "who_loses_money": "追趋势的交易者，他们看到OI上升以为趋势确认，但实际是聪明钱在反向建仓",
        "why_edge_may_persist": "OI数据在加密领域公开但多数人只看方向不看效率，OI-价格效率比是二阶信号",
        "expected_holding_period": "3-6 bar (1h)",
        "expected_turnover": "低 — 只在效率显著恶化时触发",
        "expected_cost_sensitivity": "中等 — 事件触发型，非每bar调仓",
        "required_fields": ["open_interest", "close"],
        "direction": "fade (反向交易OI方向)",
        "allowed_regimes": ["trend_mature", "range_bound"],
        "forbidden_regimes": ["panic", "strong_trend_initial"],
        "formula_sketch": "sign(-ΔOI) * (1 - |Δprice|/max_Δprice) → OI增但价格不动 → 做空",
        "parameter_neighbor_test_plan": "OI_lookback 6/9/12, price_window 1/3/6/12",
        "paper_trading_plan": "threshold_entry z=1.5 hold>=3, top/bottom 5-10%",
        "promotion_criteria": "Gross>0, Net PnL improvement over existing, C/G<300%",
        "kill_criteria": "Gross<0, 或与CrowdingFade/VolAccelFade的相关性>0.5"
    },
    {
        "factor_name": "DeleveragingSnapback",
        "factor_family": "deleveraging_reversal",
        "role": "idea_only",
        "hypothesis": "大跌+OI大降+放量 → 强制平仓潮结束 → 短期反弹。大赚后+OI大降+放量 → 获利了结 → 短期回落",
        "market_mechanism": "OI骤降+放量说明大量持仓被清算或平仓。急跌后清算释放卖方压力→反弹。急涨后获利了结→回落",
        "who_loses_money": "被强制平仓的多头（跌后反弹踏空）或追高被套的空头（涨后回落踏空）",
        "why_edge_may_persist": "强制平仓是机械性事件，清算完成后价格常常超调回归，这是微观结构效应非基本面",
        "expected_holding_period": "2-4 bar (1h)",
        "expected_turnover": "极低 — 事件触发，只在OI shock时入场",
        "expected_cost_sensitivity": "低 — 事件少，入场次数有限",
        "required_fields": ["open_interest", "close", "volume", "taker_buy_volume", "taker_sell_volume"],
        "direction": "fade (急跌后做多，急涨后做空)",
        "allowed_regimes": ["panic_down", "squeeze_up"],
        "forbidden_regimes": ["range", "chop"],
        "formula_sketch": "Δprice<−3σ AND ΔOI<−2σ AND vol>2σ → long; 对称做short",
        "parameter_neighbor_test_plan": "price_threshold 2σ/2.5σ/3σ, OI_threshold 1.5σ/2σ/2.5σ",
        "paper_trading_plan": "事件触发入场，hold 2/3/4 bar后退出",
        "promotion_criteria": "Net>0, PF>1.1, n_trades>30",
        "kill_criteria": "Gross<0, 或事件频率>每10bar一次(太多噪音信号)"
    },
    {
        "factor_name": "FundingTrapFade",
        "factor_family": "funding_pressure_failure",
        "role": "idea_only",
        "hypothesis": "funding极端但价格不配合 → 付费方正在失败 → 价格将向付费方反方向移动",
        "market_mechanism": "正funding极高(多头付费)但价格不涨→多头拥挤但没新钱推了→即将反转。负funding极低但价格不跌→空头拥挤但卖盘枯竭→即将反弹",
        "who_loses_money": "支付高funding的拥挤方，他们为持仓付费但价格不配合，最终被迫平仓",
        "why_edge_may_persist": "funding支付是显性成本，拥挤方承受时间衰减。价格不配合说明边际资金已耗尽",
        "expected_holding_period": "3-6 bar (1h)",
        "expected_turnover": "低 — 只在funding极端+价格停滞时触发",
        "expected_cost_sensitivity": "低",
        "required_fields": ["funding_rate", "close", "open_interest"],
        "direction": "fade",
        "allowed_regimes": ["funding_extreme", "range"],
        "forbidden_regimes": ["strong_trend_with_funding"],
        "formula_sketch": "sign(-funding) * |funding_z| * (1 - |price_displacement_ratio|) → funding极端但价格不动",
        "parameter_neighbor_test_plan": "funding_lookback 12/24/48, price_window 1/3/6",
        "paper_trading_plan": "threshold_entry z=1.5, hold>=3",
        "promotion_criteria": "Net>0, PF>1.1, C/G<200%",
        "kill_criteria": "Gross<0 或和现有FundingPressureRelease相关性>0.6"
    }
]

if __name__ == "__main__":
    import json
    print(json.dumps(hypotheses, indent=2, ensure_ascii=False))

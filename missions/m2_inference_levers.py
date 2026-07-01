"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
CACHE_WRITE_COST_PER_M = 0.50  # illustrative storage/write fee ($/1M cached tokens)


def _avg_cache_reads(rows) -> float:
    """Mean requests per (team, project) that reuse a cached prefix."""
    groups: dict = defaultdict(int)
    for r in rows:
        if int(num(r["cached_input_tokens"])) > 0:
            groups[(r["team"], r["project"])] += 1
    return sum(groups.values()) / len(groups) if groups else 0.0


def _reasoning_stats(rows) -> dict:
    n = len(rows)
    reasoning = [r for r in rows if bool(int(num(r["is_reasoning"])))]
    normal = [r for r in rows if not bool(int(num(r["is_reasoning"])))]
    r_cost = n_cost = r_wh = n_wh = 0.0
    for r in reasoning:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        r_cost += pricing.request_cost(inp, out, pin, pout,
                                       cached_in=int(num(r["cached_input_tokens"])),
                                       batch=bool(int(num(r["is_batch"]))))
        r_wh += sustainability.wh_per_query(inp + out, is_reasoning=True)
    for r in normal:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        n_cost += pricing.request_cost(inp, out, pin, pout,
                                       cached_in=int(num(r["cached_input_tokens"])),
                                       batch=bool(int(num(r["is_batch"]))))
        n_wh += sustainability.wh_per_query(inp + out, is_reasoning=False)
    total_cost = r_cost + n_cost
    return {
        "reasoning_requests": len(reasoning),
        "reasoning_pct_traffic": len(reasoning) / n * 100 if n else 0.0,
        "reasoning_cost": round(r_cost, 2),
        "normal_cost": round(n_cost, 2),
        "reasoning_pct_cost": r_cost / total_cost * 100 if total_cost else 0.0,
        "reasoning_wh": round(r_wh, 2),
        "normal_wh": round(n_wh, 2),
        "reasoning_pct_wh": r_wh / (r_wh + n_wh) * 100 if (r_wh + n_wh) else 0.0,
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    avg_reads = _avg_cache_reads(rows)
    be_reads = pricing.cache_break_even_reads(CACHE_WRITE_COST_PER_M)
    use_cache = pricing.cache_is_worth_it(avg_reads, CACHE_WRITE_COST_PER_M)
    reasoning = _reasoning_stats(rows)

    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"])) if use_cache else 0
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print(f"\n[Ext3] cache break-even reads: {be_reads:.2f}  avg reads/prefix: {avg_reads:.1f}  apply cache? {use_cache}")
        print(f"[Ext4] reasoning: {reasoning['reasoning_requests']} reqs ({reasoning['reasoning_pct_traffic']:.1f}% traffic)")
        print(f"       cost ${reasoning['reasoning_cost']:.2f} ({reasoning['reasoning_pct_cost']:.1f}% of ${reasoning['reasoning_cost']+reasoning['normal_cost']:.2f})")
        print(f"       energy {reasoning['reasoning_wh']:.1f} Wh ({reasoning['reasoning_pct_wh']:.1f}% of total Wh)")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_worth_it": use_cache, "avg_cache_reads": round(avg_reads, 1),
        "reasoning": reasoning,
    }


if __name__ == "__main__":
    run()

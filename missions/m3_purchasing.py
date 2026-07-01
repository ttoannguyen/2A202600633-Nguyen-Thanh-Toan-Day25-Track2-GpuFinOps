"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30
CARBON_BASE_REGION = "us-east-1"


def _job_energy_wh(gpu_hours: float, watts: float) -> float:
    return gpu_hours * watts


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    carbon_rows = []
    best_region = min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

        if interruptible:
            wh = _job_energy_wh(gpu_hours, num(c["watts"]))
            c_base = sustainability.carbon_g(wh, CARBON_BASE_REGION)
            c_clean = sustainability.carbon_g(wh, best_region)
            carbon_rows.append({
                "job_id": j["job_id"], "wh": round(wh, 1),
                "carbon_base_g": round(c_base, 1), "carbon_clean_g": round(c_clean, 1),
                "saved_g": round(c_base - c_clean, 1),
            })

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        if carbon_rows:
            total_saved = sum(r["saved_g"] for r in carbon_rows)
            print(f"\n[Ext5] Carbon-aware scheduling ({CARBON_BASE_REGION} -> {best_region}):")
            for cr in carbon_rows:
                pct = (1 - cr["carbon_clean_g"] / cr["carbon_base_g"]) * 100 if cr["carbon_base_g"] else 0
                print(f"  {cr['job_id']:18} {cr['wh']:>10.0f} Wh  "
                      f"{cr['carbon_base_g']:>8.0f} -> {cr['carbon_clean_g']:>6.0f} gCO2e  (-{pct:.0f}%)")
            print(f"  total carbon saved: {total_saved:,.0f} gCO2e/month if interruptible jobs move region")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "carbon_savings_g": round(sum(r["saved_g"] for r in carbon_rows), 1),
            "best_region": best_region}


if __name__ == "__main__":
    run()

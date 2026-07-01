# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,626  
**Projected savings:** $12,507  (**46%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $10,040 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1

_Figures are June-2026 as-of snapshots; re-baseline before acting._

## Reasoning budget (Ext4)

- Reasoning traffic: 8.4% requests, 16.5% cost, 94.0% energy
- Rule: route reasoning only when task complexity score < threshold
- Capping reasoning to 10% traffic could save ~$1/month

## Carbon-aware scheduling (Ext5)

- Move interruptible training to `europe-north1`: ~1,479,450 gCO2e/month saved vs us-east-1

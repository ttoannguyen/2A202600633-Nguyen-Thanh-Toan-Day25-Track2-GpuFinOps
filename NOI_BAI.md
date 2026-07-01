# Lab 25 — GPU FinOps Optimization

**Sinh viên:** Nguyen Thanh Toan  
**Mã số:** 2A202600633  
**Khóa học:** AICB · Phase 2 · Track 2 (Infrastructure) · Day 25  
**Ngày nộp:** 01/07/2026

---

## Tóm tắt kết quả

| Kiểm tra | Kết quả |
|---|---|
| `python verify.py` | **11/11 checks passed** |
| `pytest -q` | **16/16 tests passed** |

### File đính kèm

| File | Mô tả |
|---|---|
| `outputs/report.md` | Báo cáo kỹ thuật baseline vs optimized |
| `outputs/savings.png` | Biểu đồ waterfall 4 lever tiết kiệm |
| `outputs/focus_export.csv` | Export phân bổ chi phí theo chuẩn FOCUS (50 dòng mẫu) |
| `NOI_BAI.md` | Bài viết phân tích (file này) |

---

## 1. Baseline vs Optimized

Phân tích dữ liệu tổng hợp của NimbusAI (tháng 6/2026) cho thấy công ty đang trả quá nhiều cho cả **compute GPU** lẫn **inference token**, trong khi hiệu quả sử dụng thực tế thấp hơn nhiều so với chỉ số `nvidia-smi`.

### Inference (đo bằng $/1M-token)

| Chỉ số | Baseline | Optimized | Tiết kiệm |
|---|---|---|---|
| Chi phí inference | $48.87/ngày | $8.48/ngày | **82.6%** |
| Đơn vị $/1M-token | $6.488 | $1.126 | **82.6%** |
| Số request | 2,400 | 2,400 | — |
| Tổng token | 7,533,027 | 7,533,027 | — |

Baseline giả định mọi request chạy model lớn, không cache, không batch. Optimized áp dụng cascade (route_tier), prompt caching và Batch API.

### GPU compute (đo bằng $/GPU-tháng)

| Chỉ số | Baseline | Optimized | Tiết kiệm |
|---|---|---|---|
| Chi phí 8 workload | $25,667/tháng | $15,627/tháng | **39.1%** |

### Tổng hợp (Mission 5)

| Chỉ số | Giá trị |
|---|---|
| Baseline spend | **$27,133/tháng** |
| Optimized spend | **$14,626/tháng** |
| Projected savings | **$12,507 (46%)** |

**Kết luận:** Công ty đạt mục tiêu cắt giảm ≥40% chi phí GPU khi đo bằng $/1M-token và tổng chi phí hàng tháng. Đơn vị $/1M-token quan trọng hơn $/GPU-giờ vì nó buộc ta đo cả **hiệu quả phục vụ** — hai team cùng trả $2.5/giờ H100 nhưng team tối ưu cascade có thể phục vụ 10× nhiều token hơn.

---

## 2. Phân tích từng đòn bẩy

### Bảng breakdown tiết kiệm (M5)

| Lever | Tiết kiệm/tháng | % trong tổng savings |
|---|---|---|
| Purchasing (spot/reserved) | $10,040 | **80.3%** |
| Inference (cascade/cache/batch) | $1,212 | 9.7% |
| Right-size util-lies | $655 | 5.2% |
| Kill idle GPUs | $600 | 4.8% |

### Đòn bẩy đóng góp nhiều nhất: Purchasing (spot/reserved)

**Tại sao lớn nhất:** Chi phí compute GPU chiếm phần lớn tổng bill ($25,667 baseline). Ba job training interruptible (`job-train-llm`, `job-train-embed`, `job-finetune`) chuyển sang **spot** với checkpoint — tiết kiệm ~37–47% so với on-demand. Ba job inference chạy 24/7 (`job-infer-chat`, `job-infer-rag`, `job-infer-search`) đạt duty cycle ≥55% → **reserved 3yr** hợp lý hơn on-demand.

### Đòn bẩy inference mạnh nhất: Cascade

Trong M2, **cascade** là nguồn savings chính vì 80% request trong dataset đủ model nhỏ (`route_tier=small`). Chênh lệch giá input/output giữa small ($0.20/$0.40 per M) và large ($3.00/$15.00 per M) là ~15×. Prompt caching (chiết khấu 90% phần cached) và Batch API (chiết khấu 50%) nhân hiệu quả lên phần còn lại — `discount_stack(batch=True, cache_hit_frac=1.0) = 0.05` tức ~95% off so với naive bill.

### Right-size và Kill idle

- **Right-size:** Hai GPU bị util-lie (`gpu-h100-4`, `gpu-a10g-1`) được hạ cấp H100→A100 trong mô hình M5, tiết kiệm $655/tháng.
- **Kill idle:** `gpu-h100-5` idle 8 giờ/đêm (util <10%) → lãng phí $20/ngày = $600/tháng nếu không tắt.

---

## 3. GPU-Util Lie — Cơ chế và tác động tài chính

### GPU bị phát hiện

| GPU | GPU-Util | MFU | Vấn đề |
|---|---|---|---|
| `gpu-h100-4` | 98.2% | 0.194 | Lie nghiêm trọng nhất |
| `gpu-a10g-1` | 96.9% | 0.268 | Lie thứ hai |

Hàm `flag_util_lies()` flag khi `util ≥ 90%` **và** `MFU < 30%`.

### Tại sao đây là "lie"?

`nvidia-smi` GPU-Util đo **thời gian clock đang active**, không đo FLOPs thực sự hoàn thành. GPU có thể ở trạng thái memory stall, chờ I/O, hoặc kernel launch overhead — SM vẫn "bận" nhưng throughput tính toán thấp. Kết quả: trả **full $2.50/giờ H100** nhưng chỉ nhận ~20% peak FLOPs (990 TFLOPs FP16 × 0.194 ≈ 192 TFLOPs thực).

Theo roofline model, workload memory-bound (LLM decode ~1–2 FLOP/byte) sẽ có MFU thấp dù util cao — đây là tín hiệu cần right-size hoặc tách prefill/decode.

### Tác động tài chính

- **Util-lie GPUs:** Tiếp tục trả giá H100/A10G premium trong khi utilization thực tương đương GPU rẻ hơn → right-size tiết kiệm $655/tháng.
- **Idle waste:** `gpu-h100-5` chạy không 8h/đêm → $600/tháng lãng phí thuần túy (không có output).

---

## 4. Phần mở rộng đã thực hiện

Đã hoàn thành **4/5 extensions** (yêu cầu ≥2).

### Extension 2 — Right-sizing theo MBU (M1)

Với GPU memory-bound (MBU < 0.35), so sánh `$/GB-VRAM` và `peak_bw_tbs` để tìm GPU thay thế rẻ hơn mà vẫn đủ băng thông.

**Kết quả đo được:** 7 GPU inference memory-bound được gợi ý chuyển sang MI300X ($0.0102/GB-VRAM vs A10G $0.0417/GB) — tiết kiệm **55–76%** trên đơn vị VRAM.

**Insight:** Không nên chọn GPU rẻ nhất theo $/GPU-hr vì workload memory-bound cần đủ HBM bandwidth; MI300X có peak_bw 5.3 TB/s và $/GB thấp hơn A10G dù on-demand/hr cao hơn một chút.

### Extension 3 — `cache_is_worth_it()` (finops/pricing.py + M2)

```python
# Break-even: avg_reads × price × (1 - read_discount) > write_cost
cache_break_even_reads(0.50) = 0.56 reads/prefix
avg_cache_reads trong dataset = 300.0 reads/prefix
→ cache_is_worth_it() = True
```

Cache chỉ được tính vào optimized cost khi hàm trả về True. Với prefix chỉ đọc 1 lần (dưới ngưỡng 0.56), bật cache sẽ **lỗ tiền** vì chi phí ghi cache vượt tiết kiệm đọc lại.

### Extension 4 — Ngân sách Reasoning (M2 + M5)

| Chỉ số | Reasoning (is_reasoning=1) | Normal |
|---|---|---|
| % traffic | 8.4% (201/2400 reqs) | 91.6% |
| % chi phí $ | 16.5% | 83.5% |
| % năng lượng Wh | **94.0%** | 6.0% |

Reasoning query tiêu thụ năng lượng gấp ~80× query thường (`REASONING_ENERGY_MULTIPLIER=80`). 8.4% traffic chiếm 94% năng lượng — đây là "energy bomb" cần governance.

**Đề xuất routing rule:** Chỉ bật reasoning khi task complexity score vượt ngưỡng (ví dụ nightly-eval); cap reasoning ≤10% tổng traffic để cân bằng chất lượng và chi phí điện.

### Extension 5 — Carbon-aware Scheduling (M3)

Với 5 job `interruptible=1`, so sánh carbon nếu chạy tại `us-east-1` (380 gCO2/kWh) vs `europe-north1` (30 gCO2/kWh):

| Job | Energy (Wh) | Carbon giảm |
|---|---|---|
| job-train-llm | 3,360,000 | -92% |
| job-train-embed | 480,000 | -92% |
| job-finetune | 252,000 | -92% |
| job-dev-sandbox | 72,000 | -92% |
| job-batch-eval | 63,000 | -92% |
| **Tổng** | — | **~1,479,450 gCO2e/tháng** |

**Trade-off:** `europe-north1` (Na Uy, thủy điện) vừa rẻ điện ($0.09/kWh) vừa sạch nhất, nhưng latency cao hơn với user US — phù hợp batch training interruptible, không phù hợp real-time chat.

---

## 5. Khuyến nghị cho NimbusAI (FinOps Lead)

Nếu tôi là FinOps lead, **3 hành động đầu tiên** theo thứ tự ROI:

### Hành động 1 — Triển khai cascade + cache inference (tuần 1–2)

- Bật model router: 80% request → small model; chỉ escalate large khi cần.
- Bật prompt caching cho team `assistant` và `rag` (prefix cacheable 70%, avg 300 reads/prefix >> break-even 0.56).
- **Kỳ vọng:** $/1M-token từ $6.49 xuống $1.13 (−82.6%), không cần thay đổi hạ tầng GPU.

### Hành động 2 — Tối ưu purchasing tier (tuần 2–4)

- Training jobs interruptible → spot + checkpoint (đã simulate: savings 37–47%).
- Inference 24/7 duty ≥55% → reserved 3yr (break-even @ 45% discount = 55% utilization).
- **Kỳ vọng:** compute cost từ $25,667 xuống $15,627/tháng (−39.1%).

### Hành động 3 — MFU audit + governance (liên tục)

- Dashboard hàng tuần: flag `util ≥ 90% AND MFU < 30%` → right-size hoặc fix kernel.
- Auto-shutdown GPU idle overnight (gpu-h100-5 pattern).
- Bật chargeback: tag coverage 92% > ngưỡng 80% → sẵn sàng thu phí theo team.
- Cap reasoning traffic; cân nhắc chuyển interruptible training sang `europe-north1` để giảm carbon.

---

## Phụ lục: Sustainability

| Chỉ số | Giá trị |
|---|---|
| Energy per query | 0.24 Wh |
| Carbon per query (us-east-1) | 0.091 gCO2e |
| Vùng rẻ + sạch nhất | `europe-north1` (Na Uy, thủy điện) |
| Vùng carbon cao nhất | `europe-central2` (Ba Lan, 660 gCO2/kWh) |

---

*Báo cáo sinh từ `python missions/run_all.py` với dữ liệu seed=25. Số liệu là snapshot tháng 6/2026; cần re-baseline trước khi áp dụng production.*

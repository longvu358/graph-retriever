# So sánh retriever với baseline (K=20, test, 2026-11-11)

Bảng sinh tự động bởi `graph_retriever.compare` từ các CSV trong `results/`. Mean ± std qua 3 seed; Popularity và Markov là chạy đơn định (1 lần). Cột `n` là số lần chạy.

### Pisa (K=20, test)

| Method | Recall@20 | nDCG@20 | ILD | Coverage | Long-tail ratio | n |
|---|---|---|---|---|---|---|
| Popularity | 0.797 | 0.453 | 0.942 | 0.182 | 0.000 | 1 |
| Markov | 0.784 | 0.498 | 0.869 | 0.845 | 0.477 | 1 |
| SR-GNN | 0.813 ± 0.003 | 0.454 ± 0.005 | 0.935 ± 0.008 | 0.215 ± 0.004 | 0.119 ± 0.025 | 3 |
| GCE-GNN | 0.831 ± 0.005 | 0.423 ± 0.001 | 0.929 ± 0.002 | 0.982 ± 0.007 | 0.304 ± 0.003 | 3 |
| DHCN | 0.844 ± 0.006 | 0.458 ± 0.004 | 0.935 ± 0.001 | 0.924 ± 0.011 | 0.312 ± 0.006 | 3 |
| GRU, không đồ thị (đối chứng) | 0.806 ± 0.004 | 0.453 ± 0.003 | 0.947 ± 0.003 | 0.570 ± 0.043 | 0.191 ± 0.031 | 3 |
| HGNN v1 (có kênh session) | 0.739 ± 0.000 | 0.314 ± 0.006 | 0.928 ± 0.001 | 0.500 ± 0.027 | 0.453 ± 0.001 | 3 |
| **HGNN v2: GRU + theme + area (của ta)** | 0.830 ± 0.006 | 0.491 ± 0.004 | 0.931 ± 0.001 | 0.500 ± 0.007 | 0.343 ± 0.010 | 3 |

### Florence (K=20, test)

| Method | Recall@20 | nDCG@20 | ILD | Coverage | Long-tail ratio | n |
|---|---|---|---|---|---|---|
| Popularity | 0.350 | 0.142 | 0.927 | 0.024 | 0.000 | 1 |
| Markov | 0.424 | 0.225 | 0.921 | 0.777 | 0.238 | 1 |
| SR-GNN | 0.409 ± 0.001 | 0.190 ± 0.001 | 0.925 ± 0.003 | 0.158 ± 0.007 | 0.009 ± 0.001 | 3 |
| GCE-GNN | 0.501 ± 0.004 | 0.237 ± 0.001 | 0.920 ± 0.000 | 0.758 ± 0.010 | 0.095 ± 0.001 | 3 |
| DHCN | 0.497 ± 0.003 | 0.229 ± 0.001 | 0.920 ± 0.001 | 0.938 ± 0.006 | 0.115 ± 0.002 | 3 |
| GRU, không đồ thị (đối chứng) | 0.480 ± 0.001 | 0.244 ± 0.001 | 0.920 ± 0.000 | 0.562 ± 0.052 | 0.076 ± 0.013 | 3 |
| HGNN v1 (có kênh session) | 0.378 ± 0.008 | 0.163 ± 0.005 | 0.930 ± 0.001 | 0.207 ± 0.038 | 0.019 ± 0.004 | 3 |
| **HGNN v2: GRU + theme + area (của ta)** | 0.497 ± 0.004 | 0.248 ± 0.002 | 0.918 ± 0.001 | 0.526 ± 0.057 | 0.078 ± 0.010 | 3 |

### Rome (K=20, test)

| Method | Recall@20 | nDCG@20 | ILD | Coverage | Long-tail ratio | n |
|---|---|---|---|---|---|---|
| Popularity | 0.453 | 0.188 | 0.960 | 0.049 | 0.000 | 1 |
| Markov | 0.696 | 0.404 | 0.943 | 0.722 | 0.175 | 1 |
| SR-GNN | 0.677 ± 0.005 | 0.374 ± 0.003 | 0.938 ± 0.001 | 0.482 ± 0.021 | 0.086 ± 0.012 | 3 |
| GCE-GNN | 0.720 ± 0.001 | 0.394 ± 0.000 | 0.937 ± 0.000 | 0.746 ± 0.012 | 0.146 ± 0.001 | 3 |
| DHCN | 0.717 ± 0.001 | 0.360 ± 0.000 | 0.936 ± 0.000 | 0.949 ± 0.008 | 0.146 ± 0.001 | 3 |
| GRU, không đồ thị (đối chứng) | 0.714 ± 0.001 | 0.410 ± 0.002 | 0.938 ± 0.000 | 0.677 ± 0.024 | 0.168 ± 0.005 | 3 |
| HGNN v1 (có kênh session) | 0.620 ± 0.002 | 0.291 ± 0.001 | 0.941 ± 0.001 | 0.465 ± 0.004 | 0.064 ± 0.006 | 3 |
| **HGNN v2: GRU + theme + area (của ta)** | 0.720 ± 0.002 | 0.411 ± 0.000 | 0.936 ± 0.000 | 0.672 ± 0.009 | 0.154 ± 0.001 | 3 |

## Đọc kết quả

**Độ chính xác.** HGNN v2 (GRU + theme + area) ngang GCE-GNN và DHCN về Recall@20 ở cả ba thành phố (chênh lệch ≤ 0.014). Vượt SR-GNN ở mọi thành phố (+0.017 Pisa, +0.088 Florence, +0.043 Rome), và vượt Markov và Popularity. *(lỗi thời, xem sửa bên dưới)*

> **Sửa (2026-11-18, theo A2 và C0):** Recall@20 của v2 ngang GCE-GNN và DHCN, không cặp nào có ý nghĩa sau Holm (Florence seed 0 vs GCE-GNN thấp hơn 0.010, p Holm = 0.08). So với SR-GNN, chỉ Florence và Rome có ý nghĩa (3/3 seed ở Florence, Rome 3/3 seed có ý nghĩa); Pisa (+0.017) không có ý nghĩa sau Holm. nDCG@20 của v2 cao hơn GCE-GNN, DHCN và SR-GNN có ý nghĩa ở cả ba thành phố và mọi seed. So với GRU4Rec (C0: Pisa 0.8160, Florence 0.4417, Rome 0.6934), nDCG không nhất quán. Nguồn: `results_remote/a2_significance/summary.md`.

**Chất lượng xếp hạng (nDCG@20).** Đây là điểm mạnh nhất của v2: cao nhất ở Pisa (0.491 so với 0.423–0.458 của các baseline neural) và Florence (0.248 so với 0.190–0.237). Ở Rome, v2 ngang GRU thuần và GCE-GNN (0.411 so với 0.410 và 0.394). Lợi thế nDCG cho thấy item đúng được xếp cao hơn, không chỉ nằm trong top-20.

**Đa dạng và phủ (ILD, coverage, long-tail).** v2 không mạnh về phủ: coverage 0.50 / 0.53 / 0.67 so với GCE-GNN và DHCN (0.75–0.98). ILD gần như bằng nhau giữa các model (0.92–0.94). Tức là retriever không tạo đa dạng đáng kể; việc này được giao cho reranker LLM (xem `plan/pool_interface.md`).

**Đóng góp của đồ thị.** So với GRU không đồ thị, kênh theme + area tăng Recall@20 +0.024 (Pisa), +0.017 (Florence), +0.006 (Rome). nDCG tăng +0.038 ở Pisa, +0.004 ở Florence, 0 ở Rome. Đóng góp rõ nhất ở Pisa, nhưng Pisa là dataset nhỏ nhất (110 cluster). Cần nêu rõ đây là đóng góp nhỏ. *(lỗi thời, xem sửa bên dưới)*

> **Sửa (2026-11-18, theo A2):** Recall@20 so với GRU không đồ thị có ý nghĩa sau Holm ở Florence (3/3 seed), không có ý nghĩa ở Pisa (0/3), và 1/3 seed ở Rome. nDCG@20 có ý nghĩa ở Pisa (3/3) và Florence (2/3). Vì vậy đóng góp không nhất quán giữa các thành phố: rõ ràng nhất ở Florence về Recall, và ở Pisa về nDCG. Không nên viết "rõ nhất ở Pisa".

## Caveat khi báo cáo

- Chỉ 3 seed. Độ lệch chuẩn nhỏ (≤ 0.01 hầu hết), nhưng chưa có kiểm định thống kê. Chênh lệch dưới ~0.01 không nên gọi là khác biệt.
- Baseline SR-GNN, GCE-GNN, DHCN chạy bằng siêu tham số mặc định của repo gốc, không tune. HGNN v2 cũng không tune nhiều (chỉ chọn thiết kế trên val). Cần nói rõ để tránh so sánh không công bằng.
- Các baseline dùng protocol của ta (early stop trên val, prefix augmentation). Repo gốc early-stop trên test; số liệu ở đây không trực tiếp so được với paper gốc.
- Popularity và Markov là một lần chạy; không có độ lệch chuẩn.
- DHCN cần hai patch nhỏ (tensor CUDA trong `np.sum`, cắt padding), đã ghi trong `plan/v0.md`.
- HGNN v1 trong bảng là bản cũ có kênh session (rò rỉ). Đưa vào để minh hoạ, không phải đề xuất.
- Dispersal (penalty, substitution) không nằm trong bảng này; kết quả dispersal ở `results/tune_summary.md` dùng model v1 và cần chạy lại trên v2 trước khi báo cáo.

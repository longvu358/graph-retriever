# Tầng trước rerank: kiến trúc và giao diện cho reranker LLM (2026-11-11)

Phạm vi: phần của Long (retriever + pool ứng viên). Phần LLM reranker do Minh-san phụ trách; tài liệu này định nghĩa file đầu vào của reranker.

## 1. Kiến trúc đã chốt

| Thành phần | Quyết định | Căn cứ |
|---|---|---|
| Retriever | GRU trên prefix + kênh theme và area (hypergraph, không có kênh session) | v2: vượt SR-GNN/GCE-GNN/DHCN ở Florence và Rome, không rò rỉ target |
| Đầu vào retriever | Prefix cluster ids + bucket thời gian của điểm cuối prefix | v3: thêm session edge và ngữ cảnh thời gian không cải thiện đáng kể |
| Huấn luyện | CE trên mọi item, early stop theo val R@20, không SSL | v4: SSL hai view không cải thiện |
| Pool | Top-100 lưu đầy đủ; reranker dùng N=50 mặc định | Xem mục 2 |
| Dispersal trong pool | Không áp dụng; penalty chỉ đổi nhẹ pool | Xem mục 2 |
| Đa dạng | Để reranker LLM đảm nhiệm | Phản hồi của senpai |

## 2. Chọn kích thước pool trên validation

Recall@N (mean 3 seed, val; test trong ngoặc):

| Thành phố | N=10 | N=20 | N=50 | N=100 | Số category trung bình trong top-50 |
|---|---|---|---|---|---|
| Pisa | 0.697 (0.719) | 0.812 (0.830) | 0.939 (0.949) | 0.997 (0.995) | 24.0 |
| Florence | 0.392 (0.376) | 0.524 (0.497) | 0.683 (0.662) | 0.798 (0.780) | 32.5 |
| Rome | 0.626 (0.586) | 0.752 (0.720) | 0.885 (0.872) | 0.953 (0.949) | 87.3 |

Quyết định: **N=50**.
- Từ N=50 lên N=100 chỉ tăng khoảng 0.05 recall (Pisa, Rome) hoặc 0.12 (Florence), nhưng gấp đôi số ứng viên cho LLM. Cặp (chi phí, recall) này phù hợp hơn với N=50.
- Với N=20 (kích thước của LLM4Rerank), mất khoảng 0.13 recall ở Pisa và 0.16 ở Florence so với N=50. Điều này làm trần của reranker thấp hơn đáng kể.
- Lựa chọn N cuối cùng nên được xác nhận bằng kết quả reranker trên val, vì pool lớn có thể thêm nhiễu (xem tài liệu về recall ceiling trong mục 4).

## 3. Dispersal trong pool

So sánh pool plain với pool penalty (λ=0.25, λ=1) ở N=20:
- Recall gần như không đổi (Rome: 0.752 → 0.751 với λ=1).
- Số category giảm nhẹ với λ=1 (Rome: 58.7 → 55.3; Pisa: 17.2 → 17.0).

Kết luận: penalty trong pool không đem lại đa dạng đáng kể. Đa dạng cần đến từ reranker. Pool giữ dạng plain, reranker quyết định việc đa dạng hoá.

## 4. Căn cứ bên ngoài

- Trần recall: reranker không thể vượt Recall@N của pool, nên N phải đủ lớn ([The Recall Ceiling of LLM Recommendation Reranking](https://arxiv.org/html/2609.27953v1)).
- Pool lớn có thể làm kém đi: nghiên cứu cold-start cho thấy pool 200 cho HR@10 cao hơn pool 500 và 1000 ([Diagnosing LLM-based Rerankers in Cold-Start Recommender Systems](https://arxiv.org/html/2604.16318v1)). Kết quả này dùng cross-encoder, không phải LLM, nên chỉ là gợi ý.
- LLM4Rerank dùng 20 ứng viên mỗi người dùng và cho rằng hiệu năng giảm khi số lượng tăng do ngữ cảnh dài ([LLM4Rerank](https://arxiv.org/html/2406.12433v3)).

## 5. Giao diện file cho reranker

Các file được tạo bởi `graph_retriever.pool`:
- `results/pool/{city}_seed{s}_{split}.npz`: `items` (n, 100) int32, `rel` (n, 100) float32 (log-softmax của retriever), `targets` (n,), `prefix_last`, `ctx`. Dùng cho đánh giá, không đưa cho LLM.
- `results/pool/{city}_seed{s}_report.json`: recall và thống kê đa dạng cho từng N và từng pool.

File đầu vào cho reranker (JSON Lines, mỗi dòng một truy vấn), tạo bởi `write_candidates`:

```json
{
  "query_id": "pisa-pisa_seed0_test-12",
  "city": "pisa",
  "split": "test",
  "history": [{"item_id": "15", "name": "Torre pendente di Pisa", "categories": ["campanili", "torridipisa"], "lat": 43.72, "lon": 10.39}],
  "time_bucket": 7,
  "candidates": [
    {"item_id": "70", "name": "Cappella Dal Pozzo", "categories": ["cappelledipisa"], "lat": 43.72, "lon": 10.39,
     "rank": 1, "retriever_score": -1.9525, "pop": 0.8617}
  ]
}
```

Quy tắc:
- Không có target trong file. Nhãn chỉ nằm trong `.npz`.
- `candidates` có đúng N phần tử, `item_id` không trùng, lấy từ pool của retriever.
- `history` là prefix (theo thứ tự thời gian). `time_bucket` là bucket của điểm cuối prefix (12 giá trị).
- `pop` là popularity từ train, chuẩn hoá [0, 1]. Đây là tín hiệu cho việc đa dạng/long-tail, không phải nhãn.
- Reranker phải trả về một hoán vị của đúng các `item_id` trong `candidates`, kiểm tra bằng code (không tin vào LLM).

Lệnh tạo:

```
uv run python -m graph_retriever.pool --raw <TripBuilderRawData> --out results/pool --cities pisa florence rome --seeds 0 1 2
```

Sau đó dùng `write_candidates(city, raw, npz, out_jsonl, n=50)` để tạo file JSONL.

## 6. Trạng thái và việc còn lại

Đã xong:
- File JSONL N=50 cho 3 thành phố × 3 seed × (val, test): `results/candidates/{city}_seed{s}_{split}_n50.jsonl` trên server. Tổng ~986 MB (lặp tên và category cho mỗi ứng viên; có thể nén nếu cần).
- Lệnh tái tạo: `python -m graph_retriever.pool --raw <...> --out results/pool --cities ... --seeds ... --candidates [--n 50]`.

Còn lại:
- Gửi spec này và các file JSONL cho Minh-san để tích hợp reranker.
- Chốt N trên val khi có kết quả reranker (N ∈ {20, 50, 100}). File top-100 đã có trong `.npz`, nên đổi N không cần train lại.

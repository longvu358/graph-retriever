# Hypergraph trong recommendation: các hướng tiên tiến và áp dụng cho v0

Cập nhật: 2026-11-10. Mục tiêu: tìm cách dùng hypergraph vượt quá HGCN đơn giản (DHCN-style) cho bài toán next-POI có dispersal.

Quy ước độ tin cậy:
- **[đã kiểm]**: có nguồn trong kết quả tìm kiếm phiên này (link trong mục Nguồn).
- **[chưa kiểm]**: từ kiến thức nền, cần đối chiếu bản gốc (venue, năm, số liệu) trước khi trích dẫn.

## 1. Bức tranh tổng quát

Hypergraph trong recommendation thường dùng theo bốn cách:
1. **Hypergraph làm cấu trúc chính**, học biểu diễn node qua convolution. Đây là DHCN, HGNN gốc.
2. **Multi-view / multi-channel**: nhiều hypergraph (ngữ nghĩa khác nhau) hoặc hypergraph + graph, học song song rồi hợp nhất.
3. **Self-supervised / contrastive** giữa các view để đồng bộ biểu diễn (DHCN, SHT, HCCF). Phổ biến nhất trong các paper mới.
4. **Hypergraph theo ngữ cảnh** (không gian, thời gian, khu vực) cho POI. Đây là hướng gần bài toán nhất.

Ba điểm xuyên suốt cần lưu ý cho bài này:
- **Rò rỉ khi xây hyperedge từ chuỗi**: nếu hyperedge lấy từ chuỗi đầy đủ thì target nằm trong hyperedge (đúng lỗi kênh session của v1). Các paper POI như STHGCN xây hyperedge từ chuỗi đầu vào (prefix) — cùng nguyên tắc với bản sửa của ta.
- **Dispersal**: các paper hypergraph gần như không xử lý popularity/over-tourism trực tiếp. Phần này phải tự thiết kế (xem mục 4).
- **Dữ liệu TripBuilder**: có timestamp của từng điểm (first/last photo) và tọa độ cluster, nhưng không có rating, không có text. Nên ưu tiên các hướng dùng không gian + thời gian + category.

## 2. Các paper theo nhóm

### 2.1 Session-based, hypergraph + contrastive

| Paper | Ý chính | Trạng thái |
|---|---|---|
| DHCN — Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation (AAAI 2021) | Hypergraph session (node–session) + line graph (session–session, trọng số Jaccard), InfoNCE giữa hai view, β≈0.01 | [đã kiểm] README repo `xiaxin1998/DHCN` ghi AAAI 2021. Đã có làm baseline trong dự án. |
| SDHID — Self-supervised Dual Hypergraph learning with Intent Disentanglement for session-based recommendation | Hai kênh hypergraph, tách ý định (intent), contrastive maximize mutual information giữa biểu diễn session | [đã kiểm] ScienceDirect (năm và tạp chí chưa xác định) |
| MoHyNet — Enhancing Session-Based Recommendations via Hypergraph Motifs and Contrastive Learning (MDPI AI) | Motif hypergraph (bộ phận nhỏ lặp lại) + contrastive hai view giữa intra-session và inter-session | [đã kiểm] MDPI, năm chưa xác định |
| Motifs-based Recommender System via Hypergraph Convolution and Contrastive Learning (arXiv 2109.00676) | Dùng motif để dựng hypergraph, contrastive | [đã kiểm] arXiv |

### 2.2 Collaborative filtering, hypergraph + contrastive / transformer

| Paper | Ý chính | Trạng thái |
|---|---|---|
| HCCF — Hypergraph Contrastive Collaborative Filtering (SIGIR 2022, Xia et al.) | Hypergraph học tương quan toàn cục giữa user/item; contrastive giữa view hypergraph và view graph cục bộ | [đã kiểm] trích dẫn trong kết quả tìm kiếm |
| SHT — Self-Supervised Hypergraph Transformer for Recommender Systems (KDD 2022, Xia, Huang, Zhang) | Hypergraph transformer học quan hệ toàn cục; cross-view generative SSL để augment. Mã nguồn: `akaxlh/SHT` | [đã kiểm] DOI 10.1145/3534678.3539473, arXiv 2207.14338 |
| MHCN — Multi-channel Hypergraph Convolutional Network for social recommendation (WWW 2021) | Nhiều kênh hypergraph theo motif xã hội, hợp nhất bằng contrastive | [chưa kiểm] |

### 2.3 POI / next-POI, hypergraph theo không gian–thời gian

| Paper | Ý chính | Trạng thái |
|---|---|---|
| STHGCN — Spatio-Temporal Hypergraph Learning for Next POI Recommendation (SIGIR 2023, Yan et al.) | Hypergraph trên check-in trong quỹ đạo + quan hệ giữa các quỹ đạo, có ngữ cảnh không gian–thời gian | [đã kiểm] DOI 10.1145/3539618.3591770 (SIGIR '23). Đây là paper nền cho hướng POI của ta. |
| MvStHgL — Multi-View Hypergraph Learning with Spatial-Temporal Periodic Interests for Next POI (ACM, DOI 10.1145/3664651) | Hai view cục bộ–toàn cục, học sở thích chu kỳ theo không gian và thời gian | [đã kiểm] DOI; venue cụ thể chưa xác nhận |
| ReHDM — Region-aware Dynamic Hypergraph with Dual-level Modeling (IJCAI 2025) | Hypergraph theo khu vực (region), hai mức POI và quỹ đạo, kết hợp transformer | [đã kiểm] ijcai.org 2025/343 |
| ASTHN — Adaptive Spatio-Temporal Hypergraph Network for Next POI (ISPRS IJGI) | Ba hypergraph theo khoảng cách thời gian tối thiểu, khoảng cách không gian, sở thích theo giờ; kết hợp GRU | [đã kiểm] mdpi 15/6/242 (năm chưa xác nhận) |
| HyperMAN — Hypergraph-enhanced Meta-learning Adaptive Network for Next POI (arXiv 2503.22049) | Hypergraph + meta-learning cho cold-start; diversity theo danh mục hành vi người dùng, không phải popularity | [đã kiểm] arXiv |

### 2.4 Debias popularity / long-tail

| Paper | Ý chính | Trạng thái |
|---|---|---|
| CaDRec — Contextualized and Debiased Recommender Model (arXiv 2404.06895) | Hypergraph convolution, mã hoá popularity vào positional encoding để item cùng độ phổ biến gần nhau | [đã kiểm] arXiv |
| Topology-Aware Popularity Debiasing via Simplicial Complexes (arXiv 2411.13892, Ji et al.) | Test-time Simplicial Propagation: thêm cấu trúc bậc cao lúc inference, không cần train lại; cải thiện long-tail trên 5 bộ dữ liệu | [đã kiểm] arXiv |
| Point of Interest Recommendation: Pitfalls and Viable Solutions (arXiv 2507.13725) | Khảo sát: popularity bias làm hại công bằng và du lịch bền vững ở POI; gợi ý tối ưu đa mục tiêu có diversity | [đã kiểm] arXiv |

### 2.5 Nền tảng và khảo sát

| Paper | Ý chính | Trạng thái |
|---|---|---|
| A Survey on Hypergraph Neural Networks: An In-Depth and Step-By-Step Guide (Kim et al., 2024) | Phân loại HNN theo 4 thành phần: đặc trưng đầu vào, cấu trúc đầu vào (clique, star, line expansion), message passing, huấn luyện; có mục recommendation | [đã kiểm] trang tổng hợp; năm và venue theo kiến thức nền [chưa kiểm] |
| Recent Advances in Hypergraph Neural Networks (Yang & Xu, arXiv 2503.07959, 2025) | Phân loại HGCN, HGAT, HGAE, HGRN, DHGGM | [đã kiểm] tiêu đề và arXiv id |
| AllSet (Chien et al., ICLR 2022) | Hàm tổng hợp dạng set cho hypergraph, tổng quát hoá các HGNN trước | [chưa kiểm] |
| HyperGCN (Yadati et al., NeurIPS 2019) | HGCN bằng clique expansion có làm mịn | [chưa kiểm] |
| UniGNN (Huang & Yang, IJCAI 2021) | Message passing hai bước node↔hyperedge thống nhất | [chưa kiểm] |
| HGNN+ (Gao et al., TPAMI 2022) | Khung hypergraph tổng quát, nhiều dạng cấu trúc | [chưa kiểm] |

## 3. Các cấu trúc hypergraph và tương ứng với dữ liệu của ta

| Cấu trúc | Định nghĩa | Ta có dữ liệu không | Ghi chú |
|---|---|---|---|
| Session (trajectory) | Node = cluster trong chuỗi | Có | Dễ rò rỉ; chỉ dùng prefix |
| Theme (category) | Node = cluster có category | Có | Đã có |
| Area (không gian) | Node = cluster trong bán kính | Có | Đã có, là kênh có đóng góp nhất |
| Time window | Node = cluster ghé trong cùng khung giờ/ngày | Có (first/last photo timestamp) | Chưa dùng; phù hợp với "periodic interest" của MvStHgL |
| Region (nhiều area) | Gộp area theo lưới lớn hơn | Có | Dùng cho long-tail: nối node ít ghé với vùng lân cận |
| Motif (bộ con lặp) | Bộ cluster xuất hiện cùng nhau nhiều lần | Có | Phức tạp hơn; xem MoHyNet/Motifs-based |

## 4. Các hướng áp dụng cho v0

Xếp theo mức phù hợp với mục tiêu (accuracy + dispersal) và rủi ro:

1. **Hyperedge chỉ từ prefix (per-sample), theo STHGCN.** Bắt buộc để tránh rò rỉ. Chi phí: dựng đồ thị theo từng mẫu, chậm hơn. Có thể giảm chi phí bằng cách chỉ dùng hyperedge theme/area (không phụ thuộc chuỗi) và chỉ dùng hyperedge session từ prefix khi cần.
2. **Kênh thời gian (time-window hyperedge).** Dữ liệu có timestamp nhưng ta chưa dùng. Là khác biệt so với DHCN và phù hợp với bài toán. Rủi ro: timestamp chỉ là ảnh Flickr, có thể lệch múi giờ và thiên về mùa du lịch.
3. **Đa view contrastive (DHCN/SHT/HCCF).** Ghép view hypergraph với view graph cục bộ (đồ thị chuyển tiếp hoặc line graph), InfoNCE giữa hai view. Đã thấy DHCN có lợi nhỏ trên dữ liệu này; cần ablation SSL trước khi giữ.
4. **Popularity trong biểu diễn (CaDRec).** Mã hoá log-popularity vào embedding hoặc positional term, để model tự cân bằng thay vì penalty hậu kỳ. Đây là đường thay thế cho dispersal post-hoc.
5. **Dispersal ở inference (Simplicial propagation).** Ý tưởng test-time propagation có thể áp dụng cho hypergraph: lan truyền thêm trên hyperedge area/theme lúc suy luận để kéo điểm item ít phổ biến cùng vùng lên. Khác với substitution hiện tại ở chỗ có cơ sở toán học và không cần train lại.

## 5. Gợi ý thứ tự thử

1. Sửa rò rỉ: hyperedge session chỉ từ prefix (hoặc bỏ). Đo lại GRU + theme/area.
2. Thêm kênh time-window (1 kênh, không thêm view khác). So với v2.
3. Thêm SSL hai view (DHCN-style) với ablation β ∈ {0, 0.01, 0.1}.
4. Thử popularity encoding (CaDRec-style) so với penalty hậu kỳ.
5. Chỉ sau đó mới xét test-time propagation cho dispersal.

Mỗi bước phải qua đối chứng trên val, không chọn trên test (xem quy tắc trong `plan/v0.md`).

## 6. Việc cần làm để tài liệu đủ tin cậy

- Đối chiếu venue và năm của các mục [chưa kiểm] với bản gốc trước khi trích dẫn.
- Đọc STHGCN và DHCN để xác nhận cách dựng hyperedge và loss (không chỉ dựa vào tóm tắt).
- Đọc CaDRec và 2411.13892 để xác nhận cách mã hoá popularity và công thức test-time propagation.

## Nguồn

- Self-Supervised Hypergraph Transformer for Recommender Systems: https://arxiv.org/pdf/2207.14338 ; https://dl.acm.org/doi/10.1145/3534678.3539473
- Spatio-Temporal Hypergraph Learning for Next POI Recommendation (SIGIR 2023): https://dl.acm.org/doi/pdf/10.1145/3539618.3591770
- MvStHgL: https://dl.acm.org/doi/10.1145/3664651
- ReHDM (IJCAI 2025): https://ijcai.org/proceedings/2025/343
- ASTHN: https://www.mdpi.com/2220-9964/15/6/242
- HyperMAN: https://arxiv.org/html/2503.22049v1
- Self-supervised Dual Hypergraph learning with Intent Disentanglement (SDHID): https://www.sciencedirect.com/science/article/abs/pii/S0950705123002782
- MoHyNet: https://doi.org/10.3390/ai7060197
- Motifs-based Recommender System via Hypergraph Convolution and Contrastive Learning: https://arxiv.org/pdf/2109.00676
- Hyperbolic Hypergraphs for Sequential Recommendation: https://arxiv.org/pdf/2108.08134
- Survey on HNNs (Kim et al.): https://aminer.org/pub/660b82fa13fb2c6cf6aba30d/a-survey-on-hypergraph-neural-networks-an-in-depth-and-step-by
- Recent Advances in Hypergraph Neural Networks (Yang & Xu): https://www.themoonlight.io/review/recent-advances-in-hypergraph-neural-networks (tóm tắt; bản gốc arXiv 2503.07959)
- CaDRec: https://arxiv.org/abs/2404.06895
- Topology-Aware Popularity Debiasing via Simplicial Complexes: https://arxiv.org/abs/2411.13892
- Point of Interest Recommendation: Pitfalls and Viable Solutions: https://arxiv.org/pdf/2507.13725
- DHCN (repo): https://github.com/xiaxin1998/DHCN

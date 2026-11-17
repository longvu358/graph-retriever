# Hướng tiếp theo sau nhận xét của senpai (cập nhật 2026-11-11)

Nguồn: email Pongsakorn (Coke) gửi Long, 2026-11-11, trả lời mail về hướng Graph Retriever trên hypergraph cho over-tourism-aware POI recommendation (`review_from_senpai.pdf`).

## 1. Nhận xét chính và cách trả lời

| Nhận xét | Thực trạng của ta | Hành động |
|---|---|---|
| Chỉ đổi ranking sang hypergraph thì thiếu novelty. Cần làm phần LLM, không chỉ prompting | Retriever GRU + theme/area chỉ hơn GRU thuần 0.006–0.024. Chưa có LLM nào trong pipeline | Đưa LLM reranker có mục tiêu diversity/novelty vào (mục 3) |
| Tài liệu về diversity/novelty đã cũ (201x), cần tìm sau 2020 cho graph, hypergraph, LLM | Đã tìm (mục 2). Hypergraph + diversity/novelty gần như chưa có | Đưa vào `plan/hypergraph_methods.md` và bảng tài liệu mới |
| Gợi ý: GRU4Rec (1511.06939), baseline mạnh (2406.12433), survey RAG (hal-05561909), tutorial KDD 2026 | Đã kiểm: 2406.12433 là **LLM4Rerank**, một reranker LLM chứ không phải baseline retrieval. Tutorial KDD 2026 (Jeju, 9–13/8/2026) có survey RAG cho LLM-RecSys | 2406.12433 dùng làm baseline reranker. GRU4Rec làm mốc session cổ điển |
| Thử dataset chuẩn trước, TripBuilder là ứng viên | Hiện chỉ có TripBuilder (Rome, Florence, Pisa) | Thêm dataset POI chuẩn (xem mục 4). TripBuilder giữ làm bộ thứ hai |
| Long hỏi input: session, metadata POI, timestamp, sở thích người dùng | Có session, category, tọa độ, timestamp từng điểm. Không có profile người dùng | Trả lời trong báo cáo: input gồm session + metadata POI + timestamp; không có sở thích người dùng trong TripBuilder |

## 2. Tài liệu mới (sau 2020) cần đọc

Đã kiểm qua tìm kiếm (có nguồn):
- **LLM4Rerank** (arXiv 2406.12433, 2024): reranker LLM cân bằng accuracy, diversity, fairness bằng chain-of-thought trên đồ thị đầy đủ. Đánh giá diversity bằng α-NDCG. Chưa rõ venue.
- **Carraro & Bridge**, "Enhancing recommendation diversity by re-ranking with large language models" (ACM TORS, 2024): được trích dẫn nhiều lần như tài liệu nền cho reranking LLM để tăng diversity. Chưa đọc abstract trong phiên này.
- **Multi-Objective Recommendation in the Era of Generative AI** (arXiv 2506.16893, 2025): khảo sát đa mục tiêu, gồm diversity. Đọc để làm khung đánh giá.
- **Complementary product reranking** (arXiv 2507.16237, 2025): diversity agent rồi accuracy agent trên retriever GNN. Mẫu kiến trúc hai bước.
- **QueRec** (ACL Findings EMNLP 2025): LLM sinh mô tả item và truy vấn, rerank thích ứng để tăng novelty/diversity.
- **DLCRec** (arXiv 2408.12470, 2024): diversity trong LLM-RecSys (chỉ thấy trong danh sách tài liệu của một khảo sát, chưa đọc nội dung).
- **Survey RAG cho LLM-RecSys** (hal-05561909) và tutorial KDD 2026: nền tảng cho cách đưa ngữ cảnh ngoài (candidate, metadata) vào prompt.

Khoảng trống đã xác nhận: chưa tìm thấy paper nào kết hợp hypergraph với mục tiêu diversity/novelty có ràng buộc. Đây là chỗ có thể đóng góp, nhưng cần đọc thêm trước khi khẳng định.

Mục tiêu cần làm: đọc abstract và phần method của LLM4Rerank và Carraro & Bridge; đọc survey hal-05561909 để lấy danh sách tham chiếu RAG-RecSys.

## 3. Hướng đề xuất

Dựa trên kết quả đã có, đề xuất giữ bài toán over-tourism và đổi trọng tâm như sau:

**Vai trò của từng phần**
- Retriever: GRU + theme/area (v2). Là baseline mạnh và ổn định (ngang GCE-GNN, DHCN trên Florence/Rome). Không còn là đóng góp chính, vì đóng góp của hypergraph nhỏ.
- Đóng góp mới: **reranker có ý thức diversity/novelty, dùng LLM với tín hiệu crowd/popularity và khu vực làm đầu vào có cấu trúc**, không chỉ prompt đơn giản.

**Thiết kế reranker (đề xuất)**
1. Đầu vào: top-N ứng viên từ retriever, mỗi ứng viên kèm metadata (category, area id, popularity bucket, khoảng cách tới điểm cuối prefix, thời gian), và session prefix.
2. Quy trình hai bước kiểu Complementary reranking: bước diversity chọn tập đa dạng theo area/category; bước accuracy sắp xếp trong tập đó. Hoặc một bước với mục tiêu gộp có trọng số.
3. Mô hình: LLM qua API cho bản đầu (để kiểm chứng ý tưởng); sau đó chưng cất sang model nhỏ nếu có kết quả (senpai gợi ý knowledge distillation).
4. Ràng buộc: LLM chỉ được chọn trong tập ứng viên, và đầu ra phải hợp lệ (đủ K, không trùng). Kiểm tra bằng code, không tin vào LLM.

**Baseline và đánh giá**
- Dispersal cổ điển: penalty popularity, MMR (đã có), xQuAD (chưa làm).
- Reranker LLM: LLM4Rerank (2406.12433), Carraro & Bridge (nếu tái lập được).
- Metric: accuracy (Recall/NDCG@K), dispersal (ILD, coverage, long-tail, ARP), novelty (α-NDCG hoặc tương đương).
- Quy tắc chọn tham số: trên val, như đã làm cho dispersal.

## 4. Dataset chuẩn cho POI

Senpai đề nghị dùng dataset chuẩn trước. Ứng viên thường gặp trong POI recommendation: Foursquare (NYC, TKY) và Gowalla. Chưa kiểm tra license, độ phù hợp, và định dạng trong phiên này. Cần làm trước khi cam kết dùng.

TripBuilder giữ như bộ thứ hai, vì có timestamp và tọa độ cluster nhưng không có profile người dùng.

## 5. Lộ trình đề xuất

1. **Tuần này**: đọc LLM4Rerank, Carraro & Bridge, survey hal-05561909. Kiểm tra license và định dạng Foursquare/Gowalla.
2. **Tiếp theo**: ổn định retriever trên dataset chuẩn, so với GRU4Rec và các baseline đã có, theo protocol hiện tại.
3. **Sau đó**: xây reranker (phiên bản LLM API đầu tiên) trên đầu ra retriever, so với penalty, MMR, LLM4Rerank.
4. **Trình bày ở buổi họp**: kết quả v1–v4 (đã có), đối chứng GRU không đồ thị (đã có), và đề xuất reranker. Hỏi senpai về tiêu chí novelty muốn dùng.

## 6. Việc cần chốt với senpai/Minh-san

- Dataset chuẩn nào sẽ dùng làm chính.
- Novelty đo bằng metric nào (α-NDCG hay định nghĩa khác).
- Có chấp nhận đổi trọng tâm sang reranker LLM không. Email có nhắc đến một paper WWW trước đó; cần xác nhận có phải nên tiếp tục hướng đó không.
- Minh-san phụ trách LLM reranker (theo email gốc). Cần chia việc rõ để tránh trùng.

## Nguồn

- LLM4Rerank: https://arxiv.org/abs/2406.12433
- Tutorial RAG for RecSys (KDD 2026): https://recsys-rag-tutorial.github.io/
- Multi-Objective Recommendation in the Era of Generative AI: https://arxiv.org/pdf/2506.16893
- LLM-Enhanced Reranking for Complementary Product Recommendation: https://arxiv.org/html/2507.16237v1
- QueRec (ACL Findings EMNLP 2025): https://aclanthology.org/2025.findings-emnlp.446.pdf
- DLCRec (arXiv 2408.12470): trích từ danh sách tài liệu khảo sát, chưa mở trực tiếp
- Carraro & Bridge (ACM TORS 2024): trích dẫn trong các kết quả tìm kiếm, chưa mở trực tiếp

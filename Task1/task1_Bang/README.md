# Task 1 — Kiểm tra và phân tích dữ liệu

**Người thực hiện:** Phan Doãn Thanh Bằng

## Mục tiêu

Kiểm tra chất lượng và phân bố dữ liệu để bàn giao cho nhóm chia tập và xây baseline.

## Kết quả

- Train có 699.960 dòng, 26 cột; test có 299.984 dòng, 25 cột. Hai tập có cùng các cột đầu vào; nhãn `is_fraud` chỉ có ở train.
- Không phát hiện giá trị thiếu, ID trùng, dòng trùng đặc trưng hoặc dòng có đặc trưng giống nhau giữa train và test.
- Nhãn hợp lệ nhưng mất cân bằng: 11.689 mẫu fraud, chiếm khoảng 1,67%.
- Phân bố các đặc trưng train/test nhìn chung tương đồng; chưa phát hiện giá trị bất hợp lệ theo các quy tắc đã kiểm tra.
- `amount_usd` và `account_balance_usd` lệch phải mạnh. Chưa có căn cứ coi các giá trị lớn là lỗi.
- `cvv_retry_count` và `prior_disputes` có nhiều số 0, phù hợp với ý nghĩa cột đếm. Số 0 của `velocity_score` cần xác minh thêm.
- Tương quan Pearson giữa `txn_count_last_24h` và `velocity_score` khoảng 0,62; chưa có căn cứ loại một trong hai cột.

## Kết luận bàn giao

Chưa phát hiện vấn đề chất lượng buộc phải sửa trong các kiểm tra đã thực hiện. Giữ nguyên 24 đặc trưng đầu vào dự kiến; tách `id` để theo dõi dòng và `is_fraud` làm nhãn. Chưa loại ngoại lệ, thay số 0 hoặc áp dụng biến đổi log.

Đề xuất chia ngẫu nhiên phân tầng theo nhãn nếu không tìm thấy cấu trúc nhóm, thời gian hoặc nguy cơ rò rỉ đòi hỏi cách chia khác.

## Notebook và cách chạy

Notebook: `task1_Bang.ipynb`.

Import notebook vào Kaggle, gắn dữ liệu cuộc thi Playground Series S6E11, kiểm tra đường dẫn và chạy toàn bộ notebook từ đầu.

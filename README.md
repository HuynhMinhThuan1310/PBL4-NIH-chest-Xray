# PBL4 - Phân loại bệnh lý trên ảnh X-quang ngực

Dự án xây dựng và đánh giá các mô hình học sâu cho bài toán phân loại ảnh X-quang ngực thành ba nhóm:

- Bình thường
- Viêm phổi
- Tràn dịch màng phổi

Báo cáo chính so sánh hai mô hình:

- `VGG19-GAP`
- `DenseNet121-FCSSAM`

Trong đó, `VGG19-GAP` là mô hình CNN độc lập sử dụng backbone VGG19 kết hợp Global Average Pooling. `DenseNet121-FCSSAM` được xây dựng theo hướng FA-Net/FCSSAM, bổ sung chú ý theo kênh, chú ý theo không gian và chọn lọc kênh mờ.

## Cấu trúc thư mục

```text
.
├── AGENTS.md
├── notebooks/
│   ├── data.ipynb
│   ├── vgg-gap.ipynb
│   ├── densenet.ipynb
│   ├── gradcam-vgg.ipynb
│   ├── gradcam-densenet.ipynb
│   ├── vgg-fcssam.ipynb
│   └── so-sanh/
│       ├── vgg-2-1-3.ipynb
│       ├── densenet-2-1-3.ipynb
│       ├── vgg-3-1-3.ipynb
│       └── densenet-3-1-3.ipynb
└── report/
    ├── bao-cao.docx
    ├── figs/
    ├── refs/
    └── backups/
Notebook chính
- notebooks/data.ipynb: tạo và chia dữ liệu.
- notebooks/vgg-gap.ipynb: huấn luyện và đánh giá mô hình VGG19-GAP.
- notebooks/densenet.ipynb: huấn luyện và đánh giá mô hình DenseNet121-FCSSAM.
- notebooks/gradcam-vgg.ipynb: Grad-CAM++ và FACT deletion cho VGG19-GAP.
- notebooks/gradcam-densenet.ipynb: Grad-CAM++, bản đồ chú ý FCSSAM và FACT deletion cho DenseNet121-FCSSAM.
- notebooks/vgg-fcssam.ipynb: thử nghiệm phụ với VGG19-FCSSAM.
Dữ liệu
Bộ dữ liệu được xây dựng từ hai nguồn:
- NIH Chest X-rays
- Chest X-Ray Pneumonia
Phiên bản chính trong báo cáo sử dụng tỉ lệ huấn luyện:
bình thường : viêm phổi : tràn dịch màng phổi = 2 : 1 : 1
Các phiên bản 2 : 1 : 3 và 3 : 1 : 3 chỉ được dùng làm thí nghiệm so sánh bổ sung.
Tiền xử lý
Hai mô hình chính dùng cùng một quy trình tiền xử lý để đảm bảo so sánh công bằng:
- Đọc ảnh mức xám.
- Resize về 256 x 256.
- Tăng tương phản bằng min-max normalization và gamma correction.
- Chuyển ảnh mức xám thành ảnh ba kênh.
- Tăng cường dữ liệu nhẹ trên tập huấn luyện.
- Chuẩn hóa theo dạng Caffe/ImageNet.
Kết quả chính
Kết quả trên tập kiểm tra của bộ dữ liệu chính 2 : 1 : 1:
Mô hình
VGG19-GAP
DenseNet121-FCSSAM
Kết quả riêng cho nhóm tràn dịch màng phổi:
Mô hình
VGG19-GAP
DenseNet121-FCSSAM
DenseNet121-FCSSAM đạt kết quả tổng thể nhỉnh hơn, đặc biệt ở F1 trung bình đều và F1 của nhóm tràn dịch màng phổi. Tuy nhiên, VGG19-GAP có recall cao hơn ở nhóm tràn dịch.
Giải thích mô hình
Báo cáo sử dụng các phương pháp sau để phân tích mô hình sau huấn luyện:
- Grad-CAM++
- Bản đồ chú ý bên trong FCSSAM
- FACT deletion batch
FACT deletion AUC:
Mô hình
VGG19-GAP
DenseNet121-FCSSAM
Giá trị Deletion AUC thấp hơn cho thấy xác suất lớp mục tiêu giảm nhanh hơn khi xóa vùng nóng, tức vùng mô hình tập trung có ảnh hưởng mạnh hơn đến quyết định dự đoán.
Báo cáo
File báo cáo chính nằm tại:
report/bao-cao.docx
Các hình nguồn đã chọn nằm tại:
report/figs/
Tài liệu tham khảo nằm tại:
report/refs/
Ghi chú
Các notebook được thiết kế để chạy trên Kaggle vì đường dẫn dữ liệu và checkpoint phụ thuộc vào môi trường Kaggle. Khi chạy cục bộ, có thể kiểm tra cú pháp notebook nhưng thường không thể huấn luyện đầy đủ nếu thiếu dữ liệu gốc.

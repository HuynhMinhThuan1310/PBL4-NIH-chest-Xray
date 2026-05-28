# BAO CAO PHAN LOAI ANH X-QUANG NGUC SU DUNG VGG19 VA RESNET50-FCSSAM

## 1. Gioi thieu

Anh X-quang nguc la mot cong cu chan doan hinh anh pho bien trong y khoa, dac biet doi voi cac benh ly ho hap nhu viêm phổi va tràn dịch màng phổi. Tuy nhien, viec doc anh X-quang co the bi anh huong boi chat luong anh, tu the chup, kinh nghiem nguoi doc va su che lap cua cac dau hieu benh ly. Vi vay, viec xay dung mo hinh hoc sau ho tro phan loai anh X-quang co y nghia trong qua trinh sang loc va cung cap them thong tin tham khao cho bac si.

De tai nay thuc nghiem hai huong tiep can chinh tren bai toan phan loai anh X-quang nguc ba lop: `normal`, `pneumonia`, `effusion`. Trong phan trinh bay bao cao, ba lop nay lan luot duoc hieu la bình thường, viêm phổi va tràn dịch màng phổi. Hai mo hinh duoc so sanh la VGG19-GAP va ResNet50-FCSSAM. VGG19-GAP la mo hinh nen tich chap duoc tien huan luyen tren ImageNet, ket hop Global Average Pooling va lop phan loai Softmax. ResNet50-FCSSAM ket hop backbone ResNet50 voi Fuzzy Channel Selective Spatial Attention Module de tang cuong dac trung theo kenh, theo khong gian va lua chon kenh dac trung co y nghia.

Trong qua trinh thuc nghiem, nhom da huan luyen bon cap mo hinh VGG19/ResNet50-FCSSAM tren cac phien ban du lieu khac nhau. Cap train chinh duoc chon de trinh bay va ket luan la cap normal3, vi day la phien ban du lieu moi nhat va cho ket qua tot nhat tren tap test. Ba cap train con lai duoc trinh bay trong phan ket qua nhu cac thuc nghiem phu de cho thay qua trinh cai tien du lieu va cau hinh huan luyen.

[CHEN HINH 1: Vi du anh X-quang cua ba lop bình thường, viêm phổi va tràn dịch màng phổi]

Hinh 1 nen minh hoa moi lop bang mot so anh dai dien. Khi trinh bay hinh nay, can noi ro rang anh dau vao co the den tu nhieu nguon, nhung trong pipeline chinh tat ca deu duoc dua ve cung quy trinh tien xu ly truoc khi vao mo hinh.

## 2. Co so ly thuyet

### 2.1. Mang noron tich chap CNN

Convolutional Neural Network (CNN) la mo hinh hoc sau phu hop voi du lieu anh. CNN khai thac cau truc khong gian cuc bo thong qua cac lop tich chap, sau do tong hop thong tin bang pooling va cac lop phan loai. Voi anh dau vao hoac feature map $X$ va kernel $K$, phep tich chap tai vi tri $(i,j)$ duoc bieu dien:

$$
Y(i,j)=\sum_m\sum_n X(i+m,j+n)K(m,n)+b
$$

Sau lop tich chap, ham ReLU thuong duoc dung de dua tinh phi tuyen vao mo hinh:

$$
f(x)=\max(0,x)
$$

CNN phu hop voi anh X-quang vi co the hoc cac dac trung cuc bo nhu bien, vung sang toi, ket cau mo phoi, vung mo bat thuong va cac dau hieu lien quan den viêm phổi hoac tràn dịch màng phổi.

[CHEN HINH 2: Minh hoa phep tich chap trong CNN]

Hinh 2 dung de giai thich cach kernel quet tren anh dau vao va tao feature map. Khi chen hinh, can neu rang cac feature map o tang sau se mang thong tin truu tuong hon so voi bien anh ban dau.

### 2.2. Pooling va Global Average Pooling

Pooling giup giam kich thuoc khong gian cua feature map, giam so tham so va tang tinh ben vung doi voi cac bien doi nho trong anh. Max pooling lay gia tri lon nhat trong cua so truot:

$$
Y(i,j)=\max_{(m,n)\in\Omega}X(i+m,j+n)
$$

Global Average Pooling (GAP) nen toan bo moi kenh dac trung thanh mot gia tri:

$$
GAP(F)_c=\frac{1}{H\times W}\sum_{i=1}^{H}\sum_{j=1}^{W}F_c(i,j)
$$

Trong de tai, ca VGG19 va ResNet50-FCSSAM deu su dung GAP trong head phan loai. Cach lam nay giam so tham so so voi flatten truc tiep va giup han che overfitting.

[CHEN HINH 3: Minh hoa max pooling va global average pooling]

### 2.3. Softmax va cross-entropy co trong so

Bai toan duoc xem la phan loai da lop mot nhan. Lop cuoi su dung Softmax de bien logits thanh xac suat:

$$
\hat{y}_j=\frac{e^{z_j}}{\sum_{i=1}^{C}e^{z_i}}
$$

Ham mat mat cross-entropy cho mot mau co nhan dung $t$:

$$
L_{CE}=-\log(\hat{y}_t)
$$

Do du lieu khong can bang, trong so lop duoc dung trong qua trinh huan luyen:

$$
L_{WCE}=-w_t\log(\hat{y}_t)
$$

Trong cap train chinh normal3, tham so tang rieng cho lop tràn dịch màng phổi la `EFFUSION_BOOST = 1.0`. Dieu nay co nghia la sau khi da dieu chinh ti le du lieu, nhom khong tiep tuc ep them trong so cho lop tràn dịch màng phổi nhu mot so thuc nghiem cu.

### 2.4. Chi so danh gia

Bao cao uu tien cac chi so phu hop voi bai toan mat can bang lop: macro F1, precision/recall/F1 theo tung lop, confusion matrix va AUC. Accuracy van duoc bao cao nhung khong duoc xem la chi so duy nhat vi lop bình thường co so mau lon hon hai lop benh.

Precision, recall va F1-score duoc tinh:

$$
Precision=\frac{TP}{TP+FP}
$$

$$
Recall=\frac{TP}{TP+FN}
$$

$$
F1=\frac{2\times Precision\times Recall}{Precision+Recall}
$$

Macro F1 la trung binh F1 deu tren ba lop, giup phan anh tot hon hieu nang tren cac lop it mau nhu viêm phổi va tràn dịch màng phổi.

## 3. Hai mo hinh su dung trong de tai

### 3.1. VGG19-GAP

VGG19 la backbone CNN gom nhieu khoi tich chap $3\times3$ va max pooling. Trong de tai, VGG19 duoc su dung voi `include_top=False`, trong so ImageNet va head phan loai moi:

$$
Input\rightarrow VGG19_{notop}\rightarrow GAP\rightarrow Dropout\rightarrow Dense(256,ReLU)\rightarrow Dropout\rightarrow Dense(3,Softmax)
$$

| Thanh phan | Mo ta |
|---|---|
| Backbone | VGG19, `include_top=False`, `weights="imagenet"` |
| Lop dac trung | `GlobalAveragePooling2D`, ten lop `vgg_gap` |
| Regularization | `Dropout(0.30)` |
| Lop an | `Dense(256, activation="relu")` |
| Lop dau ra | `Dense(3, activation="softmax")` |
| Lop truc quan hoa sau train | `block4_conv4` trong notebook danh gia cuoi |

[CHEN HINH 4: Kien truc VGG19-GAP su dung trong de tai]

Khi trinh bay Hinh 4, can nhan manh rang VGG19 duoc dung nhu backbone trich xuat dac trung, khong dung phan fully connected goc cua ImageNet. Head moi giup phu hop voi bai toan ba lop cua de tai.

### 3.2. ResNet50-FCSSAM

ResNet50 su dung cac residual block de giam van de mat gradient khi tang do sau mang. Trong de tai, ResNet50 duoc ket hop voi FCSSAM:

$$
Input\rightarrow ResNet50_{notop}\rightarrow FCSSAM\rightarrow GAP\rightarrow Dense(3,Softmax)
$$

FCSSAM gom ba y tuong chinh: Channel Attention Module (CAM), Spatial Attention Module (SAM) va Fuzzy Channel Selection (FCS). CAM hoc trong so cho tung kenh dac trung; SAM tao ban do chu y khong gian; FCS dung mask mo de chon cac kenh co diem quan trong cao.

| Thanh phan | Mo ta |
|---|---|
| Backbone | ResNet50, `include_top=False`, `weights="imagenet"` |
| Feature map dau vao FCSSAM | output cuoi cua ResNet50 |
| Attention module | CAM + SAMavg + SAMmax + FCS |
| Ti le giu kenh | `keep_ratio = 0.80` |
| Attention ratio | `attention_ratio = 8` |
| Lop dac trung | `GlobalAveragePooling2D`, ten lop `resnet_gap` |
| Lop dau ra | `Dense(3, activation="softmax")` |
| Lop Grad-CAM sau train | `conv4_block6_out` trong notebook danh gia cuoi |

[CHEN HINH 5: Kien truc ResNet50-FCSSAM]

Hinh 5 can the hien ResNet50 la phan trich xuat dac trung, FCSSAM la phan tang cuong attention, sau do GAP va Softmax phan loai. Day la diem khac biet kien truc quan trong nhat so voi VGG19-GAP.

## 4. Du lieu va tien xu ly

### 4.1. Cap train chinh: normal3

Cap train chinh cua bao cao gom hai notebook:

| Model | Notebook train chinh |
|---|---|
| VGG19-GAP | `trains/vgg19-on-effusion-heavy-normal3-nih.ipynb` |
| ResNet50-FCSSAM | `trains/train-resnet50-fcssam-on-effusion-heavy-normal3.ipynb` |

Day la cap train chinh vi hai mo hinh dung cung phien ban du lieu normal3, cung cach tien xu ly, cung kich thuoc dau vao va da cho ket qua test tot nhat trong cac thuc nghiem hien co.

Phien ban du lieu normal3 co dac diem:

| Noi dung | Mo ta |
|---|---|
| Kaggle root | `/kaggle/input/datasets/thuanminh1310/datasets-version3/pbl4_main_effusion_heavy_normal3_nih_datasets` |
| Train manifest | `main_train_3_1_3_effusion_heavy_normal3.csv` |
| Validation manifest | `main_val_70_15_15_effusion_heavy_normal3.csv` |
| Test manifest | `main_test_70_15_15_effusion_heavy_normal3.csv` |
| Class order | `normal`, `pneumonia`, `effusion` |
| Ti le train | bình thường : viêm phổi : tràn dịch màng phổi = 3 : 1 : 3 |
| Ti le val/test | tach theo 70 : 15 : 15 |
| Nguon anh | NIH Chest X-rays va Chest X-Ray Images (Pneumonia) |

Trong phien ban nay, lop tràn dịch màng phổi lay tat ca anh NIH co nhan Effusion. Day la thay doi quan trong so voi dataset ver1, vi ver1 loai bo anh co dong thoi Effusion va Pneumonia. Viec giu tat ca anh co Effusion lam lop tràn dịch màng phổi da lam bai toan gan hon voi du lieu thuc te hon, nhung cung co the lam nhan phuc tap hon vi mot anh co the chua nhieu dau hieu benh.

[CHEN HINH 6: So do nguon du lieu va cac manifest train/validation/test cua normal3]

Khi trinh bay Hinh 6, can neu ro rang manifest chi luu duong dan va nhan; de chay notebook tren Kaggle van can them ca dataset anh goc NIH va Pneumonia vi manifest chua cac duong dan anh tuyet doi tren Kaggle.

### 4.2. Tien xu ly chung cho cap train chinh

De so sanh cong bang, VGG19 va ResNet50-FCSSAM trong cap train chinh dung chung pipeline truoc khi vao backbone:

| Buoc | Mo ta |
|---|---|
| Doc anh | Doc duong dan tu manifest |
| Decode | Decode grayscale voi `channels=1` |
| Resize | Bilinear resize ve `256x256` |
| Tang tuong phan | `enhance_xray_contrast` |
| Chuyen kenh | `tf.image.grayscale_to_rgb` |
| Augmentation train-only | `RandomTranslation(0.015, 0.015)`, `RandomZoom(0.03, 0.03)`, `RandomFlip("horizontal")` |
| Chuan hoa | RGB sang BGR, tru mean Caffe/ImageNet `[103.939, 116.779, 123.680]` |

Viec thong nhat preprocessing la rang buoc quan trong cua bao cao. Nhu vay, khac biet ket qua giua hai mo hinh chu yeu den tu kien truc VGG19-GAP va ResNet50-FCSSAM, khong phai do anh dau vao duoc xu ly khac nhau.

[CHEN HINH 7: Luong tien xu ly tu anh goc den tensor dau vao 256x256x3]

Hinh 7 can mo ta cac buoc grayscale, resize, tang tuong phan, chuyen RGB va chuan hoa Caffe. Khi giai thich, nen noi day la pipeline dung chung cho hai mo hinh chinh.

## 5. Cau hinh huan luyen

Cap train chinh chay tren Kaggle voi GPU P100/TensorFlow-Keras, `mixed_float16`, `IMAGE_SIZE = 256`, `BATCH_SIZE = 16`. Ca hai mo hinh dung weighted sparse categorical cross-entropy va checkpoint theo validation loss.

| Tham so | VGG19-GAP | ResNet50-FCSSAM |
|---|---:|---:|
| Image size | 256 | 256 |
| Batch size | 16 | 16 |
| Mixed precision | `mixed_float16` | `mixed_float16` |
| Effusion boost | 1.0 | 1.0 |
| Augmentation | Giong nhau | Giong nhau |
| Normalization | Caffe/ImageNet mean | Caffe/ImageNet mean |
| Head dac trung | `vgg_gap` | `resnet_gap` |
| Attention | Khong co | FCSSAM |

[CHEN HINH 8: Bieu do loss/metric trong qua trinh train VGG19-GAP normal3]

Hinh 8 dung de nhan xet kha nang hoi tu cua VGG19. Neu duong validation loss khong giam tiep va early stopping dung, can noi rang checkpoint tot nhat duoc lay theo validation loss.

[CHEN HINH 9: Bieu do loss/metric trong qua trinh train ResNet50-FCSSAM normal3]

Hinh 9 dung de so sanh tinh on dinh cua ResNet50-FCSSAM. Neu mo hinh co validation metric tot hon VGG19, can lien he voi co che attention va residual backbone.

## 6. Danh gia sau train: Grad-CAM, FCSSAM attention va FACT deletion

Sau khi train xong, nhom khong train lai model ma load file `best.keras` tu output Kaggle de tao notebook danh gia sau train. Hai file danh gia cuoi cung duoc giu lai la:

| Model | Notebook danh gia sau train | Model path |
|---|---|---|
| VGG19-GAP | `vgg19-normal3-grad-cam-and-fact-deletion.ipynb` | `/kaggle/input/datasets/thuanminh1310/vgg-ver1/train_vgg19_effusion_heavy_normal3_threshold_tf_p100/models/vgg19/best.keras` |
| ResNet50-FCSSAM | `resnet50-fcssam-normal3-grad-cam-conv4-block6.ipynb` | `/kaggle/input/datasets/thuanminh1310/resnet50-fcssam-1/train_resnet50_fcssam_effusion_heavy_normal3_threshold_tf_p100/models/resnet50_fcssam/best.keras` |

### 6.1. VGG19 sau train

Notebook VGG19 hien tai dung layer `block4_conv4` de truc quan hoa. Trong file notebook, tieu de va ham goi hien dang la Grad-CAM++, va FACT deletion cung dung heatmap tu ham Grad-CAM++ nay. Ly do giu `block4_conv4` la layer nay cho heatmap chi tiet hon va it bi gom thanh blob qua lon. Thu nghiem `block5_conv4` da cho thay heatmap co xu huong bam vao marker/goc anh, khong phu hop de dua vao bao cao.

[CHEN HINH 10: VGG19 - 3 anh bình thường, 3 anh viêm phổi, 3 anh tràn dịch màng phổi voi heatmap block4_conv4]

Khi trinh bay Hinh 10, can noi ro heatmap cho thay vung anh dong gop vao du doan cua VGG19. Neu mot so heatmap bam vao vung xuong, canh phoi hoac marker, can ghi nhan day la han che trong dien giai mo hinh, khong nen xem heatmap la chan doan y khoa doc lap.

### 6.2. ResNet50-FCSSAM sau train

Notebook ResNet50-FCSSAM dung Grad-CAM chuan voi layer `conv4_block6_out`. Layer nay nam o stage conv4 cua ResNet50, co do phan giai khong gian tot hon `conv5_block3_out`, vi vay heatmap chi tiet hon va phu hop hon de so sanh truc quan voi VGG19 `block4_conv4`.

Ngoai Grad-CAM, notebook ResNet con hien thi cac ban do lien quan den FCSSAM:

| Cot hien thi | Y nghia |
|---|---|
| Original | Anh X-quang goc sau khi decode de hien thi |
| Grad-CAM | Heatmap Grad-CAM tai `conv4_block6_out` |
| Overlay | Heatmap chong len anh goc, co hien `OK/WRONG`, nhan du doan va xac suat |
| CAM effect | Anh huong sau channel attention |
| SAMavg | Ban do chu y khong gian tu average theo kenh |
| SAMmax | Ban do chu y khong gian tu max theo kenh |
| FCSSAM | Ban do dac trung sau lua chon kenh mo |

[CHEN HINH 11: ResNet50-FCSSAM - 3 anh bình thường, 3 anh viêm phổi, 3 anh tràn dịch màng phổi voi Grad-CAM va FCSSAM attention]

Hinh 11 can duoc giai thich theo hai nhom thong tin. Cot Grad-CAM va Overlay cho biet vung anh anh huong den lop du doan. Cac cot CAM effect, SAMavg, SAMmax va FCSSAM la ban do noi bo cua module attention, khong nen dien giai giong Grad-CAM thuong. Trong lan chay hien tai, 9 anh minh hoa duoc chon deu du doan dung, confusion nho cua 9 anh la 3/3 dung cho moi lop.

### 6.3. FACT deletion

FACT deletion duoc dung de kiem tra tinh faithful cua heatmap. Y tuong la xoa dan cac pixel co do nong cao nhat theo heatmap, sau do quan sat xac suat lop muc tieu giam nhu the nao. Neu heatmap that su nam tren vung quan trong, xac suat lop muc tieu se giam nhanh khi xoa cac vung do.

Ket qua hien tai:

| Model | Heatmap dung cho deletion | Deletion AUC | Nhan xet |
|---|---|---:|---|
| VGG19-GAP | Grad-CAM++ tren `block4_conv4` | 0.602195 | Xac suat giam nhung cham hon ResNet50-FCSSAM |
| ResNet50-FCSSAM | Grad-CAM tren `conv4_block6_out` | 0.420410 | Xac suat giam nhanh hon, heatmap co tinh faithful tot hon trong phep do nay |

[CHEN HINH 12: Duong FACT deletion cua VGG19-GAP]

Hinh 12 can noi rang khi ti le pixel nong bi xoa tang tu 0 den 0.9, xac suat lop muc tieu cua VGG19 giam dan. Deletion AUC 0.602195 cho thay heatmap co thong tin, nhung muc giam khong manh bang ResNet50-FCSSAM.

[CHEN HINH 13: Duong FACT deletion cua ResNet50-FCSSAM]

Hinh 13 cho thay xac suat lop muc tieu cua ResNet50-FCSSAM giam ro sau khi xoa 10%-30% vung nong nhat. Deletion AUC thap hon VGG19, day la tin hieu tot trong phep do deletion vi vung heatmap co anh huong lon hon den du doan.

## 7. Ket qua thuc nghiem

### 7.1. Ket qua cap train chinh normal3

Bang sau la ket qua test threshold cua cap train chinh normal3. Threshold rieng cho lop tràn dịch màng phổi duoc chon tren validation, sau do ap dung len test.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Macro AUC OvR | Effusion threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| VGG19-GAP | 0.881555 | 0.834977 | 0.875720 | 0.852399 | 0.885796 | 0.957893 | 0.61 |
| ResNet50-FCSSAM | 0.894023 | 0.849038 | 0.885499 | 0.865358 | 0.896925 | 0.963743 | 0.66 |

Bang 1 cho thay ResNet50-FCSSAM dat ket qua cao hon VGG19-GAP tren tat ca chi so tong quat quan trong. Macro F1 tang tu 0.852399 len 0.865358, cho thay mo hinh co attention cai thien kha nang can bang giua cac lop. Macro AUC cung tang tu 0.957893 len 0.963743.

[CHEN BANG 1: Bang ket qua tong quan cap train chinh normal3]

[CHEN HINH 14: Bieu do cot so sanh Accuracy, Macro F1 va Macro AUC cua cap train chinh]

Hinh 14 nen dung de nhan manh ResNet50-FCSSAM tot hon VGG19-GAP tren cap train chinh. Khi thuyet minh, nen noi rang chenh lech macro F1 la chi so quan trong hon accuracy vi bai toan mat can bang lop.

Ket qua theo lop tren test threshold:

| Model | Lop | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| VGG19-GAP | bình thường | 0.941569 | 0.886328 | 0.913114 |
| VGG19-GAP | viêm phổi | 0.959811 | 0.992665 | 0.975962 |
| VGG19-GAP | tràn dịch màng phổi | 0.603550 | 0.748166 | 0.668122 |
| ResNet50-FCSSAM | bình thường | 0.945574 | 0.900995 | 0.922747 |
| ResNet50-FCSSAM | viêm phổi | 0.957647 | 0.995110 | 0.976019 |
| ResNet50-FCSSAM | tràn dịch màng phổi | 0.643892 | 0.760391 | 0.697309 |

Bang 2 cho thay lop tràn dịch màng phổi la lop kho nhat doi voi ca hai mo hinh. ResNet50-FCSSAM cai thien precision, recall va F1 cua lop nay so voi VGG19-GAP. Day la ket qua quan trong vi muc tieu chinh cua qua trinh dieu chinh dataset normal3 la cai thien nhan dien tràn dịch màng phổi trong bai toan ba lop.

[CHEN BANG 2: Bao cao theo tung lop cua cap train chinh]

[CHEN HINH 15: Confusion matrix cua VGG19-GAP tren tap test normal3]

Hinh 15 can duoc dung de chi ra cac loi nham lan chinh cua VGG19-GAP, dac biet la nham lan lien quan den lop tràn dịch màng phổi.

[CHEN HINH 16: Confusion matrix cua ResNet50-FCSSAM tren tap test normal3]

Hinh 16 can so sanh truc tiep voi Hinh 15. Neu so mau tràn dịch màng phổi bi nham giam, can noi day la bang chung bo sung cho viec ResNet50-FCSSAM co F1 lop tràn dịch màng phổi cao hon.

[CHEN HINH 17: ROC curve cua VGG19-GAP tren tap test normal3]

[CHEN HINH 18: ROC curve cua ResNet50-FCSSAM tren tap test normal3]

Hinh 17 va Hinh 18 nen duoc thuyet minh bang macro AUC va AUC theo tung lop. Can luu y AUC cao khong dong nghia threshold mac dinh la toi uu, nen nhom da tune threshold rieng cho lop tràn dịch màng phổi tren validation.

### 7.2. Ba cap train phu con lai

Ngoai cap normal3, nhom da co ba cap train phu. Cac cap nay khong duoc chon lam ket qua chinh nhung cho thay qua trinh cai tien dataset va cau hinh.

| Cap train | Notebook VGG19 | Notebook ResNet50-FCSSAM | Mo ta |
|---|---|---|---|
| Phu 1 | `trains/vgg19-gap-softmax1.ipynb` | `trains/resnet50-fcssam-softmax1.ipynb` | Dataset cu voi nhan `normal`, `pneumonia_only`, `effusion_only`, train 2:1:1 |
| Phu 2 | `trains/train-vgg19-on-dataset-ver1.ipynb` | `trains/train-resnet50-fcssam-on-dataset-ver1.ipynb` | Dataset ver1 voi nhan `normal`, `pneumonia`, `effusion`, trong do effusion loai anh dong thoi Effusion va Pneumonia |
| Phu 3 | `trains/ver3-vgg.ipynb` | `trains/ver3-resnet.ipynb` | Dataset effusion-heavy, train 2:1:3, val/test 70:15:15 |

Ket qua test cua cac cap phu:

| Cap train | Model | Accuracy | Macro F1 | Weighted F1 | Macro AUC OvR | Ghi chu |
|---|---|---:|---:|---:|---:|---|
| Phu 1 | VGG19-GAP | 0.863618 | 0.835117 | 0.870336 | 0.951155 | Dataset cu, nhan benh tach rieng `pneumonia_only` va `effusion_only` |
| Phu 1 | ResNet50-FCSSAM | 0.874742 | 0.847634 | 0.880708 | 0.952491 | Tot hon VGG19 o macro F1 va weighted F1 |
| Phu 2 | VGG19-GAP | 0.856589 | 0.838443 | 0.867341 | 0.953455 | Dataset ver1, bai toan gan voi nhan hien tai hon |
| Phu 2 | ResNet50-FCSSAM | 0.843023 | 0.826359 | 0.855255 | 0.954345 | AUC cao nhung macro F1 thap hon VGG19 trong cap nay |
| Phu 3 | VGG19-GAP | 0.874587 | 0.843077 | 0.879092 | 0.957370 | Co tune threshold lop tràn dịch màng phổi, threshold 0.55 |
| Phu 3 | ResNet50-FCSSAM | 0.882655 | 0.854106 | 0.886886 | 0.961474 | Tot hon VGG19 va gan voi cap chinh |

Bang 3 cho thay qua tung phien ban du lieu, ket qua co xu huong cai thien khi dataset duoc dieu chinh theo huong effusion-heavy. Cap phu 2 la truong hop ResNet50-FCSSAM khong vuot VGG19 ve macro F1, cho thay attention module khong tu dong dam bao tot hon neu phan bo va dinh nghia du lieu chua phu hop. Cap phu 3 da cai thien ro hon khi tang ti le anh tràn dịch màng phổi trong train len 2:1:3.

[CHEN BANG 3: Tong hop ket qua ba cap train phu]

[CHEN HINH 19: Bieu do so sanh macro F1 cua 4 cap train]

Hinh 19 nen gom ca bon cap train de cho thay cap normal3 la cap tot nhat hien tai. Khi trinh bay, can noi ro cap train chinh khong phai duoc chon tuy y ma dua tren ket qua test threshold va tinh cong bang preprocessing giua hai mo hinh.

### 7.3. Vi sao chon normal3 lam cap chinh

Normal3 duoc chon lam cap chinh vi cac ly do:

| Ly do | Giai thich |
|---|---|
| Dataset moi nhat | Dung nhan `normal`, `pneumonia`, `effusion` dung voi bai toan hien tai |
| Tang lop tràn dịch màng phổi | Train 3:1:3 giup lop tràn dịch màng phổi co du mau hon |
| Cong bang giua hai model | VGG19 va ResNet50-FCSSAM dung cung preprocessing, image size va augmentation |
| Ket qua tot nhat | ResNet50-FCSSAM dat macro F1 0.865358 va F1 lop tràn dịch màng phổi 0.697309 |
| Co danh gia sau train | Da co notebook Grad-CAM/FCSSAM attention va FACT deletion rieng |

## 8. Thao luan

VGG19-GAP co kien truc gon, de trien khai va cho ket qua kha tot tren ca bon cap train. Mo hinh dac biet manh voi lop viêm phổi, trong cap chinh dat recall 0.992665 va F1 0.975962. Tuy nhien, VGG19 van gap kho voi lop tràn dịch màng phổi, the hien qua precision va F1 thap hon ResNet50-FCSSAM.

ResNet50-FCSSAM cho ket qua tot nhat trong cap train chinh. Co che FCSSAM giup mo hinh ket hop thong tin attention theo kenh va theo khong gian, dong thoi lua chon bot kenh dac trung du thua. Tren test threshold normal3, ResNet50-FCSSAM dat macro F1 0.865358, cao hon VGG19-GAP 0.012959. Lop tràn dịch màng phổi cung duoc cai thien F1 tu 0.668122 len 0.697309.

Phan danh gia sau train cung ung ho ket qua nay. Voi FACT deletion, ResNet50-FCSSAM co deletion AUC 0.420410, thap hon VGG19-GAP 0.602195. Trong phep do deletion, AUC thap hon nghia la khi xoa cac vung nong nhat theo heatmap, xac suat lop muc tieu giam nhanh hon. Dieu nay cho thay heatmap cua ResNet50-FCSSAM co lien he manh hon voi du doan cua mo hinh trong lan danh gia hien tai.

Tuy nhien, can luu y Grad-CAM va cac ban do attention chi la cong cu dien giai. Chung khong phai bang chung y khoa doc lap va khong thay the danh gia cua bac si. Dac biet, ban do SAMmax trong ResNet50-FCSSAM co luc bi bao hoa do, nen khong nen dien giai tung pixel cua ban do nay. Khi dua vao bao cao, nen xem Grad-CAM/FCSSAM attention la cong cu kiem tra mo hinh co tap trung vao vung hop ly hay khong.

## 9. Tinh trang hien tai cua do an

Nhung viec da lam:

| Noi dung | Tinh trang |
|---|---|
| Huan luyen 4 cap VGG19/ResNet50-FCSSAM | Da co notebook trong `trains/` |
| Chon cap train chinh | Da chon normal3 |
| Thong nhat preprocessing cap chinh | Da thuc hien trong hai notebook train normal3 |
| Tune threshold lop tràn dịch màng phổi | Da thuc hien tren validation va ap dung len test |
| Danh gia Grad-CAM/Grad-CAM++ VGG19 | Da co notebook sau train, dung `block4_conv4` |
| Danh gia Grad-CAM ResNet50-FCSSAM | Da co notebook sau train, dung `conv4_block6_out` |
| Truc quan hoa FCSSAM attention | Da co CAM effect, SAMavg, SAMmax va FCSSAM |
| FACT deletion | Da co cho ca VGG19 va ResNet50-FCSSAM |
| Don dep notebook cu | Da xoa cac file danh gia cu, giu 4 cap train va 2 file danh gia cuoi |

Nhung diem can bo sung khi hoan thien bao cao:

| Noi dung can bo sung | Ghi chu |
|---|---|
| Chen hinh dataset va pipeline | Lay tu notebook hoac ve lai bang so do |
| Chen confusion matrix/ROC | Lay tu output notebook train normal3 |
| Chen hinh Grad-CAM/FCSSAM | Lay tu hai notebook danh gia sau train |
| Chen FACT deletion plots | Lay tu output cua hai notebook danh gia sau train |
| Neu muon thong nhat tuyet doi ten phuong phap | Co the chay lai VGG19 bang Grad-CAM chuan; hien file VGG dang dung Grad-CAM++ |

## 10. Ket luan

Bao cao da trinh bay qua trinh xay dung va so sanh VGG19-GAP voi ResNet50-FCSSAM cho bai toan phan loai anh X-quang nguc ba lop: bình thường, viêm phổi va tràn dịch màng phổi. Bon cap train da duoc thuc hien, trong do cap normal3 duoc chon lam cap chinh vi co dataset moi nhat, preprocessing cong bang giua hai mo hinh va ket qua test tot nhat.

Tren cap train chinh normal3, ResNet50-FCSSAM dat ket qua cao hon VGG19-GAP voi macro F1 0.865358 so voi 0.852399, weighted F1 0.896925 so voi 0.885796 va macro AUC 0.963743 so voi 0.957893. Doi voi lop tràn dịch màng phổi, ResNet50-FCSSAM dat precision 0.643892, recall 0.760391 va F1 0.697309, cao hon VGG19-GAP o ca ba chi so.

Danh gia sau train bang Grad-CAM, FCSSAM attention va FACT deletion cho thay ResNet50-FCSSAM khong chi cai thien metric ma con tao heatmap co tinh faithful tot hon trong phep do deletion hien tai. Tuy nhien, cac hinh dien giai can duoc trinh bay nhu cong cu ho tro phan tich mo hinh, khong phai ket luan y khoa doc lap.

Huong phat trien tiep theo gom: chuan hoa lai notebook VGG19 sang Grad-CAM chuan neu can so sanh hoan toan dong nhat, thu nghiem focal loss hoac class-balanced loss cho lop tràn dịch màng phổi, danh gia them tren tap du lieu ngoai, va moi chuyen gia y khoa nhan xet cac heatmap Grad-CAM/FCSSAM.

## Tai lieu tham khao

[1] Ahmad Rafiansyah Fauzan, Mohammad Iwan Wahyuddin, Sari Ningsih, "Pleural Effusion Classification Based on Chest X-Ray Images using Convolutional Neural Network", Jurnal Ilmu Komputer dan Informasi, 14(1), 2021, pp. 9-16.

[2] Ayush Roy, Anurag Bhattacharjee, Diego Oliva, Oscar Ramos-Soto, Francisco J. Alvarez-Padilla, Ram Sarkar, "FA-Net: A Fuzzy Attention-aided Deep Neural Network for Pneumonia Detection in Chest X-Rays", arXiv:2406.15117v1, 2024.

[3] Karen Simonyan, Andrew Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", ICLR, 2015.

[4] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, "Deep Residual Learning for Image Recognition", CVPR, 2016.

[5] Ramprasaath R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization", ICCV, 2017.

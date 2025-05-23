import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import io
import soundfile as sf
import requests # Thêm thư viện requests
from pathlib import Path # Thêm Path

# --- Training Flag ---
IS_TRAINING = False

# --- FastAPI App ---
app = FastAPI(title="Phân Loại Âm Thanh Deepfake", description="API và giao diện phân loại âm thanh thật/giả")

# --- Mount static files và templates ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Config ---
SR = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
MAX_FRAMES_SPEC = 313
FMIN = 0.0
FMAX = None # Sẽ được tính là sr/2 nếu None
NORM_EPSILON = 1e-6
DEVICE = torch.device("cpu") # Luôn dùng CPU trên Vercel Serverless
VIT_PATCH_SIZE = 16
VIT_EMBED_DIM = 192
VIT_DEPTH = 6
VIT_NUM_HEADS = 6
VIT_MLP_RATIO = 4.0
VIT_DROP_RATE = 0.1
VIT_ATTN_DROP_RATE = 0.1
CNN_DROPOUT_RATE = 0.4

# --- ĐƯỜNG DẪN MODEL TỪ GITHUB RELEASES ---
# !!! QUAN TRỌNG: THAY THẾ CÁC URL SAU BẰNG URL THỰC TẾ TỪ GITHUB RELEASE CỦA BẠN !!!
MODEL_URL_VIT = "https://github.com/namthse182380/audio_ai_detection/releases/download/1.0.0/best_vit_model_pytorch.pth"
MODEL_URL_CNN = "https://github.com/namthse182380/audio_ai_detection/releases/download/1.0.0/best_cnn_model_pytorch.pth"
# Ví dụ:
# MODEL_URL_VIT = "https://github.com/namthse182380/audio_deepfake_detection/releases/download/v1.0.0/best_vit_model_pytorch.pth"
# MODEL_URL_CNN = "https://github.com/namthse182380/audio_deepfake_detection/releases/download/v1.0.0/best_cnn_model_pytorch.pth"


# Đường dẫn cục bộ trong môi trường Lambda (Vercel cho phép ghi vào /tmp)
LOCAL_MODEL_PATH_VIT = Path("/tmp/best_vit_model_pytorch.pth")
LOCAL_MODEL_PATH_CNN = Path("/tmp/best_cnn_model_pytorch.pth")

# --- Định Nghĩa Mô Hình ViT (giữ nguyên từ code bạn cung cấp) ---
class PatchEmbed(nn.Module):
    def __init__(self, img_size=(N_MELS, MAX_FRAMES_SPEC), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # Để linh hoạt hơn, có thể bỏ assert này nếu kích thước đầu vào có thể thay đổi nhẹ
        # Hoặc đảm bảo img_size truyền vào ViT constructor khớp với dữ liệu thực tế
        if H != self.img_size[0] or W != self.img_size[1]:
             # Cân nhắc resize hoặc padding ở đây nếu cần thiết, hoặc đảm bảo dữ liệu đầu vào luôn đúng kích thước
             # print(f"Cảnh báo: Kích thước ảnh đầu vào ({H}*{W}) không khớp với mô hình mong đợi ({self.img_size[0]}*{self.img_size[1]}).")
             pass # Tạm thời bỏ qua để tránh lỗi cứng, nhưng cần kiểm tra
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=(N_MELS, MAX_FRAMES_SPEC), patch_size=16, in_chans=3, num_classes=1,
                 embed_dim=192, depth=6, num_heads=6, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        # Đảm bảo img_size được truyền đúng vào PatchEmbed
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)]) # Sửa i thành _
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: # Sửa: isinstance(m, nn.Linear) and m.bias is not None
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x) # Sửa lại
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# --- Định Nghĩa Mô Hình CNN (giữ nguyên từ code bạn cung cấp, có sửa lỗi tính toán flattened_size) ---
class AudioCNN(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.4):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.drop1 = nn.Dropout2d(dropout_rate / 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.drop2 = nn.Dropout2d(dropout_rate / 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.drop3 = nn.Dropout2d(dropout_rate)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.drop4 = nn.Dropout2d(dropout_rate)

        # Tính toán kích thước đầu vào cho lớp Linear một cách linh hoạt
        # Tạo một dummy input để xác định kích thước
        with torch.no_grad(): # Quan trọng: không tính gradient khi chạy dummy pass
            # Kích thước dummy input: (batch_size, channels, height, width)
            dummy_input = torch.zeros(1, 1, N_MELS, MAX_FRAMES_SPEC)
            dummy_output = self._forward_conv(dummy_input)
            self.flattened_size = dummy_output.view(dummy_output.size(0), -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.drop_fc1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.drop_fc2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)

    def _forward_conv(self, x): # Helper function để tính flattened_size và dùng trong forward chính
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.drop1(x)
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.drop4(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1) # Flatten
        x = nn.functional.relu(self.bn_fc1(self.fc1(x)))
        x = self.drop_fc1(x)
        x = nn.functional.relu(self.bn_fc2(self.fc2(x)))
        x = self.drop_fc2(x)
        x = self.fc3(x)
        return x


# --- Hàm Tiền Xử Lý Âm Thanh ---
def audio_to_melspectrogram(audio_data_source, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, max_frames=MAX_FRAMES_SPEC, fmin=FMIN, fmax=FMAX):
    try:
        y = None
        sr_orig = sr # Mặc định

        if isinstance(audio_data_source, bytes): # Từ UploadFile
            y, sr_orig = sf.read(io.BytesIO(audio_data_source))
        elif isinstance(audio_data_source, (str, Path)): # Từ đường dẫn file (cho training)
            # if not Path(audio_data_source).exists(): # Kiểm tra file tồn tại
            #     print(f"Cảnh báo: File không tồn tại {audio_data_source}")
            #     return None
            y, sr_orig = sf.read(str(audio_data_source)) # sf.read cần string
        else:
            raise ValueError("audio_data_source phải là bytes (từ UploadFile) hoặc đường dẫn file (str/Path).")

        if y.ndim > 1: # Chuyển thành mono nếu là stereo
             y = librosa.to_mono(y.T if y.shape[0] > y.shape[1] else y) # Đảm bảo đúng chiều cho to_mono

        if sr_orig != sr:
            y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr)

        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax if fmax is not None else sr/2.0
        )
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        current_frames = log_mel_spectrogram.shape[1]
        if current_frames < max_frames:
            pad_value = log_mel_spectrogram.min() # Hoặc có thể dùng 0 hoặc giá trị cụ thể
            pad_width = max_frames - current_frames
            padded_log_mel_spectrogram = np.pad(
                log_mel_spectrogram, ((0, 0), (0, pad_width)), mode='constant', constant_values=pad_value
            )
            return padded_log_mel_spectrogram
        elif current_frames > max_frames:
            truncated_log_mel_spectrogram = log_mel_spectrogram[:, :max_frames]
            return truncated_log_mel_spectrogram
        else:
            return log_mel_spectrogram
    except Exception as e:
        error_detail = f"Lỗi xử lý âm thanh: {str(e)} với input type {type(audio_data_source)}"
        # print(error_detail) # Ghi log chi tiết hơn ở server
        # Khi training, trả về None để DataLoader có thể bỏ qua sample lỗi
        if IS_TRAINING and isinstance(audio_data_source, (str, Path)):
            return None
        # Khi inference, raise HTTPException
        raise HTTPException(status_code=400, detail=error_detail)


# --- Hàm tải model từ URL về /tmp ---
def download_file(url, destination: Path):
    """Tải file từ URL về destination nếu chưa tồn tại hoặc nếu URL là placeholder."""
    if url.startswith("YOUR_GITHUB_RELEASE_URL_FOR_"):
        print(f"CẢNH BÁO: URL model '{url}' chưa được cấu hình. Model sẽ không được tải.")
        if destination.exists(): # Xóa file cũ nếu URL là placeholder để tránh dùng nhầm
            try:
                destination.unlink()
                print(f"Đã xóa file model cũ tại {destination} do URL placeholder.")
            except OSError as e:
                print(f"Lỗi khi xóa file model cũ {destination}: {e}")
        return False # Không tải được

    if not destination.exists():
        print(f"Đang tải {url} về {destination}...")
        try:
            response = requests.get(url, stream=True, timeout=60) # Thêm timeout
            response.raise_for_status()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Tải xong: {destination.name}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Lỗi khi tải model từ {url}: {e}")
            # Không raise HTTPException ở đây vì hàm này gọi lúc khởi tạo module
            return False # Không tải được
    else:
        print(f"Model {destination.name} đã tồn tại tại {destination}.")
        return True # Đã tồn tại


# --- Tải Mô Hình ---
def load_model_from_path(model_class_fn, model_path: Path):
    try:
        model = model_class_fn().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        print(f"Lỗi tải mô hình từ {model_path}: {str(e)}")
        # Không raise HTTPException ở đây
        return None # Trả về None nếu không tải được

vit_model = None
cnn_model = None

# --- Chỉ Load Mô Hình Khi Không Huấn Luyện ---
if not IS_TRAINING:
    print("Đang ở chế độ inference, tiến hành tải models...")
    # Tải model ViT
    if download_file(MODEL_URL_VIT, LOCAL_MODEL_PATH_VIT) and LOCAL_MODEL_PATH_VIT.exists():
        print(f"Đang khởi tạo và tải trọng số ViT model từ {LOCAL_MODEL_PATH_VIT}...")
        vit_model = load_model_from_path(
            lambda: VisionTransformer(
                img_size=(N_MELS, MAX_FRAMES_SPEC), patch_size=VIT_PATCH_SIZE, in_chans=3, num_classes=1,
                embed_dim=VIT_EMBED_DIM, depth=VIT_DEPTH, num_heads=VIT_NUM_HEADS, mlp_ratio=VIT_MLP_RATIO,
                qkv_bias=True, drop_rate=VIT_DROP_RATE, attn_drop_rate=VIT_ATTN_DROP_RATE
            ),
            LOCAL_MODEL_PATH_VIT
        )
        if vit_model:
            print("ViT model đã tải thành công.")
        else:
            print("Không thể tải ViT model.")
    else:
        print(f"Không thể tải hoặc không tìm thấy file ViT model tại {LOCAL_MODEL_PATH_VIT} sau khi cố gắng download.")

    # Tải model CNN
    if download_file(MODEL_URL_CNN, LOCAL_MODEL_PATH_CNN) and LOCAL_MODEL_PATH_CNN.exists():
        print(f"Đang khởi tạo và tải trọng số CNN model từ {LOCAL_MODEL_PATH_CNN}...")
        cnn_model = load_model_from_path(
            lambda: AudioCNN(num_classes=1, dropout_rate=CNN_DROPOUT_RATE),
            LOCAL_MODEL_PATH_CNN
        )
        if cnn_model:
            print("CNN model đã tải thành công.")
        else:
            print("Không thể tải CNN model.")
    else:
        print(f"Không thể tải hoặc không tìm thấy file CNN model tại {LOCAL_MODEL_PATH_CNN} sau khi cố gắng download.")

    if not vit_model and not cnn_model:
        print("CẢNH BÁO NGHIÊM TRỌNG: Không có model nào được tải thành công!")
    elif not vit_model:
        print("CẢNH BÁO: ViT model không được tải. Chỉ CNN model khả dụng (nếu có).")
    elif not cnn_model:
        print("CẢNH BÁO: CNN model không được tải. Chỉ ViT model khả dụng (nếu có).")


# --- Pydantic Model Cho Response (giữ nguyên) ---
class PredictionResponse(BaseModel):
    model: str
    ket_qua: str
    xac_suat_gia: float

# --- Trang Chủ (Frontend) (giữ nguyên) ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- API Endpoint Cho Form HTML ---
@app.post("/predict/", response_class=HTMLResponse)
async def predict_audio_from_form( # Đổi tên để tránh trùng với hàm predict_audio (nếu có)
    request: Request,
    file: UploadFile = File(...),
    model_type: str = Form("both")
):
    if not file.filename.endswith((".wav", ".mp3", ".flac")):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Vui lòng tải lên file WAV, MP3 hoặc FLAC."}
        )
    audio_data = await file.read()
    try:
        mel_spec = audio_to_melspectrogram(audio_data)
        if mel_spec is None:
             return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "Lỗi xử lý file âm thanh, không thể tạo spectrogram."}
            )
    except HTTPException as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": str(e.detail)}
        )

    mean = np.mean(mel_spec)
    std = np.std(mel_spec)
    if std < NORM_EPSILON: # Tránh chia cho 0 nếu std quá nhỏ (ví dụ file im lặng)
        print("Cảnh báo: Độ lệch chuẩn của mel spectrogram quá nhỏ. Sử dụng NORM_EPSILON.")
        std = NORM_EPSILON
    mel_spec_normalized = (mel_spec - mean) / std # Bỏ NORM_EPSILON ở mẫu số nếu đã xử lý ở trên

    results = []

    if model_type in ["vit", "both"]:
        if vit_model is None:
            results.append({"model": "ViT", "ket_qua": "Lỗi: Model ViT chưa được tải hoặc tải thất bại.", "xac_suat_gia": -1.0})
        else:
            try:
                mel_spec_vit = np.stack([mel_spec_normalized]*3, axis=0) # (3, H, W)
                mel_spec_tensor = torch.tensor(mel_spec_vit, dtype=torch.float32).unsqueeze(0).to(DEVICE) # (1, 3, H, W)
                with torch.no_grad():
                    output = vit_model(mel_spec_tensor)
                    prob = torch.sigmoid(output).item()
                    prediction = "Giả" if prob > 0.5 else "Thật"
                    results.append({
                        "model": "ViT",
                        "ket_qua": prediction,
                        "xac_suat_gia": prob
                    })
            except Exception as e:
                print(f"Lỗi khi dự đoán với ViT model: {e}")
                results.append({"model": "ViT", "ket_qua": f"Lỗi dự đoán: {str(e)[:100]}...", "xac_suat_gia": -1.0})


    if model_type in ["cnn", "both"]:
        if cnn_model is None:
            results.append({"model": "CNN", "ket_qua": "Lỗi: Model CNN chưa được tải hoặc tải thất bại.", "xac_suat_gia": -1.0})
        else:
            try:
                mel_spec_cnn = np.expand_dims(mel_spec_normalized, axis=0) # (1, H, W)
                mel_spec_tensor = torch.tensor(mel_spec_cnn, dtype=torch.float32).unsqueeze(0).to(DEVICE) # (1, 1, H, W)
                with torch.no_grad():
                    output = cnn_model(mel_spec_tensor)
                    prob = torch.sigmoid(output).item()
                    prediction = "Giả" if prob > 0.5 else "Thật"
                    results.append({
                        "model": "CNN",
                        "ket_qua": prediction,
                        "xac_suat_gia": prob
                    })
            except Exception as e:
                print(f"Lỗi khi dự đoán với CNN model: {e}")
                results.append({"model": "CNN", "ket_qua": f"Lỗi dự đoán: {str(e)[:100]}...", "xac_suat_gia": -1.0})

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "predictions": results,
            "filename": file.filename,
            "selected_model_type": model_type # Để giữ lại lựa chọn model trên form
        }
    )

# --- API Endpoint Cho Client ---
@app.post("/api/predict/", response_model=list[PredictionResponse])
async def api_predict(file: UploadFile = File(...), model_type: str = Query("both", enum=["vit", "cnn", "both"])): # Thêm Query và enum
    if not file.filename.endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(status_code=400, detail="Vui lòng tải lên file WAV, MP3 hoặc FLAC.")
    audio_data = await file.read()
    try:
        mel_spec = audio_to_melspectrogram(audio_data)
        if mel_spec is None:
            raise HTTPException(status_code=500, detail="Lỗi xử lý file âm thanh, không thể tạo spectrogram.")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi không xác định khi xử lý âm thanh: {str(e)}")

    mean = np.mean(mel_spec)
    std = np.std(mel_spec)
    if std < NORM_EPSILON:
        std = NORM_EPSILON
    mel_spec_normalized = (mel_spec - mean) / std
    results = []

    if model_type in ["vit", "both"]:
        if vit_model is None:
            raise HTTPException(status_code=503, detail="Model ViT hiện không khả dụng hoặc tải thất bại.")
        try:
            mel_spec_vit = np.stack([mel_spec_normalized]*3, axis=0)
            mel_spec_tensor = torch.tensor(mel_spec_vit, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = vit_model(mel_spec_tensor)
                prob = torch.sigmoid(output).item()
                prediction = "Giả" if prob > 0.5 else "Thật"
                results.append({"model": "ViT", "ket_qua": prediction, "xac_suat_gia": prob})
        except Exception as e:
            print(f"Lỗi API khi dự đoán với ViT model: {e}")
            # Trả về lỗi cụ thể hơn hoặc một PredictionResponse với thông báo lỗi
            # Ví dụ: results.append({"model": "ViT", "ket_qua": f"Lỗi dự đoán ViT", "xac_suat_gia": -1.0})
            # Hoặc raise HTTPException nếu muốn dừng hẳn
            raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán với ViT: {str(e)[:100]}...")


    if model_type in ["cnn", "both"]:
        if cnn_model is None:
            raise HTTPException(status_code=503, detail="Model CNN hiện không khả dụng hoặc tải thất bại.")
        try:
            mel_spec_cnn = np.expand_dims(mel_spec_normalized, axis=0)
            mel_spec_tensor = torch.tensor(mel_spec_cnn, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = cnn_model(mel_spec_tensor)
                prob = torch.sigmoid(output).item()
                prediction = "Giả" if prob > 0.5 else "Thật"
                results.append({"model": "CNN", "ket_qua": prediction, "xac_suat_gia": prob})
        except Exception as e:
            print(f"Lỗi API khi dự đoán với CNN model: {e}")
            raise HTTPException(status_code=500, detail=f"Lỗi khi dự đoán với CNN: {str(e)[:100]}...")

    if not results and model_type != "none": # Trường hợp model_type hợp lệ nhưng không có model nào chạy
        raise HTTPException(status_code=503, detail=f"Không có model nào được chọn ({model_type}) hoặc các model đã chọn không khả dụng.")

    return results

# --- Phần Huấn Luyện (Chỉ Chạy Nếu IS_TRAINING = True) ---
if IS_TRAINING:
    # ... (Giữ nguyên code training của bạn, chỉ đảm bảo nó chạy độc lập khi IS_TRAINING=True)
    # ... Sau khi training, bạn sẽ upload thủ công các file .pth lên GitHub Releases.
    import random
    import glob
    import time
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    from fastapi import Query # Thêm import này nếu dùng Query trong API endpoint

    # --- Định Nghĩa Dataset và Training Loop ---
    class AudioDataset(Dataset):
        def __init__(self, filepaths, labels, transform_spectrogram_fn, augment=False, is_vit_input=False):
            self.filepaths = filepaths
            self.labels = labels
            self.transform_spectrogram_fn = transform_spectrogram_fn # audio_to_melspectrogram
            self.augment = augment
            self.is_vit_input = is_vit_input

        def __len__(self):
            return len(self.filepaths)

        def __getitem__(self, idx):
            filepath = self.filepaths[idx]
            label = self.labels[idx]

            mel_spec = self.transform_spectrogram_fn(filepath) # Truyền filepath
            if mel_spec is None:
                # print(f"Cảnh báo (Dataset): Không thể xử lý file {filepath}, bỏ qua.")
                return None

            # Augmentation (nếu có) có thể được thêm ở đây

            mean = np.mean(mel_spec)
            std = np.std(mel_spec)
            if std < NORM_EPSILON:
                std = NORM_EPSILON
            mel_spec_normalized = (mel_spec - mean) / std

            if self.is_vit_input:
                mel_spec_final = np.stack([mel_spec_normalized]*3, axis=0)
            else:
                mel_spec_final = np.expand_dims(mel_spec_normalized, axis=0)

            mel_spec_tensor = torch.tensor(mel_spec_final, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)
            return mel_spec_tensor, label_tensor

    def collate_fn_skip_none(batch): # Một collate_fn chung
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            # Cần trả về tensor rỗng với đúng số chiều để tránh lỗi DataLoader
            # Giả sử output là (batch, channels, height, width) và (batch)
            # Điều này khó làm chung chung, nên có collate_fn riêng cho ViT và CNN nếu cần
            # Hoặc đảm bảo dataset không bao giờ trả về batch rỗng hoàn toàn
            # print("Cảnh báo: Batch rỗng sau khi lọc None.")
            # Trả về một tuple tensor rỗng nếu batch hoàn toàn rỗng
            # Kích thước cụ thể (ví dụ: 0,3,N_MELS,MAX_FRAMES_SPEC) phụ thuộc vào model
            # Tạm thời trả về tensor 1D rỗng, có thể gây lỗi nếu model mong đợi nhiều chiều hơn
            return torch.empty(0), torch.empty(0)
        return torch.utils.data.dataloader.default_collate(batch)

    # --- Hàm Huấn Luyện (Rút Gọn) ---
    def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch_num, num_epochs, model_name="Model"):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for i, batch_data in enumerate(train_loader):
            if not batch_data or len(batch_data) < 2 or batch_data[0].nelement() == 0:
                # print(f"Epoch {epoch_num+1} - {model_name}: Batch {i} rỗng hoặc không hợp lệ, bỏ qua.")
                continue
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

        if total_samples == 0:
            print(f"Cảnh báo: Không có sample nào được xử lý trong epoch {epoch_num+1} cho {model_name}.")
            return 0.0, 0.0

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        return epoch_loss, epoch_acc

    # --- Cấu Hình Huấn Luyện ---
    # !!! QUAN TRỌNG: CẬP NHẬT CÁC ĐƯỜNG DẪN NÀY CHO MÔI TRƯỜNG TRAINING CỦA BẠN !!!
    REAL_AUDIO_PATH_TRAIN = "path/to/your/real/audio_files_for_training"
    FAKE_AUDIO_PATH_TRAIN = "path/to/your/fake/audio_files_for_training"
    BATCH_SIZE_TRAIN = 16
    LEARNING_RATE_TRAIN = 1e-4
    EPOCHS_TRAIN = 10 # Nên tăng lên cho training thực tế
    WEIGHT_DECAY_TRAIN = 1e-5

    # Tên file để lưu model sau training (sẽ được upload thủ công lên GitHub Releases)
    TRAINED_MODEL_FILENAME_VIT = "best_vit_model_pytorch.pth" # Giống tên file sẽ download
    TRAINED_MODEL_FILENAME_CNN = "best_cnn_model_pytorch.pth" # Giống tên file sẽ download

    def get_audio_files_and_labels(real_dir, fake_dir):
        real_files = glob.glob(os.path.join(real_dir, '*.wav')) + \
                     glob.glob(os.path.join(real_dir, '*.mp3')) + \
                     glob.glob(os.path.join(real_dir, '*.flac'))
        fake_files = glob.glob(os.path.join(fake_dir, '*.wav')) + \
                     glob.glob(os.path.join(fake_dir, '*.mp3')) + \
                     glob.glob(os.path.join(fake_dir, '*.flac'))
        
        if not real_files and not fake_files:
            print(f"CẢNH BÁO (Training): Không tìm thấy file âm thanh nào trong {real_dir} hoặc {fake_dir}")
            return [], []

        filepaths = real_files + fake_files
        labels = [0] * len(real_files) + [1] * len(fake_files) # 0 for real, 1 for fake
        
        if not filepaths: return [],[] # Tránh lỗi zip nếu rỗng
        combined = list(zip(filepaths, labels))
        random.shuffle(combined)
        filepaths_shuffled, labels_shuffled = zip(*combined)
        return list(filepaths_shuffled), list(labels_shuffled)

    print("Đang ở chế độ training (IS_TRAINING=True).")
    if not Path(REAL_AUDIO_PATH_TRAIN).exists() or not Path(REAL_AUDIO_PATH_TRAIN).is_dir() or \
       not Path(FAKE_AUDIO_PATH_TRAIN).exists() or not Path(FAKE_AUDIO_PATH_TRAIN).is_dir():
        print(f"Đường dẫn REAL_AUDIO_PATH_TRAIN ('{REAL_AUDIO_PATH_TRAIN}') hoặc "
              f"FAKE_AUDIO_PATH_TRAIN ('{FAKE_AUDIO_PATH_TRAIN}') không tồn tại hoặc không phải thư mục. "
              "Bỏ qua phần training.")
    else:
        filepaths_all, labels_all = get_audio_files_and_labels(REAL_AUDIO_PATH_TRAIN, FAKE_AUDIO_PATH_TRAIN)
        if not filepaths_all:
            print("Không có dữ liệu để huấn luyện. Bỏ qua training.")
        else:
            X_train_paths, X_val_paths, y_train, y_val = train_test_split(
                filepaths_all, labels_all, test_size=0.2, random_state=42,
                stratify=labels_all if len(set(labels_all)) > 1 and len(labels_all) > 1 else None
            )

            print(f"Số lượng file training: {len(X_train_paths)}, validation: {len(X_val_paths)}")

            # --- DataLoader for ViT ---
            train_dataset_vit = AudioDataset(X_train_paths, y_train, audio_to_melspectrogram, augment=True, is_vit_input=True)
            val_dataset_vit = AudioDataset(X_val_paths, y_val, audio_to_melspectrogram, augment=False, is_vit_input=True)
            train_loader_vit = DataLoader(train_dataset_vit, batch_size=BATCH_SIZE_TRAIN, shuffle=True, collate_fn=collate_fn_skip_none, num_workers=2, pin_memory=True)
            val_loader_vit = DataLoader(val_dataset_vit, batch_size=BATCH_SIZE_TRAIN, shuffle=False, collate_fn=collate_fn_skip_none, num_workers=2, pin_memory=True)

            # --- DataLoader for CNN ---
            train_dataset_cnn = AudioDataset(X_train_paths, y_train, audio_to_melspectrogram, augment=True, is_vit_input=False)
            val_dataset_cnn = AudioDataset(X_val_paths, y_val, audio_to_melspectrogram, augment=False, is_vit_input=False)
            train_loader_cnn = DataLoader(train_dataset_cnn, batch_size=BATCH_SIZE_TRAIN, shuffle=True, collate_fn=collate_fn_skip_none, num_workers=2, pin_memory=True)
            val_loader_cnn = DataLoader(val_dataset_cnn, batch_size=BATCH_SIZE_TRAIN, shuffle=False, collate_fn=collate_fn_skip_none, num_workers=2, pin_memory=True)

            # --- Huấn Luyện ViT ---
            print("Bắt đầu huấn luyện ViT model...")
            vit_model_train = VisionTransformer(
                img_size=(N_MELS, MAX_FRAMES_SPEC), patch_size=VIT_PATCH_SIZE, in_chans=3, num_classes=1,
                embed_dim=VIT_EMBED_DIM, depth=VIT_DEPTH, num_heads=VIT_NUM_HEADS, mlp_ratio=VIT_MLP_RATIO,
                qkv_bias=True, drop_rate=VIT_DROP_RATE, attn_drop_rate=VIT_ATTN_DROP_RATE
            ).to(DEVICE)
            optimizer_vit = optim.AdamW(vit_model_train.parameters(), lr=LEARNING_RATE_TRAIN, weight_decay=WEIGHT_DECAY_TRAIN)
            criterion_vit = nn.BCEWithLogitsLoss()

            for epoch in range(EPOCHS_TRAIN):
                train_loss, train_acc = train_one_epoch(vit_model_train, train_loader_vit, criterion_vit, optimizer_vit, DEVICE, epoch, EPOCHS_TRAIN, "ViT")
                # Thêm validation ở đây nếu cần
                print(f"Epoch {epoch+1}/{EPOCHS_TRAIN} - ViT - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            torch.save(vit_model_train.state_dict(), TRAINED_MODEL_FILENAME_VIT)
            print(f"Đã lưu ViT model đã huấn luyện vào: {TRAINED_MODEL_FILENAME_VIT}")


            # --- Huấn Luyện CNN ---
            print("Bắt đầu huấn luyện CNN model...")
            cnn_model_train = AudioCNN(num_classes=1, dropout_rate=CNN_DROPOUT_RATE).to(DEVICE)
            optimizer_cnn = optim.AdamW(cnn_model_train.parameters(), lr=LEARNING_RATE_TRAIN, weight_decay=WEIGHT_DECAY_TRAIN)
            criterion_cnn = nn.BCEWithLogitsLoss()

            for epoch in range(EPOCHS_TRAIN):
                train_loss, train_acc = train_one_epoch(cnn_model_train, train_loader_cnn, criterion_cnn, optimizer_cnn, DEVICE, epoch, EPOCHS_TRAIN, "CNN")
                # Thêm validation ở đây nếu cần
                print(f"Epoch {epoch+1}/{EPOCHS_TRAIN} - CNN - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            torch.save(cnn_model_train.state_dict(), TRAINED_MODEL_FILENAME_CNN)
            print(f"Đã lưu CNN model đã huấn luyện vào: {TRAINED_MODEL_FILENAME_CNN}")

            print("Hoàn tất quá trình huấn luyện. Hãy upload các file .pth đã lưu lên GitHub Releases.")


# --- Chạy Local Server (Chỉ khi không training và file này được chạy trực tiếp) ---
if __name__ == "__main__" and not IS_TRAINING:
    import uvicorn
    print("Chạy uvicorn server cục bộ trên http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
elif __name__ == "__main__" and IS_TRAINING:
    print("Đã chạy xong phần training (nếu có dữ liệu). Không khởi động server uvicorn.")

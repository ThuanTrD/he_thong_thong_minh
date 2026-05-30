# image_preprocessing.py
# Module xử lý ảnh đầu vào cho CNN

from PIL import Image
from torchvision import transforms
import io

def get_inference_transform(img_size: int = 224):
    """Lấy transform chuẩn tương thích với quá trình huấn luyện CNN."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def preprocess_image(image_bytes: bytes, img_size: int = 224):
    """Chuyển đổi bytes ảnh từ API thành Tensor PyTorch."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_inference_transform(img_size)
    tensor = transform(img).unsqueeze(0)
    return tensor

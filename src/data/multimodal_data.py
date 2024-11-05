import os
import pandas as pd
import torch
from torchvision import transforms
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from PIL import Image

# Custom Dataset for flat image directory (no class folders)
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # 이미지 파일 목록 가져오기 (._로 시작하는 파일 제외)
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) 
                            if fname.endswith(('.png', '.jpg', '.jpeg')) and not fname.startswith('._')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = 0  # 라벨 매핑이 없으므로 임시로 0으로 설정
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 이미지와 텍스트, 정형 데이터를 함께 로드하는 함수
def multimodal_data_load(args):
    # 텍스트 데이터 로드 (예: books.csv에서 책 제목이나 설명을 사용)
    text_data_path = os.path.join(args.dataset.data_path, "books.csv")
    text_df = pd.read_csv(text_data_path)

    # 이미지 데이터 로드
    image_data_path = os.path.join(args.dataset.data_path, "images")  # 이미지 폴더 경로
    # 이미지 전처리 (resize, normalize 등)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 입력 사이즈에 맞춤
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 커스텀 데이터셋으로 이미지 로드
    image_dataset = CustomImageDataset(image_data_path, transform=transform)

    # BERT를 위한 토크나이저 설정
    tokenizer = BertTokenizer.from_pretrained(args.model_args[args.model].pretrained_model)
    texts = text_df['book_title'].tolist()  # books.csv 파일에서 텍스트 컬럼을 선택 (예: book_title)
    encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=args.model_args[args.model].max_length)

    # input_ids와 attention_mask 생성
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # 정형 데이터 로드 (예: users.csv 또는 train_ratings.csv에서 사용자 정보 로드)
    numerical_data_path = os.path.join(args.dataset.data_path, "users.csv")
    numerical_df = pd.read_csv(numerical_data_path)

    # 숫자형 열만 선택 (object 타입 제외)
    numerical_df = numerical_df.select_dtypes(include=['number'])  # 숫자형 열만 선택
    numerical_features = torch.tensor(numerical_df.values, dtype=torch.float32)

    # 이미지 데이터, 텍스트 데이터, 정형 데이터를 함께 반환
    data = {
        "images": image_dataset,  # CustomImageDataset
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "numerical_features": numerical_features
    }
    
    return data

def multimodal_data_split(args, data):
    # 데이터셋의 각 부분 (이미지, 텍스트, 정형 데이터)에서 필요한 데이터 분리
    images = data['images']
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    numerical_features = data['numerical_features']

    # train_test_split을 사용하여 데이터를 학습/검증 세트로 분리
    train_images, val_images = train_test_split(images, test_size=args.dataset.valid_ratio)
    train_input_ids, val_input_ids = train_test_split(input_ids, test_size=args.dataset.valid_ratio)
    train_attention_mask, val_attention_mask = train_test_split(attention_mask, test_size=args.dataset.valid_ratio)
    train_numerical, val_numerical = train_test_split(numerical_features, test_size=args.dataset.valid_ratio)

    # 분할된 데이터들을 다시 묶어서 반환
    train_data = {
        "images": train_images,
        "input_ids": train_input_ids,
        "attention_mask": train_attention_mask,
        "numerical_features": train_numerical
    }
    
    val_data = {
        "images": val_images,
        "input_ids": val_input_ids,
        "attention_mask": val_attention_mask,
        "numerical_features": val_numerical
    }

    return train_data, val_data

def multimodal_data_loader(args, data):
    # 이미지, 텍스트, 정형 데이터를 각각 텐서로 변환
    images = data['images']
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    numerical_features = data['numerical_features']

    # TensorDataset 생성 (모델 학습에 사용할 데이터셋)
    dataset = TensorDataset(images, input_ids, attention_mask, numerical_features)

    # DataLoader 생성 (모델 학습 시 배치 처리 등을 쉽게 할 수 있도록 DataLoader로 감싸줌)
    dataloader = DataLoader(dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    
    return dataloader

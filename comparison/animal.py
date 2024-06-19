import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

# 모델 로드
model_path = r'C:/Users/USER/source/repos/Project2/Project2/model/animal_model.h5'
model = tf.keras.models.load_model(model_path)
print("모델이 성공적으로 로드되었습니다.")

# 중간 레이어 모델 생성
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)


# 특징 벡터 추출 함수
def extract_features(img_array, model):
    features = model.predict(img_array, batch_size=1)
    return features.flatten()


# 전처리된 이미지 시각화 함수
def visualize_image(img_array, title):
    img = np.squeeze(img_array, axis=0)  # 배치 차원 제거
    img = img - np.min(img)  # 최소값을 0으로 이동
    img = img / np.max(img)  # 최대값을 1로 이동
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()


# captured_face.png 이미지 경로
captured_face_path = r'C:/Users/USER/source/repos/Project2/Project2/captured_face.png'
captured_face_img = cv2.imread(captured_face_path)
if captured_face_img is None:
    raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {captured_face_path}")
captured_face_img = cv2.resize(captured_face_img, (224, 224))  # 모델의 입력 크기에 맞게 조정

# 이미지 전처리 확인
print(f"Original image shape: {captured_face_img.shape}")
captured_face_array = img_to_array(captured_face_img)
captured_face_array = np.expand_dims(captured_face_array, axis=0)
captured_face_array = preprocess_input(captured_face_array)  # 전처리 방식 적용
print(f"Processed image shape: {captured_face_array.shape}")

# 전처리된 이미지 시각화
visualize_image(captured_face_array, "Preprocessed Captured Face")

# 중간 레이어 출력 확인
intermediate_output = intermediate_layer_model.predict(captured_face_array)
print(f"Intermediate output shape: {intermediate_output.shape}")
print(f"Intermediate output: {intermediate_output}")

# captured_face.png의 특징 벡터 추출
captured_face_features = extract_features(captured_face_array, model)
print(f"Captured face features: {captured_face_features[:10]}")  # 특징 벡터의 일부를 출력하여 확인

# 특징 벡터 시각화
plt.plot(captured_face_features)
plt.title("Captured Face Features")
plt.show()

# 동물 이미지 경로 리스트
animal_images_folder = r'C:/Users/USER/Pictures/archive (1)/animals/val'  # 수정된 로컬 경로
animal_images_paths = []

for folder_name in os.listdir(animal_images_folder):
    folder_path = os.path.join(animal_images_folder, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpeg') or file_name.endswith('.jpg') or file_name.endswith('.png'):
                animal_images_paths.append(os.path.join(folder_path, file_name))

# 동물 이미지 파일 수 확인
print(f"동물 이미지 파일 수: {len(animal_images_paths)}")

# 모든 동물 이미지의 특징 벡터 추출 및 비교
min_dist = float('inf')
most_similar_img_path = None

# 일정 수의 이미지만 확인하도록 제한 (예: 10개)
max_images_to_check = 10

for i, animal_img_path in enumerate(animal_images_paths):
    if i >= max_images_to_check:
        break

    print(f"Processing {animal_img_path}")
    animal_img = cv2.imread(animal_img_path)
    if animal_img is None:
        print(f"이미지를 불러올 수 없습니다: {animal_img_path}")
        continue
    animal_img = cv2.resize(animal_img, (224, 224))  # 모델의 입력 크기에 맞게 조정

    # 동물 이미지 전처리 확인
    animal_img_array = img_to_array(animal_img)
    animal_img_array = np.expand_dims(animal_img_array, axis=0)
    animal_img_array = preprocess_input(animal_img_array)

    animal_features = extract_features(animal_img_array, model)
    print(f"Animal features: {animal_features[:10]}")  # 각 동물 이미지의 특징 벡터 일부를 출력하여 확인

    dist = np.linalg.norm(captured_face_features - animal_features)
    print(f"Distance to {animal_img_path}: {dist}")  # 각 동물 이미지와의 거리를 출력하여 확인

    if dist < min_dist:
        min_dist = dist
        most_similar_img_path = animal_img_path

if most_similar_img_path:
    print(f"Most similar image: {most_similar_img_path}")
    # 예측된 동물 이미지 불러오기
    predicted_animal_img = cv2.imread(most_similar_img_path)
    predicted_animal_img = cv2.resize(predicted_animal_img, (224, 224))  # 동일한 크기로 조정

    # 두 이미지를 가로로 붙이기
    captured_face_img_resized = cv2.resize(cv2.imread(captured_face_path), (224, 224))
    combined_img = np.hstack((captured_face_img_resized, predicted_animal_img))

    # 두 이미지를 나란히 보여주기
    cv2.imshow('Captured Face and Most Similar Animal', combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("유사한 동물 이미지를 찾을 수 없습니다.")

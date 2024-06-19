import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# 데이터셋 경로 설정
train_dataset_path = 'C:/Users/USER/Pictures/archive (1)/animals/train'

# 데이터셋 생성
batch_size = 32
img_height = 224
img_width = 224

raw_train_ds = image_dataset_from_directory(
    train_dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

raw_val_ds = image_dataset_from_directory(
    train_dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# 클래스 이름 추출
class_names = raw_train_ds.class_names
print("Class names:", class_names)

# 데이터셋 캐시 및 프리페치 설정
AUTOTUNE = tf.data.AUTOTUNE

train_ds = raw_train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = raw_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 사전 학습된 MobileNetV2 모델 로드 (탑 없이)
base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                         include_top=False,
                         weights='imagenet')

# 분류층 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)  # 클래스 수 자동 설정

model = Model(inputs=base_model.input, outputs=predictions)

# 사전 학습된 모델의 모든 층을 동결
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 모델 학습
epochs = 10

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 모델 평가
loss, accuracy = model.evaluate(val_ds)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# 모델 저장
model_save_path = 'C:/Users/USER/Pictures/archive (1)/animals/model/animal_model.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")


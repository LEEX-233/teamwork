import os
import numpy as np
import logging
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置日志记录器
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# 加载单个图像的函数（已修改以包含错误处理）
def load_image(img_id, rawdata_dir):
    img_path = os.path.join(rawdata_dir, f'{img_id}')
    try:
        with open(img_path, 'rb') as img_file:
            img_data = np.frombuffer(img_file.read(), dtype=np.uint8)
        if img_data.size == 16384:
            img_matrix = img_data.reshape((128, 128))
            return img_matrix
        else:
            logging.error(f"Expected 16384 bytes for image {img_id}, but got {img_data.size}. Using a placeholder image.")
            placeholder_image = np.zeros((128, 128), dtype=np.uint8)
            return placeholder_image
    except Exception as e:
        logging.error(f"Failed to load image {img_id} from {img_path}: {e}")
        placeholder_image = np.zeros((128, 128), dtype=np.uint8)
        return placeholder_image

# 读取数据的函数
def load_data(filepath, rawdata_dir):
    data = []
    labels = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            img_id = int(parts[0])
            gender = parts[1]
            age = parts[2]
            image = load_image(img_id, rawdata_dir)
            data.append(image.flatten())  # 展平为向量
            labels.append((gender, age))
    return np.array(data), labels

# 示例：加载数据
rawdata_dir = 'rawdata'  # 替换为您的原始数据目录路径
labels_filepath = 'cleared_faceDS.txt'  # 替换为您的标签文件路径
data, labels = load_data(labels_filepath, rawdata_dir)

# 数据归一化
def normalize_data(data):
    return (data / 255.0).astype(np.float32)

# 加载训练和测试数据
train_data, train_labels = load_data('cleared_faceDR.txt', 'rawdata')
test_data, test_labels = load_data('cleared_faceDS.txt', 'rawdata')

# 在加载数据后进行归一化
train_data = normalize_data(train_data)
test_data = normalize_data(test_data)

# 标签编码
gender_encoder = LabelEncoder()
age_encoder = LabelEncoder()
train_gender_labels = gender_encoder.fit_transform([label[0] for label in train_labels])
train_age_labels = age_encoder.fit_transform([label[1] for label in train_labels])
test_gender_labels = gender_encoder.transform([label[0] for label in test_labels])
test_age_labels = age_encoder.transform([label[1] for label in test_labels])

# 合并所有训练数据
X_train_all = np.array(train_data)

# PCA降维
n_components = 150
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_all)

# 划分训练集和测试集
X_train, X_test, y_train_gender, y_test_gender = train_test_split(X_train_pca, train_gender_labels, test_size=0.2, random_state=42)
X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(X_train_pca, train_age_labels, test_size=0.2, random_state=42)

# 逻辑回归模型：性别分类
clf_gender = LogisticRegression(max_iter=1000)
clf_gender.fit(X_train, y_train_gender)  # 训练模型
y_pred_gender = clf_gender.predict(X_test)  # 预测
print("Gender Classification Report:")
print(classification_report(y_test_gender, y_pred_gender, target_names=gender_encoder.classes_))

# 逻辑回归模型：年龄分类
clf_age = LogisticRegression(max_iter=1000)
clf_age.fit(X_train_age, y_train_age)  # 训练模型
y_pred_age = clf_age.predict(X_test_age)  # 预测
print("Age Classification Report:")
print(classification_report(y_test_age, y_pred_age, target_names=age_encoder.classes_))

# 性别分类混淆矩阵
conf_matrix_gender = confusion_matrix(y_test_gender, y_pred_gender)
plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix_gender, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Gender Classification Confusion Matrix')
plt.colorbar()
plt.xticks(range(len(gender_encoder.classes_)), gender_encoder.classes_, rotation=45)
plt.yticks(range(len(gender_encoder.classes_)), gender_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 年龄分类混淆矩阵
conf_matrix_age = confusion_matrix(y_test_age, y_pred_age)
plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix_age, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Age Classification Confusion Matrix')
plt.colorbar()
plt.xticks(range(len(age_encoder.classes_)), age_encoder.classes_, rotation=45)
plt.yticks(range(len(age_encoder.classes_)), age_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 测试数据PCA转换
X_test_pca = pca.transform(np.array(test_data))

# 测试性别分类
y_test_gender_pred = clf_gender.predict(X_test_pca)
print("Test Gender Classification Report:")
print(classification_report(test_gender_labels, y_test_gender_pred, target_names=gender_encoder.classes_))

# 测试年龄分类
y_test_age_pred = clf_age.predict(X_test_pca)
print("Test Age Classification Report:")
print(classification_report(test_age_labels, y_test_age_pred, target_names=age_encoder.classes_))

# 性别分类混淆矩阵（测试数据）
conf_matrix_gender_test = confusion_matrix(test_gender_labels, y_test_gender_pred)
plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix_gender_test, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Test Gender Classification Confusion Matrix')
plt.colorbar()
plt.xticks(range(len(gender_encoder.classes_)), gender_encoder.classes_, rotation=45)
plt.yticks(range(len(gender_encoder.classes_)), gender_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 年龄分类混淆矩阵（测试数据）
conf_matrix_age_test = confusion_matrix(test_age_labels, y_test_age_pred)
plt.figure(figsize=(10, 7))
plt.imshow(conf_matrix_age_test, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Test Age Classification Confusion Matrix')
plt.colorbar()
plt.xticks(range(len(age_encoder.classes_)), age_encoder.classes_, rotation=45)
plt.yticks(range(len(age_encoder.classes_)), age_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
# 1. 讀入三個 CSV
df1 = pd.read_csv('hand0.csv')   # label = 0
df2 = pd.read_csv('hand1.csv')   # label = 1
df3 = pd.read_csv('hand2.csv')   # label = 2

# 2. 合併資料
df = pd.concat([df1, df2, df3], ignore_index=True)

# 3. 丟掉不需要欄位
if 'frame' in df.columns:
    df = df.drop(columns=['frame'])

# 4. 分離特徵與標籤
X = df.drop(columns=['label'])
y = df['label']

# 5. 切分訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 訓練模型
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# 7. 評估
y_pred = model.predict(X_test)
print("✅ 準確率 Accuracy:", accuracy_score(y_test, y_pred))
print("📊 分類報告:\n", classification_report(y_test, y_pred))

# ✅ 8. 儲存模型
joblib.dump(model, 'svm_hand_model.pkl')
print("✅ 模型已儲存為 svm_hand_model.pkl")

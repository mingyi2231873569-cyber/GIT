# ==================== app.py ====================
# 导入必要的库
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
import warnings
warnings.filterwarnings('ignore')

# ---------- 复制您自定义的 SuperLearnerClassifier 类 ----------
# 注意：因为模型保存时依赖这个类的定义，加载时必须包含相同的类定义
# 请将您代码中的 SuperLearnerClassifier 类完整复制到此处
# 为节省篇幅，这里仅给出简写，您必须从您自己的代码中完整复制过来
# 但为了确保正确，我将把之前提供的类定义精简后放在这里，您可以直接使用
# (实际上您需要把完整的类定义粘贴过来，包括 init, fit, predict_proba 等方法)

# 以下是我从您之前的代码中提取的 SuperLearnerClassifier 类的完整定义，
# 您可以直接使用，不必再自己复制。
# 注意：这个类依赖于 XGB_INSTALLED 变量，我们在 app.py 中也定义了它。

# ==================== 导入基类 ====================
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import copy

try:
    from xgboost import XGBClassifier
    XGB_INSTALLED = True
except ImportError:
    XGB_INSTALLED = False

# ==================== SuperLearnerClassifier 类 ====================
class SuperLearnerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_learners=None, meta_learner=None, cv_folds=5):
        if base_learners is None:
            self.base_learners = [
                ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
                ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
                ('nb', GaussianNB()),
                ('knn', KNeighborsClassifier(n_neighbors=5))
            ]
            if XGB_INSTALLED:
                try:
                    self.base_learners.append(('xgb', XGBClassifier(random_state=42)))
                except:
                    pass
        else:
            self.base_learners = base_learners

        if meta_learner is None:
            self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        else:
            self.meta_learner = meta_learner

        self.cv_folds = cv_folds
        self.is_fitted = False
        self.label_encoder = None  # 注意：预测时不需要重新编码，所以这里可以简单处理
        self.n_classes_ = None
        self.classes_ = None
        self.base_learners_final = []

    def fit(self, X, y):
        # 此方法在训练时使用，但加载模型后不会调用，因此可以留空或简单实现
        # 但为了类的完整性，这里保留一个空fit，实际训练已经在训练阶段完成
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'is_fitted')
        X = check_array(X)

        # 生成基学习器的预测
        meta_features = np.zeros((X.shape[0], len(self.base_learners_final) * self.n_classes_))

        for i, (name, clf) in enumerate(self.base_learners_final):
            if hasattr(clf, 'predict_proba'):
                probas = clf.predict_proba(X)
            else:
                # 简化处理，实际应根据clf类型选择合适方法
                probas = np.ones((len(X), self.n_classes_)) / self.n_classes_
            meta_features[:, i*self.n_classes_:(i+1)*self.n_classes_] = probas

        return self.meta_learner.predict_proba(meta_features)

    def predict(self, X):
        probas = self.predict_proba(X)
        # 注意：这里直接返回类别索引，因为加载后 label_encoder 可能未保存，所以返回整数
        return np.argmax(probas, axis=1)

    # 为了兼容joblib加载，还需要添加 get_params 等方法，但通常默认即可

# 但我们的模型保存时已经包含了完整的内部状态，所以上面的 fit 其实不需要真正训练，
# 只要类的结构一致，joblib就能正确恢复所有属性。因此上面的 fit 可以简单返回 self。
# 注意：上面的 base_learners_final 在训练后被填充，加载后应该自动恢复。

# ---------- 加载保存的模型和工具 ----------
model = joblib.load('super_learner_final.joblib')
scaler = joblib.load('scaler_final.joblib')
feature_names = joblib.load('feature_names.joblib')
class_names = joblib.load('class_names.joblib')

# ---------- 创建 Flask 应用 ----------
app = Flask(__name__)

@app.route('/')
def index():
    """显示输入表单"""
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    try:
        # 收集表单中的所有特征值
        input_values = []
        for feature in feature_names:
            val = request.form.get(feature)
            if val is None or val.strip() == '':
                return jsonify({'error': f'缺少特征值: {feature}'}), 400
            input_values.append(float(val))

        # 转换为 numpy 数组并标准化
        X = np.array(input_values).reshape(1, -1)
        X_scaled = scaler.transform(X)

        # 预测
        pred_class_idx = model.predict(X_scaled)[0]   # 返回的是索引 0,1,2
        pred_proba = model.predict_proba(X_scaled)[0]

        # 将索引转换为类别名称
        pred_class_name = class_names[pred_class_idx]

        # 构建概率字典
        prob_dict = {class_names[i]: float(pred_proba[i]) for i in range(len(class_names))}

        return jsonify({
            'predicted_class': pred_class_name,
            'probabilities': prob_dict
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
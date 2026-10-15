# Train an Iris flower model and save it as model.joblib
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

MODEL_VERSION = "1.0"

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model {MODEL_VERSION} accuracy: {accuracy:.2f}")

joblib.dump({"model": model, "version": MODEL_VERSION,
             "accuracy": round(accuracy, 2)}, "model.joblib")

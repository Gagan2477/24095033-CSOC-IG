
input_size = X_train.shape[1]
hidden_size = 16  # You can tune this
model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size)
model.train(X_train, y_train, epochs=1000, lr=0.01)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve, auc

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

precision, recall, _ = precision_recall_curve(y_test, y_pred)
pr_auc = auc(recall, precision)
print("PR-AUC:", pr_auc)

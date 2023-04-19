import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# load data from CSV files
# y_true = pd.read_csv('C:\Users\Gustavo Valentim\OneDrive\Área de Trabalho\python\model_files\test.csv')[1, 2, 3, 4, 5, 6, 7, 8]
y_true = pd.read_csv(
    'C:\\Users\\Gustavo Valentim\\OneDrive\\Área de Trabalho\\python\\model_files\\test_valentao.csv')

y_pred = pd.read_csv(
    'C:\\Users\\Gustavo Valentim\\OneDrive\\Área de Trabalho\\python\\model_files\\predictions.csv')
#cm = confusion_matrix(y_true, y_pred)
cm = [[420,   0,   2,   2,   2,   0,   0,   0,   0],
      [0,  92,   2,   0,   1,   0,   0,   5,   0],
      [0,   1, 300,   0,   2,   5,   2,   5,   0],
      [0,   0,   5, 124,   2,   3,   0,   4,   0],
      [0,   2,   2,   0, 307,   4,   0,   1,   0],
      [0,   0,   6,   1,   0, 323,   0,   5,   1],
      [0,   0,   3,   0,   2,   0, 202,   2,   2],
      [0,   2,   2,   3,   0,   1,   0, 305,   2],
      [0,   0,   0,   0,   0,   1,   0,   1, 214]]
# print(cm)
labels = ['caterpilar', 'bacterial_blight', 'target_spot', 'downey_mildew',
          'frogeye', 'cercospora_leaf_blight', 'potassium_deficiency', 'healthy', 'soybean_rust']

# Create a pandas DataFrame from the confusion matrix and labels
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Create the heatmap using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')

# Add labels and title
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Show the plot
plt.show()

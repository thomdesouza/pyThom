import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import fetch_rrlyrae_combined
from sklearn.linear_model import RidgeClassifierCV
from sklearn import metrics

#Classifier Parametrization
print('')
classificador = RidgeClassifierCV()
print('Classification Algorithm: Ridge CV')

# as páginas citadas referem-se ao livro AstroML (Ivezic et al., 2014)
#----------------------------------------------------------------------
#X = np.random.random((100, 2)) # 100 pts in 2 dims
#y = (X[:, 0] + X[:, 1] > 1).astype(int)

# This example downloads and plots the colors of RR Lyrae stars
#  along with those of the non-variable stars. 
#  Several of the classification examples in the book figures use this dataset.
#  http://www.astroml.org/examples/datasets/plot_rrlyrae_mags.html

# get data and split into training & testing sets
X, y = fetch_rrlyrae_combined() #pág.365
X = X[-5000:]
y = y[-5000:]

stars = (y == 0)
rrlyrae = (y == 1)
#------------------------------------------------------------
# plot the results
ax = plt.axes()

ax.plot(X[stars, 0], X[stars, 1], '.', ms=5, c='b', label='stars')
ax.plot(X[rrlyrae, 0], X[rrlyrae, 1], '.', ms=5, c='r', label='RR-Lyrae')

ax.legend(loc=3)
ax.set_xlabel('$u-g$')
ax.set_ylabel('$g-r$')
ax.set_xlim(0.7, 1.4)
ax.set_ylim(-0.2, 0.4)
plt.show()
#------------------------------------------------------------
# Classification
classificador.fit(X, y)
y_pred = classificador.predict(X)

matches = (y_pred == y)
print('Total Number of Instances:', len(matches))
print('Correctly Classified Instances:', matches.sum())
print('Incorrectly Classified Instances:', matches.sum())
print('')
print('Classification Accuracy:', 100 * metrics.accuracy_score(y, y_pred), '%')
print('Error Rate:', 100 - 100 * metrics.accuracy_score(y, y_pred), '%')
print('')
print('Confusion Matrix:')
print(metrics.confusion_matrix(y, y_pred))
print('')
print('Resumed Report:')
target_names = ['class 0    (stars)', 'class 1 (RR-Lyrae)']
print(metrics.classification_report(y, y_pred, target_names=target_names))
print('F1-Score:', metrics.f1_score(y, y_pred))
# Plot ROC curve
fig = plt.figure(figsize=(5, 2.5))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.25)
# roc curves
ax1 = plt.subplot(121)
ax1.plot(fpr, tpr)
ax1.set_xlim(0, 0.04)
ax1.set_ylim(0, 1.02)
ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
ax1.set_xlabel('false positive rate')
ax1.set_ylabel('true positive rate')
ax1.legend(loc=4)
plt.title('ROC Curve')
plt.show()

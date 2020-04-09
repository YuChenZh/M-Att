from matplotlib import pyplot as plt
import numpy as np
x = ['20%', '40%', '60%', '80%', '100%']
# plt.plot(x,[0.71,0.726,0.691,0.826,0.79],label ='S-LSTM')
# plt.plot(x,[0.68,0.726,0.691,0.716,0.74],label ='M-MLP')
# plt.plot(x, [0.737,0.65,0.754,0.611,0.78], label='M-LSTM')
# plt.plot(x, [0.735,0.70,0.808,0.773,0.81], label='M-CNN/LSTM')
# plt.plot(x, [0.764,0.749,0.769,0.831,0.864], label='M_Att')

# plt.xlabel('Percentage of training data')
# plt.ylabel('Balanced accuracy')

# plt.title("Balanced Accuracy comparison(6 users)",pad=20)
# plt.grid(linestyle='--',color='grey')
# plt.legend()
# plt.savefig('chart_BA_trend.eps',format='eps',dpi=1000)
# plt.close()
# plt.show()

plt.plot(x,[0.71,0.726,0.691,0.826,0.79],label ='S-LSTM')
plt.plot(x,[0.6,0.741,0.767,0.815,0.74],label ='M-MLP')
plt.plot(x, [0.797,0.701,0.758,0.661,0.78], label='M-LSTM')
plt.plot(x, [0.68,0.7443,0.787,0.741,0.81], label='M-CNN/LSTM')
plt.plot(x, [0.776,0.786,0.815,0.822,0.86], label='M-Att')

plt.xlabel('Percentage of training data')
plt.ylabel('Balanced accuracy')
plt.title("Balanced Accuracy comparison",pad=20)
plt.grid(linestyle='--',color='grey')

plt.legend()
plt.savefig('chart_BA_trend.eps',format='eps',dpi=1000)
plt.close()


# plt.plot(x,[0.391,0.476,0.403,0.66,0.578],label ='Single-task')
# plt.plot(x, [0.551,0.388,0.563,0.552,0.712], label='Shared-MTL')
# plt.plot(x, [0.719,0.602,0.722,0.765,0.845], label='SHybrid-MTL')
# plt.plot(x, [0.624,0.637,0.696,0.746,0.791], label='Attention-MTL')
# plt.xlabel('Percentage of training data')
# plt.ylabel('Sensitivity')
# plt.title("True Positive Rate comparison(6 users)",pad=20)
# plt.grid(linestyle='--',color='grey')

# plt.legend()
# plt.savefig('chart_TPR_trend_6tasks.eps',format='eps',dpi=1000)
# plt.close()

plt.plot(x, [0.28,0.547,0.586,0.666,0.61],label ='S-LSTM')
plt.plot(x, [0.24,0.507,0.556,0.646,0.52],label ='M-MLP')
plt.plot(x, [0.683,0.513,0.594,0.552,0.65], label='M-LSTM')
plt.plot(x, [0.618,0.602,0.753,0.725,0.73], label='M-CNN/LSTM')
plt.plot(x, [0.669,0.581,0.72,0.736,0.74], label='M_Att')
plt.xlabel('Percentage of training data')
plt.ylabel('Recall')
plt.title("Recall score comparison",pad=20)
plt.grid(linestyle='--',color='grey')

plt.legend()
plt.savefig('chart_Recall_trend.eps',format='eps',dpi=1000)
plt.close()


plt.plot(x, [0.43,0.507,0.686,0.656,0.72],label ='S-LSTM')
plt.plot(x, [0.34,0.507,0.46,0.56,0.55],label ='M-MLP')
plt.plot(x, [0.23,0.33,0.34,0.42,0.43], label='M-LSTM')
plt.plot(x, [0.22,0.38,0.33,0.37,0.45], label='M-CNN/LSTM')
plt.plot(x, [0.589,0.621,0.70,0.716,0.72], label='M_Att')
plt.xlabel('Percentage of training data')
plt.ylabel('Precision')
plt.title("Precision score comparison",pad=20)
plt.grid(linestyle='--',color='grey')

plt.legend()
plt.savefig('chart_Precision_trend.eps',format='eps',dpi=1000)
plt.close()


plt.plot(x, [0.34,0.52,0.63,0.656,0.66],label ='S-LSTM')
plt.plot(x, [0.28,0.43,0.49,0.56,0.52],label ='M-MLP')
plt.plot(x, [0.34,0.426,0.432,0.47,0.517], label='M-LSTM')
plt.plot(x, [0.32,0.46,0.458,0.49,0.556], label='M-CNN/LSTM')
plt.plot(x, [0.625,0.60,0.71,0.706,0.72], label='M_Att')
plt.xlabel('Percentage of training data')
plt.ylabel('F1 score')
plt.title("F1 score comparison",pad=20)
plt.grid(linestyle='--',color='grey')

plt.legend()
plt.savefig('chart_F1_trend.eps',format='eps',dpi=1000)
plt.close()
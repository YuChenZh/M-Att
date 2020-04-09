
import pygal

# line_chart = pygal.Line()
# line_chart.title = 'Validation Performance comparison (in %)'
# line_chart.x_labels = map(str, range(10, 60,10))
# line_chart.add('LR-BA', [0.84, 0.81,    0.78, 0.76,   0.75])
# line_chart.add('MLP(2)-BA',  [0.9, 0.88, 0.88, 0.85, 0.85])
# line_chart.add('LR-TPR', [0.7, 0.63,    0.58, 0.54,   0.52])
# line_chart.add('MLP(2)-TPR',  [0.81, 0.78, 0.76, 0.71, 0.71])
#
# line_chart.render()
#
# line_chart.render_to_file('chart.svg')



line_chart2 = pygal.Line()
line_chart2.title = 'Balance Accuracy_Prediction Performance comparison (in %)'
line_chart2.x_labels = map(str, range(1, 11,1))
line_chart2.add('LR', [0.61,0.62,0.66,0.71,0.72,0.66,0.65,0.73,0.76,0.75])
line_chart2.add('MLP(2)',  [0.62,0.62,0.64,0.68,0.7,0.67,0.7,0.71,0.74,0.73])
line_chart2.add('LSTM',  [0.74,0.76,0.77,0.81,0.83,0.85,0.84,0.85,0.84,0.87])

line_chart2.render()

line_chart2.render_to_file('chart_BA.svg')

line_chart2 = pygal.Line()
line_chart2.title = 'True Positive Rate_Prediction Performance comparison (in %)'
line_chart2.x_labels = map(str, range(1, 11,1))

line_chart2.add('LR-TPR', [0.27,0.28,0.4,0.45,0.49,0.36,0.37,0.52,0.58,0.55])
line_chart2.add('MLP(2)-TPR',  [0.27,0.26,0.31,0.39,0.44,0.38,0.43,0.45,0.51,0.51])
line_chart2.add('LSTM-TPR',  [0.49,0.52,0.55,0.63,0.66,0.7,0.69,0.7,0.69,0.74])

line_chart2.render()

line_chart2.render_to_file('chart_TPR.svg')
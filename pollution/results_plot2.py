
import pygal



line_chart = pygal.Bar()
line_chart.title = 'SMAPE comparison between single-task model and multi-task model'
line_chart.x_labels = 'task1', 'task2', 'task3', 'task4', 'task5','task6','task7','task8','task9'

line_chart.add('Single-task', [0.608, 0.594, 0.356, 0.616,0.437,0.427,0.602,0.602, 0.465])
line_chart.add('Multi-task',  [0.614, 0.564, 0.420, 0.625, 0.423, 0.441,0.612,0.532,0.455])
line_chart.render()


line_chart.render_to_file('single_multi_comparison.svg')
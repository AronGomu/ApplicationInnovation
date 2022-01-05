import matplotlib.pyplot as plt 
import json

with open('logs.json', 'r') as f:
	j = json.load(f)
	for rnn_data in j:
		data_x = []
		data_y = rnn_data[4]
		for i in range(1, len(data_y)+1):
			data_x.append(i * 150)
			data_y[i-1] = float(data_y[i-1])
		print(data_x)
		print(data_y)
		plt.figure()
		plt.plot(data_x, data_y)
		plt.show()
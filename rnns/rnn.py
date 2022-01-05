import torch
import torch.nn as nn

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size, device):
		super(RNN, self).__init__()

		self.type = 'rnn'

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.bidirectional = True
		self.num_directions = 1
		if self.bidirectional:
			self.num_directions = 2

		self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True)
		self.out = nn.Linear(self.num_directions * self.hidden_size, output_size)
		
		self.dropout = nn.Dropout(0.1)
		self.softmax = nn.LogSoftmax(dim=1)

		self.device = device
		
	def forward(self, input, hidden):
		_, hidden = self.rnn(input.unsqueeze(0), hidden)

		hidden_concatenated = hidden

		if self.bidirectional:
			hidden_concatenated = torch.cat((hidden[0], hidden[1]), 1)
		else:
			hidden_concatenated = hidden.squeeze(0)

		output = self.out(hidden_concatenated)

		output = self.dropout(output)
		output = self.softmax(output)

		return output, hidden

	def init_hidden(self):
		return torch.zeros(self.num_directions, 1, self.hidden_size)
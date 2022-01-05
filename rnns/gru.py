import torch
import torch.nn as nn

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size, device):
		super(RNN, self).__init__()

		self.type = 'gru'

		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.embed = nn.Embedding(input_size, hidden_size)
		self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, output_size)

		self.device = device
		
	def forward(self, x, hidden):
		out = self.embed(x)
		out = self.rnn(out.unsqueeze(1), hidden)
		out = self.fc(out.reshape(out.shape[0], -1))
		return out, hidden
	
	def init_hidden(self, batch_size):
		hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
		return hidden
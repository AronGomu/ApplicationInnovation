import torch
import torch.nn as nn
import string
import random
import sys
import unidecode
import json
import time

import rnns.rnn as rnn
import rnns.lstm as lstm
import rnns.gru as gru

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get characters from string.printable


# all printable characters
all_characters = string.printable
n_characters = len(all_characters)


print(all_characters)

class Generator():
	def __init__(self):
		self.chunk_len = 250
		self.num_epochs = 10000
		self.batch_size = 1
		self.print_every = 50
		self.hidden_size = 128
		self.num_layers = 2
		self.lr = 0.003

	def char_tensor(self, string):
		tensor = torch.zeros(len(string)).long()
		for c in range(len(string)):
			tensor[c] = all_characters.index(string[c])
		return tensor

	def get_random_batch(self):
		start_idx = random.randint(0, len(file) - self.chunk_len)
		end_idx = start_idx + self.chunk_len + 1
		text_str = file[start_idx:end_idx]
		text_input = torch.zeros(self.batch_size, self.chunk_len)
		text_target = torch.zeros(self.batch_size, self.chunk_len)

		for i in range(self.batch_size):
			text_input[i,:] = self.char_tensor(text_str[:-1])
			text_target[i,:] = self.char_tensor(text_str[1:])
		
		return text_input.long(), text_target.long()

	def generate(self, initial_str='a', predict_len=100, temperature=0.85):
		hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
		initial_input = self.char_tensor(initial_str)
		predicted = initial_str

		for p in range(len(initial_str) -1):
			_, (hidden, cell) = self.rnn(initial_input[p].view(1).to(device), hidden, cell)
		
		last_char = initial_input[-1]

		for p in range(predict_len):
			output, (hidden, cell) = self.rnn(last_char.view(1).to(device), hidden, cell)
			output_dist = output.data.view(-1).div(temperature).exp()
			top_char = torch.multinomial(output_dist, 1)[0]
			predicted_char = all_characters[top_char]
			predicted += predicted_char
			last_char = self.char_tensor(predicted_char)

		return predicted

	def generateOneWord(self, initial_str='a', predict_len=30, temperature=0.85):
		words = self.generate(initial_str, predict_len, temperature) # generate characters string
		words = words.split('\n') # split into list of words
		return words[1] # return only the second one

	def train(self, model_start_file_name, file_test_path):
		optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
		criterion = nn.CrossEntropyLoss()
		#writer = SummaryWriter(f'runs/{run_filename}')

		# file test setup
		all_data = []
		model_file_name = f"{model_start_file_name}_{self.chunk_len}_{self.num_epochs}_{self.batch_size}_{self.hidden_size}_{self.num_layers}_{self.lr}.pt"
		test_words = file_test_path.split('\n')
		test_words.pop()
		number_words_generated_total = 0
		number_words_found_total = 0

		print('=> Starting training')

		for epoch in range(1, self.num_epochs + 1):
			inp, target = self.get_random_batch()
			if self.rnn.type == 'lstm':
				hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
			elif self.rnn.type == 'rnn':
				hidden = self.rnn.init_hidden()

			self.rnn.zero_grad()
			loss = 0
			inp = inp.to(device)
			target = target.to(device)

			for c in range(self.chunk_len):
				output, (hidden, cell) = self.rnn(inp[:, c], hidden, cell)
				loss += criterion(output, target[:, c])

			loss.backward()
			optimizer.step()
			loss = loss.item()/ self.chunk_len

			if epoch % self.print_every == 0:

				torch.save(self.rnn, f"models/{model_file_name}")
				

				print(f'Epoch {epoch} - Loss: {loss}')
				generated_words = self.generate()
				generated_words = generated_words.split('\n')
				print(generated_words)
				# delete the first and last word because they are biased
				generated_words.pop(0)
				generated_words.pop()

				number_words_generated = len(generated_words)
				number_words_found = 0

				number_words_generated_total += len(generated_words)

				for word in generated_words:
					if word in test_words: 
						number_words_found += 1
						number_words_found_total += 1

				all_data.append((epoch, loss, f'{number_words_found * 100 / number_words_generated}%', generated_words))


				with open(f'logs/{model_file_name}.json', 'w') as f:
					f.write(json.dumps(all_data))


			#writer.add_scalar('Training loss', loss, global_step=epoch)

	def evaluate_epoch_train(self, model_start_file_name, file_test_path, print_every=100):
		optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
		criterion = nn.CrossEntropyLoss()

		# file test setup
		model_file_name = f"{model_start_file_name}_{self.chunk_len}_{self.num_epochs}_{self.batch_size}_{self.hidden_size}_{self.num_layers}_{self.lr}"
		
		print('=> Starting epoch evaluating training')

		for epoch in range(1, self.num_epochs + 1):
			# input_line_tensor & target_line_tensor
			inp, target = self.get_random_batch()
			if self.rnn.type == 'lstm':
				hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
			elif self.rnn.type == 'rnn':
				inp.unsqueeze_(-1)
				hidden = self.rnn.init_hidden()

			self.rnn.zero_grad()
			loss = 0

			inp = inp.to(device)
			target = target.to(device)

			for c in range(self.chunk_len):
				if self.rnn.type == 'lstm':
					output, (hidden, cell) = self.rnn(inp[:, c], hidden, cell)
				elif self.rnn.type == 'rnn':
					print(inp)
					print(hidden)
					output, hidden = self.rnn(inp[:, c], hidden)

				loss += criterion(output, target[:, c])

			loss.backward()
			optimizer.step()
			loss = loss.item() / self.chunk_len

			if epoch % print_every == 0:

				print(f'Epoch {epoch} - Loss: {loss}')

				model_file_name = f"{model_start_file_name}_{self.chunk_len}_{epoch}_{self.batch_size}_{self.hidden_size}_{self.num_layers}_{self.lr}"
				torch.save(self.rnn, f"models/{model_file_name}.pt")

				print('Starting testing batch...')
				self.testing_batch(f"{model_file_name}", file_test_path, 300)
				print('End testing batch')
				
	def testing(self, test_name, file_test_path, number_samples=0, percent=15, number_starting_characters=2, predict_len=30, temperature=0.85, print_scale=50):
		print('=> Starting testing')

		# setup file_test_path
		file_test_path = unidecode.unidecode(open(file_test_path).read())
		test_words = file_test_path.split('\n')
		test_words.pop()

		# variable initialisation
		accuracy = 0
		predicted = "a"
		all_accurate_words = []
		all_predicted_words = []
		
		if number_samples > 0: # Si il y a un nombre précisé de test

			for i in range(1, number_samples + 1):

				# Si la prédiction a déjà été faite, on relance
				while predicted in all_predicted_words:
					starting_letters = "" # on reset la lettre de départ
					for n in range(number_starting_characters): # pour chaque caractère qu'on donne pour la prédiction
						random_character = random.randint(0, len(string.ascii_uppercase) - 1) # on genère un caractère aléatoire
						starting_letters = starting_letters + string.ascii_uppercase[random_character] # on ajoute le caractère généré aléatoirement aux lettres de départs

					predicted = self.generateOneWord(initial_str=starting_letters, predict_len=predict_len, temperature=temperature).lower() # on genère une prédiction puis on vérifie qu'elle n'a pas déjà était faite



				all_predicted_words.append(predicted)

				if predicted in test_words:
					all_accurate_words.append(predicted)

				accuracy = 100 * len(all_accurate_words) / len(all_predicted_words)

				if i % print_scale == 0:

					with open(f'logs/{test_name}.json', 'w') as f:
						f.write(json.dumps((i, f'{accuracy}%', all_accurate_words, all_predicted_words)))

					print(f'{i}\t\t: {accuracy}% Accuracy')

		else:
			i = 0
			percentage_words_found = 0 / len(test_words)
			while percentage_words_found < percent:
				nc = random.randint(1, int(predict_len / 2 - 1))

				while predicted in all_predicted_words:
					starting_letters = ""
					for n in range(nc):
						rc = random.randint(0, len(string.ascii_uppercase) - 1)
						starting_letters = starting_letters + string.ascii_uppercase[rc]

					predicted = self.generateOneWord(initial_str=starting_letters, predict_len=predict_len, temperature=temperature).lower() # on genère une prédiction puis on vérifie qu'elle n'a pas déjà était faite

				all_predicted_words.append(predicted)

				if predicted in test_words:
					all_accurate_words.append(predicted)

				accuracy = 100 * len(all_accurate_words) / len(all_predicted_words)
				percentage_words_found = 100 * len(all_accurate_words) / len(test_words)

				if i % print_scale == 0:

					with open(f'logs/{test_name}.json', 'w') as f:
						f.write(json.dumps((i, f'{percentage_words_found}%', f'{accuracy}%', len(all_accurate_words), all_accurate_words, len(all_predicted_words), all_predicted_words)))

					print(f'{i}\t\t: {percentage_words_found}% words found with {accuracy}% Accuracy')

	def testing_batch(self, test_name, file_test_path, number_samples, percent=15, number_starting_characters=2, predict_len=100, temperature=0.85, print_scale=50):
		print('=> Starting testing batch')

		# setup file_test_path
		file_test_path = unidecode.unidecode(open(file_test_path).read())
		test_words = file_test_path.split('\n')
		test_words.pop()

		# variable initialisation
		accuracy = 0
		predicted = "a"
		all_accurate_words = []
		all_predicted_words = []
		all_accurante_words_by_predicted_index = {} # occurence of success by index
		
		if number_samples > 0: # Si il y a un nombre précisé de test

			for i in range(1, number_samples + 1):
				
				starting_letters = "" # on reset la lettre de départ
				for n in range(number_starting_characters): # pour chaque caractère qu'on donne pour la prédiction
					random_character = random.randint(0, len(string.ascii_uppercase) - 1) # on genère un caractère aléatoire
					starting_letters = starting_letters + string.ascii_uppercase[random_character] # on ajoute le caractère généré aléatoirement aux lettres de départs

				predicted = self.generate(initial_str=starting_letters, predict_len=predict_len, temperature=temperature).lower() # on genère une prédiction puis on vérifie qu'elle n'a pas déjà était faite
				predicted = predicted.split('\n')
				predicted.pop() # remove the last word because it's probably cut

				
				
				for j, word in enumerate(list(predicted)):
					if word in all_predicted_words: predicted.remove(word)
					else:
						all_predicted_words.append(word)
						if word in test_words:
							all_accurate_words.append(predicted)
							if str(j) in all_accurante_words_by_predicted_index.keys():
								all_accurante_words_by_predicted_index[str(j)]
							else: all_accurante_words_by_predicted_index[str(j)] = [word] 
					

				for word in predicted:
					if word in test_words: all_accurate_words.append(word)

				accuracy = 100 * len(all_accurate_words) / len(all_predicted_words)

				if i % print_scale == 0:

					#print(f'testing : {i}')

					with open(f'logs/{test_name}.json', 'w') as f:
						f.write(json.dumps((i, f'{accuracy}%', all_accurante_words_by_predicted_index, all_predicted_words)))

					#print(f'{i}\t\t: {accuracy}% Accuracy')

		else:
			i = 0
			percentage_words_found = 0 / len(test_words)
			while percentage_words_found < percent:
				nc = random.randint(1, int(predict_len / 2 - 1))

				while predicted in all_predicted_words:
					starting_letters = ""
					for n in range(nc):
						rc = random.randint(0, len(string.ascii_uppercase) - 1)
						starting_letters = starting_letters + string.ascii_uppercase[rc]

					predicted = self.generateOneWord(initial_str=starting_letters, predict_len=predict_len, temperature=temperature).lower() # on genère une prédiction puis on vérifie qu'elle n'a pas déjà était faite

				all_predicted_words.append(predicted)

				if predicted in test_words:
					all_accurate_words.append(predicted)

				accuracy = 100 * len(all_accurate_words) / len(all_predicted_words)
				percentage_words_found = 100 * len(all_accurate_words) / len(test_words)

				if i % print_scale == 0:

					with open(f'logs/{test_name}.json', 'w') as f:
						f.write(json.dumps((i, f'{percentage_words_found}%', f'{accuracy}%', len(all_accurate_words), all_accurate_words, len(all_predicted_words), all_predicted_words)))

					print(f'{i}\t\t: {percentage_words_found}% words found with {accuracy}% Accuracy')

# get files
#file_train_path = 'data/RussianTrain.txt'
#file_test_path = 'data/RussianTest.txt'

file_train_path = 'data/PasswordTrain.txt'
file_test_path = 'data/PasswordEval.txt'

file = unidecode.unidecode(open(file_train_path).read())

gennames = Generator()
#gennames.rnn = rnn.RNN(n_characters, gennames.hidden_size, gennames.num_layers, n_characters, device).to(device)
gennames.rnn = lstm.LSTM(n_characters, gennames.hidden_size, gennames.num_layers, n_characters, device).to(device)
#gennames.rnn = gru.GRU(n_characters, gennames.hidden_size, gennames.num_layers, n_characters, device).to(device)

#gennames.train('russian_names', unidecode.unidecode(open(file_test_path).read()))
#gennames.testing_batch('models/russian_names_250_5000_1_256_2_0.003.pt', 'russian_names_test_batch_1', file_test_path, number_samples=10000)
gennames.evaluate_epoch_train('password_lstm', file_test_path)
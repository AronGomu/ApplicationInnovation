from os import write
import string
import json
import random
import math

numbers = "0123456789"
lower_letters = "abcdefghijklmnopqrstuvwxyz"
upper_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
special_characters = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~♂♀"

def removeLastCharactersOfLine(file_path, save_path):
	with open(file_path, 'r') as infile, \
		open(save_path, 'w') as outfile:
		text = infile.read()
		text = text.replace("\\", "")
		outfile.write(text)


def calculateStats(file_path, outfile_path):

	with open(file_path, 'r') as file:
		
		lines = file.readlines()

		total_characters = 0

		number = 0
		lower = 0
		upper = 0
		special = 0
		number_lower = 0
		number_upper = 0
		number_special = 0
		lower_upper = 0
		lower_special = 0
		upper_special = 0
		no_number = 0
		no_lower = 0
		no_upper = 0
		no_special = 0


		nb_words_by_length = {}

		for line in lines:
			nb_numbers = 0
			nb_lower_letters = 0
			nb_upper_letters = 0
			nb_special_characters = 0

			word_length = len(line)
			total_characters += word_length

			if str(word_length) in nb_words_by_length.keys():
				nb_words_by_length[str(word_length)] += 1
			else: 
				nb_words_by_length[str(word_length)] = 1 

			for letter in line:
				if letter in numbers: nb_numbers += 1
				elif letter in lower_letters: nb_lower_letters += 1
				elif letter in upper_letters: nb_upper_letters += 1
				elif letter in special_characters: nb_special_characters += 1
			
			if nb_numbers != 0 and nb_lower_letters == 0 and nb_upper_letters == 0 and nb_special_characters == 0: number += 1
			elif nb_numbers == 0 and nb_lower_letters != 0 and nb_upper_letters == 0 and nb_special_characters == 0: lower += 1
			elif nb_numbers == 0 and nb_lower_letters == 0 and nb_upper_letters != 0 and nb_special_characters == 0: upper += 1
			elif nb_numbers == 0 and nb_lower_letters == 0 and nb_upper_letters == 0 and nb_special_characters != 0: special += 1
			
			elif nb_numbers != 0 and nb_lower_letters != 0 and nb_upper_letters == 0 and nb_special_characters == 0: number_lower += 1
			elif nb_numbers != 0 and nb_lower_letters == 0 and nb_upper_letters != 0 and nb_special_characters == 0: number_upper += 1
			elif nb_numbers != 0 and nb_lower_letters == 0 and nb_upper_letters == 0 and nb_special_characters != 0: number_special += 1
			elif nb_numbers == 0 and nb_lower_letters != 0 and nb_upper_letters != 0 and nb_special_characters == 0: lower_upper += 1
			elif nb_numbers == 0 and nb_lower_letters != 0 and nb_upper_letters == 0 and nb_special_characters != 0: lower_special += 1
			elif nb_numbers == 0 and nb_lower_letters == 0 and nb_upper_letters != 0 and nb_special_characters != 0: upper_special += 1

			elif nb_numbers == 0 and nb_lower_letters != 0 and nb_upper_letters != 0 and nb_special_characters != 0: no_number += 1
			elif nb_numbers != 0 and nb_lower_letters == 0 and nb_upper_letters != 0 and nb_special_characters != 0: no_lower += 1
			elif nb_numbers != 0 and nb_lower_letters != 0 and nb_upper_letters == 0 and nb_special_characters != 0: no_upper += 1
			elif nb_numbers != 0 and nb_lower_letters != 0 and nb_upper_letters != 0 and nb_special_characters == 0: no_special += 1

	mean = total_characters / len(lines)

	with open(outfile_path, 'w') as outfile:
		data = {
			"total": len(lines),
			"number": number,
			"lower": lower,
			"upper": upper,
			"special": special,
			"number_lower": number_lower,
			"number_upper": number_upper,
			"number_special": number_special,
			"lower_upper": lower_upper,
			"lower_special": lower_special,
			"upper_special": upper_special,
			"no_number": no_number,
			"no_lower": no_lower,
			"no_upper": no_upper,
			"no_special": no_special,
			"mean": mean,
			"nb_words_by_length": nb_words_by_length
			
		}
		outfile.write(json.dumps(data))


def getXRandomword(file_path, outfile_path, n):
	words = []
	with open(file_path, 'r') as file:
		lines = file.readlines()
		for i in range(n): words.append(lines[random.randint(0, len(lines)-1)])
	for i, word in enumerate(list(words)): words[i] = word.replace('\n', '')
	with open(outfile_path, 'w') as file:
		for word in words: file.write(f'{word} ')

def getXRandomGeneratedWord(file_path, outfile_path, n):
	data = []
	with open(file_path, 'r') as file:
		data = json.load(file)
	with open(outfile_path, 'w') as out:
		for i in range(n):
			out.write(f'{data[3][random.randint(0, len(data[3])-1)]} ')

def createGraphDataOccurence(file_path, outfile_path):
	data = {}
	nb_words = 0
	colon_data = 0
	colon_name = ""
	colons = []
	with open(file_path, 'r') as file:
		data = json.load(file)
		nb_words = data['total']
		data = data['nb_words_by_length']
	
	for i in range(1,250):
		if str(i) in data.keys():
			if (colon_data + data[str(i)] > nb_words / 5):
				colons.append((colon_name, colon_data))
				colon_data = 0
				colon_name = ""
			colon_data += data[str(i)]
			colon_name += f'{str(i)}-'
	colons.append((colon_name, colon_data))
	
	with open(outfile_path, 'w') as file:
		for colon in colons: file.write(f'{colon} ')

def createPiePasswordTypes(file_path, outfile_path):
	with open(file_path, 'r') as file, \
		open(outfile_path, 'w') as out :
		data = json.load(file)
		for key in data.keys():
			if key != 'total' and key != 'mean' and key != 'nb_words_by_length':
				out.write(f"{round(data[key] * 100 / data['total'], 2)}/{key},\n")


	


#removeLastCharactersOfLine('eval.txt', 'data/PasswordEval.txt')
#calculateStats('data/RussianTest.txt', 'stats/russian/stats_russian_test.json')
#calculateStats('data/RussianTrain.txt', 'stats/russian/stats_russian_train.json')
#getXRandomword('data/names/Russian.txt', 'stats/russian/random_words_russian.txt', 30)
#createGraphDataOccurence('stats/russian/stats_russian.json', 'stats_russian_occurence_data_graph.txt')
#createPiePasswordTypes('stats/russian/stats_russian.json', 'stats/russian/stats_russian_train_pie_types.txt')

getXRandomGeneratedWord('logs/russian_names_lstm_250_x_1_256_2_0.003/russian_names_train_250_400_1_256_2_0.003.json', 'sample_256_400_russian.txt', 30)
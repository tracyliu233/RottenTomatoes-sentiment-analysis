__author__ = 'tracy'
# -*- coding: utf-8 -*-
import glob
import os
import argparse
import random
from collections import Counter
import numpy as np

stopWords = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also',
             'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be',
	     'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear',
	     'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for',
	     'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers',
	     'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is',
	     'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may',
	     'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor',
	     'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our',
	     'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since',
	     'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then',
	     'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us',
	     've', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which',
	     'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet',
	     'you', 'your']
	     
def parseArgument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('-d', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def main():
	args = parseArgument()
	directory = args['d'][0]
	print directory

	def filelist(pathspec):
		"""This function read in filepath and
		return a list of file path
		"""
		files = []
		for name in glob.glob(pathspec):
			if os.path.getsize(name) > 0:
				files.append(name)
		return files

	def sample(files):
		"""This function read in file path and
		return list of filepath
		"""
		size = len(files)/3
		data_1 = random.sample(files, size)
		rest = []
		for file in files:
			if file not in data_1:
				rest.append(file)
		data_2 = random.sample(rest, size)
		data_3 = []
		for file in files:
			if (file not in data_1) and (file not in data_2):
				data_3.append(file)
		return data_1, data_2, data_3

	def words(filelist):
		"""This function read in a list of file names and
		return Couter object with all the words in files and their count
		and the total number of words
		"""
		wordlist = []
		if isinstance(filelist, basestring):
			fh = open(filelist)
			for line in fh.readlines():
				line = line.strip(" \n")
				words = line.split(" ")
				for word in words:
					if word not in stopWords and len(word)>=3:
						wordlist.append(word)
		else:
			for name in filelist:
				fh = open(name)
				for line in fh.readlines():
					line = line.strip(" \n")
					words = line.split(" ")
					for word in words:
						if word not in stopWords and len(word)>=3:
							wordlist.append(word)
		tol_count = len(wordlist)
		word_counter = Counter(wordlist)
		return word_counter, tol_count

	files_neg =  filelist(directory + "/neg/*.txt")
	dataset_neg = sample(files_neg)
	files_pos =  filelist(directory + "/pos/*.txt")
	dataset_pos = sample(files_pos)

	def classifier(dataset_neg, dataset_pos, x, y, z):
		"""This function read in train and test data and
		return result of classification
		"""
		train_neg = dataset_neg[x] + dataset_neg[y]
		test_neg = dataset_neg[z]
		train_pos = dataset_pos[x] + dataset_pos[y]
		test_pos = dataset_pos[z]

		(neg_counter, neg_count) = words(train_neg)
		(pos_counter, pos_count) = words(train_pos)

		# calculate V
		V = float(len(neg_counter)+len(pos_counter))

		# calculate P_w_neg counter
		P_wc_neg = {}
		for w in neg_counter:
			temp = (neg_counter[w]+1)/(neg_count + V + 1)
			P_wc_neg[w] = temp
		P_wc_neg["Unkown"] = 1/(neg_count + V + 1)

		# calculate P_w_pos counter
		P_wc_pos = {}
		for w in pos_counter:
			temp = (pos_counter[w]+1)/(pos_count + V + 1)
			P_wc_pos[w] = temp
		P_wc_pos["Unkown"] = 1/(pos_count + V + 1)

		# calculate P_c
		total = float(len(train_neg) + len(train_pos))
		P_c_neg = len(train_neg)/total
		P_c_pos = len(train_pos)/total

		# test neg data
		num_neg_correct_docs = 0
		for file in test_neg:
			word_counter= words(file)[0]
			temp_neg = 0
			temp_pos = 0
			for word in word_counter:
				n = word_counter[word]
				if word in P_wc_neg:
					temp_neg = temp_neg + n * np.log(P_wc_neg[word])
				else:
					temp_neg = temp_neg + n * np.log(P_wc_neg["Unkown"])
				if word in P_wc_pos:
					temp_pos = temp_pos + n * np.log(P_wc_pos[word])
				else:
					temp_pos = temp_pos + n * np.log(P_wc_pos["Unkown"])
			P_neg_d = np.log(P_c_neg)+ temp_neg
			P_pos_d = np.log(P_c_pos)+ temp_pos
			if P_neg_d > P_pos_d:
				num_neg_correct_docs += 1

		# test pos data
		num_pos_correct_docs = 0
		for file in test_pos:
			word_counter= words(file)[0]
			temp_neg = 0
			temp_pos = 0
			for word in word_counter:
				n = word_counter[word]
				if word in P_wc_neg:
					temp_neg = temp_neg + n * np.log(P_wc_neg[word])
				else:
					temp_neg = temp_neg + n * np.log(P_wc_neg["Unkown"])
				if word in P_wc_pos:
					temp_pos = temp_pos + n * np.log(P_wc_pos[word])
				else:
					temp_pos = temp_pos + n * np.log(P_wc_pos["Unkown"])
			P_neg_d = np.log(P_c_neg)+ temp_neg
			P_pos_d = np.log(P_c_pos)+ temp_pos
			if P_neg_d < P_pos_d:
				num_pos_correct_docs += 1
		accuracy = float((num_pos_correct_docs + num_neg_correct_docs))/(len(test_pos) + len(test_neg))

		return len(test_pos), len(train_pos), num_pos_correct_docs, len(test_neg), len(train_neg), num_neg_correct_docs, accuracy

	d1 = classifier(dataset_neg, dataset_pos, 0,1,2)
	d2 = classifier(dataset_neg, dataset_pos, 0,2,1)
	d3 = classifier(dataset_neg, dataset_pos, 1,2,0)

	print "iteration 1:"
	print "num_pos_test_docs:" + str(d1[0])
	print "num_pos_training_docs:" + str(d1[1])
	print "num_pos_correct_docs:" + str(d1[2])
	print "num_neg_test_docs:" + str(d1[3])
	print "num_neg_training_docs:" + str(d1[4])
	print "num_neg_correct_docs:" + str(d1[5])
	print "accuracy:" + '{0:.0%}'.format(d1[6])
	print "iteration 2:"
	print "num_pos_test_docs:" + str(d2[0])
	print "num_pos_training_docs:" + str(d2[1])
	print "num_pos_correct_docs:" + str(d2[2])
	print "num_neg_test_docs:" + str(d2[3])
	print "num_neg_training_docs:" + str(d2[4])
	print "num_neg_correct_docs:" + str(d2[5])
	print "accuracy:" + '{0:.0%}'.format(d2[6])
	print "iteration 3:"
	print "num_pos_test_docs:" + str(d3[0])
	print "num_pos_training_docs:" + str(d3[1])
	print "num_pos_correct_docs:" + str(d3[2])
	print "num_neg_test_docs:" + str(d3[3])
	print "num_neg_training_docs:" + str(d3[4])
	print "num_neg_correct_docs:" + str(d3[5])
	print "accuracy:" + '{0:.0%}'.format(d3[6])
	print "ave_accuracy:" + '{0:.0%}'.format((d1[6] + d2[6] + d3[6])/3)

main()






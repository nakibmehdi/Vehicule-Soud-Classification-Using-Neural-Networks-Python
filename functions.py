import librosa
import numpy as np
import os
import random
import pandas as pd
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score,f1_score,precision_score,recall_score
from sklearn import tree
import joblib


def loadData(path):
	return pd.read_csv(path, header = None)

def trainingLogReg(trainData):
	features=trainData.iloc[:,0:193].copy()
	classifier = LogisticRegression(random_state=0)
	classifier.fit(features, trainData[193])
	return classifier

def trainingDecTrees(trainData):
	features=trainData.iloc[:,0:193].copy()
	classifier=DecisionTreeClassifier(random_state=0)
	classifier.fit(features,trainData[193])
	return classifier

def trainingNeuralNetwork(trainData):
	features=trainData.iloc[:,0:193].copy()
	classifier=MLPClassifier()
	classifier.fit(features,trainData[193])
	return classifier

def testing(classifier,testdata):
	prediction = classifier.predict(testdata.iloc[:,0:193])
	kappa = cohen_kappa_score(testdata.iloc[:,-1].tolist(),prediction)
	f1 = f1_score(testdata.iloc[:,-1].tolist(),prediction,average='micro')
	preci = precision_score(testdata.iloc[:,-1].tolist(),prediction,average='micro')
	recall = recall_score(testdata.iloc[:,-1].tolist(),prediction,average='micro')  
	return kappa,f1,preci,recall

def crossValidation(dataset,k=10):
	X = dataset.iloc[:,0:193]
	y = dataset.iloc[:,-1]
	kf = KFold(n_splits=k)
	rettt= []
	avgKappa=0
	avgF1 = 0
	avgPrec = 0
	avgRec = 0
	for train_index, test_index in kf.split(X):
		kappa,f1,preci,recall=testing(trainingNeuralNetwork(dataset.ix[train_index]),dataset.ix[test_index])
		avgKappa += kappa
		avgF1 += f1
		avgPrec += preci
		avgRec += recall
	avgKappa=avgKappa/k
	avgF1 = avgF1/k
	avgPrec = avgPrec/k
	avgRec = avgRec/k
	tot = [avgKappa,avgF1,avgPrec,avgRec]
	rettt.append(tot)
	avgKappa=0
	avgF1 = 0
	avgPrec = 0
	avgRec = 0
	for train_index, test_index in kf.split(X):
		kappa,f1,preci,recall=testing(trainingDecTrees(dataset.ix[train_index]),dataset.ix[test_index])
		avgKappa += kappa
		avgF1 += f1
		avgPrec += preci
		avgRec += recall
	avgKappa=avgKappa/k
	avgF1 = avgF1/k
	avgPrec = avgPrec/k
	avgRec = avgRec/k
	tot = [avgKappa,avgF1,avgPrec,avgRec]
	rettt.append(tot)
	avgKappa=0
	avgF1 = 0
	avgPrec = 0
	avgRec = 0
	for train_index, test_index in kf.split(X):
		kappa,f1,preci,recall=testing(trainingLogReg(dataset.ix[train_index]),dataset.ix[test_index])
		avgKappa += kappa
		avgF1 += f1
		avgPrec += preci
		avgRec += recall
	avgKappa=avgKappa/k
	avgF1 = avgF1/k
	avgPrec = avgPrec/k
	avgRec = avgRec/k
	tot = [avgKappa,avgF1,avgPrec,avgRec]
	rettt.append(tot)
	return rettt

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return [mfccs,chroma,mel,contrast,tonnetz]

def getValues(features):
    x=features.tolist()
    return x

def save_features(directory):
	directoryb = os.fsencode(directory)
	data = []
	nbfile = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
	c = 0
	print("Progress : 0%")
	for file in os.listdir(directoryb):
		filename = os.fsdecode(file)
		if filename.endswith(".wav") :
			data.append(ex_features("dataset/" + filename))
			c += 1
			print("Progress : "+ str(int((100*c)/nbfile)) + "%" )
		else:
			print("File not supported : " + filename)

	df=pd.DataFrame(data)
	df.to_csv( "datasets/dataset.csv" ,  header=False , index=False)
	print("done")
	return df

def ex_features(audoPath):
	mfccs,chroma,mel,contrast,tonnetz = extract_feature(audoPath)
	features = getValues(chroma)
	features.extend(getValues(mfccs))
	features.extend(getValues(mel))
	features.extend(getValues(contrast))
	features.extend(getValues(tonnetz))
	features.append(audoPath[8:12])
	return features

def savemodel(clf,nom):
	joblib.dump(clf,"classifiers/"+nom+'.pkl')


def predict(audio,clasi):
	clas = joblib.load("classifiers/"+clasi)
	d = ex_features(audio)[0:193]
	print(d)
	return clas.predict(d)
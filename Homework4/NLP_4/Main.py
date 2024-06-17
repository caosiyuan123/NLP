# -*- coding: utf-8 -*-
import random
import Model
import jieba
import os
import torch
from gensim import corpora
from torch.utils.data import Dataset
# 运行参数：
databaseFilePath = os.path.join('.\\data\\database', '白马啸西风.txt')  # 测试文本
stopWordsFilePath = '.\\data\\stopwords'  # 停用词目录
dictPath = '.\\models\\S2S.dict'  # 词典保存路径
extraTokens = ['<bos>', '<eos>', '<pad>']  # 添加的token
trainRate = 0.8  # 训练集占比
maxTokens = 512  # 最大token长度
testLength = 5  # 测试初始长度

# 读取文件，清洗数据，获得词段落序列
def readFile(databaseFilePath, stopWordsFilePath):
    # 读取stopWords
    stopWords = []
    for file in os.listdir(stopWordsFilePath):
        with open(os.path.join(stopWordsFilePath, file), 'r', encoding='utf-8') as stopWordFile:
            for line in stopWordFile.readlines():
                stopWords.append(line.strip())  # 去掉回车
    # 读取文件
    databaseFilePath = databaseFilePath
    File = open(databaseFilePath, 'r', encoding='gb18030')
    if File.closed:
        raise IOError(databaseFilePath + 'File Open error!')
    # 读取内容并删除冗余，转换成段落序列
    rawTxt = File.read()
    rawTxt = rawTxt.replace('----〖新语丝电子文库(www.xys.org)〗', '')
    rawTxt = rawTxt.replace('本书来自www.cr173.com免费txt小说下载站', '')
    txtData = rawTxt.replace('更多更新免费电子书请关注www.cr173.com', '')
    sentenceData = txtData.split('\n')
    # 需要段落序列
    words = []
    for par in sentenceData:
        rawparWords = list(jieba.cut(par))
        Words = [word for word in rawparWords if not word.isspace() and word not in stopWords]
        if len(Words) > 0:
            words.append(Words)
    return words


def preProcessing(sentences, tokenSizeLimit):
    OneSentences = []
    for sen in sentences:
        OneSentences.append('<bos>')
        OneSentences.extend(sen)
        OneSentences.append('<eos>')
    postSentences = []
    i = 0
    while(i + tokenSizeLimit < len(OneSentences)):
        postSentences.append(OneSentences[i:i+tokenSizeLimit])
        i += tokenSizeLimit
    postSentences.append(OneSentences[i:len(OneSentences)]
                         + ['<pad>'] * (i+tokenSizeLimit-len(OneSentences)))
    return postSentences

def generateDictionary(sentences, otherTokens, saveDictPath):
    dictionary = corpora.Dictionary([otherTokens])
    dictionary.add_documents(sentences)
    dictionary.save(saveDictPath)
    return dictionary

# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, sentencesOfWords, dictionary):
        self.dictionary = dictionary
        self.sentencesOfWords = sentencesOfWords
    def __len__(self):
        return len(self.sentencesOfWords)
    def __getitem__(self, idx):
        sen = self.sentencesOfWords[idx]
        nowVec = [self.dictionary.token2id[word] for word in sen]
        return torch.LongTensor(nowVec)

if __name__ == '__main__':
    # 读取文件
    sentences = readFile(databaseFilePath, stopWordsFilePath)
    # 插入tokens，padding到相同长度
    sentences = preProcessing(sentences, maxTokens)
    # 创建词典
    dictionary = None
    if os.path.exists(dictPath):
        dictionary = corpora.Dictionary.load(dictPath)
    else:
        dictionary = generateDictionary(sentences, extraTokens, dictPath)

    # 划分训练集和测试集
    random.seed(20240610)
    random.shuffle(sentences)
    trainNum = int(trainRate * len(sentences))
    trainDataset = TextDataset(sentences[:trainNum], dictionary)
    testDataset = TextDataset(sentences[trainNum:], dictionary)
    # 训练
    if not os.path.exists(Model.S2SArgs.save_path):
        Model.S2SModel_train(trainDataset, dictionary, testLength)
    Model.S2SModel_test(testDataset, dictionary, testLength, maxTokens)

    if not os.path.exists(Model.TfArgs.save_path):
        Model.TfModel_train(trainDataset, dictionary, testLength, maxTokens)
    Model.TfModel_test(testDataset, dictionary, testLength, maxTokens)



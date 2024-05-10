import os
import numpy as np
import jieba
import gensim
from sklearn.svm import SVC

class LDAsvm:
    def __init__(self, Mode, docNum, docLen, topicNum, crossNum):
        self.Mode = Mode
        self.docNum = docNum
        self.docLen = docLen
        self.topicNum = topicNum
        self.crossNum = crossNum
        self.sampledData = []

    def proSampleData(self, stopWordsPath, databasePath):
        stopWords = self.loadStopWords(stopWordsPath)
        tokenWithDatas = self.readdatabase(databasePath, stopWords)
        self.sampleData(tokenWithDatas)

    def loadStopWords(self, stopWordsPath):
        stopWords = []
        for file in os.listdir(stopWordsPath):
            with open(os.path.join(stopWordsPath, file), 'r', encoding='utf-8') as File:
                stopWords.extend([line.strip() for line in File.readlines()])
        return stopWords

    def readdatabase(self, databasePath, stopWords):
        tokenWithDatas = []
        for filePath in os.listdir(databasePath):
            with open(os.path.join(databasePath, filePath), 'r', encoding='gb18030') as File:
                rawTxt = File.read().replace('----〖新语丝电子文库(www.xys.org)〗', '').replace(
                    '本书来自www.cr173.com免费txt小说下载站', '').replace('更多更新免费电子书请关注www.cr173.com', '')
                txtData = list(jieba.lcut(rawTxt)) if self.Mode == 'word' else list(rawTxt)
                words = [word for word in txtData if word not in stopWords and not word.isspace()]
                words = np.array(words)
                cutLen = len(words) % self.docLen
                if cutLen != 0:
                    words = words[:-cutLen]
                txtData = np.split(words, len(words) // self.docLen)
                tokenWithDatas.append((filePath.split('.txt')[0], txtData))
        return tokenWithDatas

    def sampleData(self, tokenWithDatas):
        numArray = np.array([len(data[1]) for data in tokenWithDatas])
        numArrayFloat = self.docNum * numArray / numArray.sum()
        numArrayInt = np.floor(numArrayFloat)
        while numArrayInt.sum() < self.docNum:
            Idx = np.argmax(numArrayFloat - numArrayInt)
            numArrayInt[Idx] += 1
        for i in range(len(numArrayInt)):
            ParagNums = numArrayInt[i]
            Label = tokenWithDatas[i][0]
            Docs = tokenWithDatas[i][1]
            sampleParagraphIdxArr = np.random.choice(range(len(Docs)), size=int(ParagNums), replace=False)
            self.sampledData.extend([Label, Docs[paragIdx]] for paragIdx in sampleParagraphIdxArr)

    def getTrainTestSet(self, i):
        groupSize = len(self.sampledData) // self.crossNum
        startIdx = i * groupSize
        endIdx = startIdx + groupSize
        testSet = self.sampledData[startIdx:endIdx]
        trainSet = self.sampledData[:startIdx] + self.sampledData[endIdx:]
        return trainSet, testSet

    def trainLDASVM(self, trainData):
        trainDocData = [data[1] for data in trainData]
        trainDictionary = gensim.corpora.Dictionary(trainDocData)
        traindatabase = [trainDictionary.doc2bow(text) for text in trainDocData]
        ldaModel = gensim.models.LdaModel(corpus=traindatabase, id2word=trainDictionary, num_topics=self.topicNum)
        docLabel = [data[0] for data in trainData]
        probabilityForDocs = np.array(ldaModel.get_document_topics(traindatabase, minimum_probability=0.0))[:, :, 1]
        model = SVC(kernel='linear', probability=True)
        model.fit(probabilityForDocs, docLabel)
        return model,ldaModel,trainDictionary

    def Test(self, classifyModel, ldaModel, trainDictionary, testSet):
        testDocData = [data[1] for data in testSet]
        testLabel = [data[0] for data in testSet]
        testdatabase = [trainDictionary.doc2bow(text) for text in testDocData]
        testProbability = np.array(ldaModel.get_document_topics(testdatabase, minimum_probability=0.0))[:, :, 1]
        testAccuracy = classifyModel.score(testProbability, testLabel)
        return testAccuracy

    def LDACSVMTrainAndTest(self):
        np.random.shuffle(self.sampledData)
        testAccuracySum =0
        for i in range(self.crossNum):
            trainSet, testSet = self.getTrainTestSet(i)
            classifier,ldaModel,trainDictionary= self.trainLDASVM(trainSet)
            testAccuracy = self.Test(classifier, ldaModel, trainDictionary, testSet)
            testAccuracySum += testAccuracy
        return  testAccuracySum / self.crossNum


def main():
    dataRootPath = './data/'
    databaseFilePath = os.path.join(dataRootPath, 'database')
    stopWordsFilePath = os.path.join(dataRootPath, 'stopwords')
    docNum = 1000
    modeList = ['char', 'word']
    docLenmode = {'char': [20, 100, 500, 1000],'word': [20, 100, 500, 1000]}
    topicNumList = [2, 5, 10, 20, 50, 100]
    crossNum = 10
    for mode in modeList:
        docLenList = docLenmode[mode]
        for docLen in docLenList:
            for topicNum in topicNumList:
                ldaClassifyModel = LDAsvm(Mode=mode, docNum=docNum, docLen=docLen,
                                                     topicNum=topicNum, crossNum=crossNum)
                ldaClassifyModel.proSampleData(stopWordsPath=stopWordsFilePath, databasePath=databaseFilePath)
                meanTest = ldaClassifyModel.LDACSVMTrainAndTest()
                print(f'Mode: {mode}, docLength: {docLen}, topicNum: {topicNum}')
                print(f'average test accuracy: {meanTest:.4f}')

if __name__ == '__main__':
    main()

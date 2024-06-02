# -*- coding:utf-8 -*-
import os
import numpy as np
import jieba
from gensim.models import Word2Vec
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def loadStopWords(stopWordsPath):
    stopWords = []
    for file in os.listdir(stopWordsPath):
        with open(os.path.join(stopWordsPath, file), 'r', encoding='utf-8') as File:
            stopWords.extend([line.strip() for line in File.readlines()])
    return stopWords
def readdatabase(databasePath: str ,stopWords: list):
    word = []
    words = []
    for filePath in os.listdir(databasePath):
        with open(os.path.join(databasePath, filePath), 'r', encoding='gb18030') as File:
            rawTxt = File.read().replace('----〖新语丝电子文库(www.xys.org)〗', '').replace(
                '本书来自www.cr173.com免费txt小说下载站', '').replace('更多更新免费电子书请关注www.cr173.com', '')
            rawTxt = rawTxt.split('\n')
            for par in rawTxt:
                txtData = list(jieba.lcut(par))
                word = [word for word in txtData if word not in stopWords and not word.isspace()]
                if len(word) > 0:
                    words.append(word)
    return words
def main():
    dataRootPath = './data/'
    databaseFilePath = os.path.join(dataRootPath, 'database')
    stopWordsFilePath = os.path.join(dataRootPath, 'stopwords')

    stopWords = loadStopWords(stopWordsFilePath)
    words = readdatabase(databaseFilePath, stopWords)
    model = Word2Vec(words, vector_size=100 ,window=5 ,min_count=5 ,epochs=50 , workers= 3,negative=10)


    #有效指标1-测试词语向量之间的距离
    similarity1=model.wv.similarity('郭靖', '黄蓉')
    print(f'Similarity between 郭靖 and 黄蓉 is {similarity1}')
    similarity2=model.wv.similarity('杨过', '小龙女')
    print(f'Similarity between 杨过 and 小龙女 is {similarity2}')

    #有效指标2-测试词对
    simWord = model.wv.most_similar(positive=['杨过', '黄蓉'], negative=['小龙女'])[0]
    print(f'杨过-小龙女 is 黄蓉-{simWord}')

    #有效指标3-K-means方法
    CoreWord =  '陈家洛 袁承志 杨过 张无忌 萧峰 令狐冲 狄云'.split(' ')
    wordVecs = []
    for w in CoreWord:
        wordVecs.extend(model.wv.similar_by_word(w, topn=10))
    wordsGroup = [item[0] for item in wordVecs]
    words2Vec = np.array([model.wv[word] for word in wordsGroup])
    kMeansModel = KMeans(n_clusters=7)
    kMeansModel.fit(words2Vec)

    #有效指标4-计算测试段落距离
    paragraphs = [
        '''这般于一刹那间化刚为柔的急剧转折，已属乾坤大挪移心法的第七层神功，灭绝师太武功虽高，但于对方刚猛掌力袭体之际，再也难以拆解他转折轻柔的擒拿手法。
        张无忌虽然得胜，但对灭绝师太这般大敌，实是戒惧极深，丝毫不敢怠忽，以倚天剑指住她咽喉，生怕她又有奇招使出，慢慢的退开两步。''',
        '''
        灭绝师太横剑一封，正要递剑出招，张无忌早已转得不不知去向。他在未练乾坤大挪移法之时，轻功已比灭绝师太为高，这时越奔越快，如风如火，似雷似电，连韦一笑素以轻功
        睥睨群雄，也自暗暗骇异。但见他四下转动，迫近身去便是一刀，招术未老，已然避开。这一次攻守异势，灭绝师太竟无反击一剑之机，只是张无忌碍于倚天剑的锋锐，却也不敢
        过份逼近。
        '''
    ]
    words1 = jieba.cut(paragraphs[0])
    words2 = jieba.cut(paragraphs[1])
    vec1 = (np.array([model.wv[word] for word in words1 if word in model.wv.index_to_key]).mean(axis=0))
    vec2 = (np.array([model.wv[word] for word in words2 if word in model.wv.index_to_key]).mean(axis=0))
    simPar = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    print(f'语段1和语段2的语义关联性是{simPar}')

    # TSNE 降维,后进行二维空间投影
    tsne = TSNE(n_components=2, random_state=0)
    tsneMod = tsne.fit_transform(words2Vec)
    vectorLabels = np.vstack((tsneMod[:, 0], tsneMod[:, 1], kMeansModel.labels_)).transpose()

    colors = [ 'yellow', 'red', 'green', 'blue','black', 'cyan']
    #设置中文字体，否则乱码
    zhfont1 = matplotlib.font_manager.FontProperties(fname='./AlibabaPuHuiTi-2-35-Thin.ttf',size =16)
    for i in range(len(vectorLabels)):
        data = vectorLabels[i]
        if data[2] < 5:
            plt.plot(data[0], data[1], color=colors[int(data[2])],marker='.')
            plt.annotate(wordsGroup[i], xy=(data[0], data[1]), xytext=(data[0] + 0.05, data[1] + 0.05),fontproperties = zhfont1)
    plt.show()
if __name__ == '__main__':
    main()
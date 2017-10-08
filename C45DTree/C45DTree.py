# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 22:00:44 2017

@author: zzd
"""

import numpy as np
import DrawDTree
import copy

class C45DTree(object):
    def __init__(self):
        self.tree={}
        self.nodeName=[]
        self.trainSet=[]
        self.labels=[]
    
    def fit(self,trainSet,labels,nodeName):
        self.trainSet=trainSet
        self.labels=labels
        self.nodeName=copy.copy(nodeName)
        self.tree=self.createTree(trainSet,labels,nodeName)
        
        #
        pass
    #构建树
    def createTree(self,trainSet,labels,nodeName):
        #程序终止条件1：如果labels中只有一种决策标签了，停止划分，返回这个决策标签
        if labels.count(labels[0])==len(labels):
            return labels[0]
        #程序终止条件2：如果数据集中的决策标签只剩一个了，则返回这个决策标签
        if len(trainSet[0])==1:
            return self.maxLabel(labels)
        
        #算法核心
        #返回数据集的最有特征轴
        bestFeat,featValueList=self.getBestFeat(trainSet,labels)

        bestFeatNodeName=nodeName[bestFeat]
        tree={bestFeatNodeName:{}}
        del(nodeName[bestFeat])
        #抽取最优特征轴的列向量

        #决策树递归生长
        for value in featValueList:
            subNodeName=nodeName[:]
            #按最优特征列和值分割数据集
            [subTrainSet,subLabels]=self.splitTrainSet(trainSet,labels,bestFeat,value)
            #构建子树
            subTree=self.createTree(subTrainSet,subLabels,subNodeName)
            tree[bestFeatNodeName][value]=subTree
        return tree
        
    #计算最优特征
    def getBestFeat(self,trainSet,labels):
        #计算特征向量维
        numFeatures=len(trainSet[0])
        #计算训练数据行数
        totality=len(trainSet)
        #基础熵，源数据的信息熵
        baseEntropy=self.calEntropy(trainSet,labels)
        #初始化条件熵
        conditionEntropy=[]
        #计算信息增益率
        splitInfo=[]
        #特征向量列表
        allFeatVList=[]
        
        for f in range(numFeatures):
            featList=[example[f] for example in trainSet]
            [splitI,featureValueList]=self.calSplitInfo(featList)
            allFeatVList.append(featureValueList)
            splitInfo.append(splitI)
            resultGain=0.0
            for value in featureValueList:
                [subTrainSet,subLabels]=self.splitTrainSet(trainSet,labels,f,value)
                appearNum=float(len(subTrainSet))
                subEntropy=self.calEntropy(subTrainSet,subLabels)
                resultGain+=(appearNum/totality)*subEntropy
            #总条件熵
            conditionEntropy.append(resultGain)
        #计算信息增益
        infoGainArray=baseEntropy*np.ones(numFeatures)-np.array(conditionEntropy)
        #计算信息增益率
        infoGainRatio=infoGainArray/np.array(splitInfo)
        bestFeatureIndex=np.argsort(-infoGainRatio)[0]
        return bestFeatureIndex,allFeatVList[bestFeatureIndex]
        
        
    #计算信息熵（又叫香农熵）
    def calEntropy(self,trainSet,labels):
        #统计labels中各个类别出现的次数
        items=dict([(i,labels.count(i)) for i in labels])
        #初始化信息熵
        infoEntropy=0.0
        dataLen=len(trainSet)
        
        for item in items:
            p=float(items[item])/dataLen
            #信息熵：sum(-p*log(p))，该对数一般取2
            infoEntropy-=p*np.log2(p)
        return infoEntropy
    
    #计算划分信息
    def calSplitInfo(self,featureVList):
        numEntries=len(featureVList)
        featureValueSetList=list(set(featureVList))
        valueCounts=[featureVList.count(featVec) for featVec in featureValueSetList]
        
        pList=[float(item)/numEntries for item in valueCounts]
        logList=[item*np.log2(item) for item in pList]
        splitInfo=-sum(logList)
        return splitInfo,featureValueSetList
    
    #划分数据集，删除特征轴所在的数据列，返回剩余的数据集
    def splitTrainSet(self,trainSet,labels,axis,value):
        rtnList=[]
        labList=[]
        for featVec,labVec in zip(trainSet,labels):
            if featVec[axis]==value:
                rFeatVec=featVec[:axis]                
                rFeatVec.extend(featVec[axis+1:])
                
                rtnList.append(rFeatVec)
                labList.append(labVec)

        return [rtnList,labList]
        
    #计算出现次数最多的类别标签
    def maxLabel(self,labels):
        #这里采用出现的次数作为键，由于这里只要出现次数最多的一个标签，所以即使出现次数一样时出现字典元素的覆盖也不要紧
        items=dict([(labels.count(i),i) for i in labels])
        return items[max(items.keys())]
    
    def predict(self,testVec):
        return self.doPredict(self.tree,self.nodeName,testVec)
    
    def doPredict(self,inputTree,featLabels,testVec):
        #树根节点
        root=list(inputTree.keys())[0]
        secondDic=inputTree[root]
        featIndex=featLabels.index(root)
        key=testVec[featIndex]
        valueOfFeat=secondDic[key]
        
        if isinstance(valueOfFeat,dict):
            classLabel=self.doPredict(valueOfFeat,featLabels,testVec)
        else:
            classLabel=valueOfFeat
        return classLabel

if __name__=="__main__":
    dataPath='dataset.dat'
    with open(dataPath,'r') as f:
        conList=f.readlines()

    contents=[row.replace('\n','').split('\t') for row in conList if row.strip()]
    
    
    trainSet=[row[0:-1] for row in contents]    
    labels=[row[-1] for row in contents]
    nodeName=['age','revenue','student','credit']
    
    c45=C45DTree()
    c45.fit(trainSet,labels,nodeName)
    print(c45.tree)
    
    draw=DrawDTree.createPlot(c45.tree)
    
    testVec=['0','1','0','0']
    predicted=c45.predict(testVec)
    print(predicted)
    
    
    
    
    
    
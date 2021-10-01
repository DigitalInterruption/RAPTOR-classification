import numpy as np

from pandas import DataFrame, read_csv
from sklearn.mixture import GaussianMixture

from clustering_tools.dataProcessing import collateBatchData, normaliseData
from clustering_tools.classification import electLabels, remapLabels,\
        mapLabels, evaluate
from clustering_tools.plotClust import multiVar

# clustering function used to train and predict the classNames for the data
def find_clusters(x, y, n_clusters = 9, testSplit = False):
    gmm = GaussianMixture(
            n_components=n_clusters,
            init_params='random',
            n_init=16)
    xLabels = gmm.fit_predict(x)
    if testSplit: yLabels = gmm.predict(y)
    else: yLabels = xLabels
    centers = gmm.means_
    return centers, xLabels, yLabels

def main(dataset = 'datasetName', iteration = 0, batch = False):
    # flags used to control the clustering process
    testSplit = True
    testSplitRatio = .2
    dropOutliers = False
    norm = True
    normMethod = 'zscaled'

    plot = False
    save = True

    # index map of features to be used for clustering
    featureMask = [
            'degree',
            'in-closeness-wtd',
            'closeness-unwtd',
            'betweenness-wtd',
            'first-order-influence-wtd',
            'clustering-coefficient'
            ]

    # import datasets, if test split not requested X and Y are the same, this
    #   means two separate implementations are not needed
    X, xLabels, xBounds, xSamples, Y, yLabels, yBounds, ySamples,\
            numSamples, sampleBounds, numDataPoints, trueLabels, classDict =\
            collateBatchData(dataset, featureMask,
                    testSplitRatio, testSplit, dropOutliers)
    numSamples = len(sampleBounds)
    T = numSamples - 1

    # normalise the data by the selected method if flag is set
    if norm: X, Y  = normaliseData(X, Y, normMethod)

    # run the clustering function 
    cent, xPredictedLabels, yPredictedLabels = find_clusters(X, Y,
            len(classDict), testSplit)

    # elect a class for each sample based on the mode cluster assignment of
    #   the elements representing that sample
    if testSplit:
        predictedLabels = []
        for l in [xPredictedLabels, yPredictedLabels]: predictedLabels.extend(l)
    else: predictedLabels = xPredictedLabels
    pLabels, tLabels, pConf = electLabels(
            predictedLabels, trueLabels, sampleBounds)
    classes = np.unique(tLabels)

    # map clusters into their respective classes and evaluate performance.
    #   cluster mapping is done with all samples but evaluation should be for
    #   the testing set only to check for model overfitting
    if testSplit:
        trainAccuracy, _, remappedClasses = remapLabels(pLabels, tLabels)
        pLabels, tLabels, pConf = electLabels(
                yPredictedLabels, yLabels, yBounds)
        remappedLabels = mapLabels(pLabels, remappedClasses)
        testAccuracy = evaluate(remappedLabels, tLabels)
    else:
        trainAccuracy, remappedLabels, remappedClasses = remapLabels(pLabels,
                tLabels)

    print('training accuracy =', trainAccuracy * 100)
    if testSplit: print('testing accuracy =', testAccuracy * 100)

    # plot the multi-variate data if flag is set
    if plot:
        if testSplit: multiVar(Y, yLabels, cent, list(classDict.keys()))
        else: multiVar(X, xLabels, cent, list(classDict.keys()))

    # save results to file if flag is set
    if save:
        fields = ['hash', 'class', 'prediction', 'confidence']
        bogus = [None] * len(fields)
        if testSplit: sHash = ySamples
        else: sHash = xSamples

        results = DataFrame([bogus] * len(sHash), columns=fields)

        results['hash'] = sHash
        results['class'] = tLabels
        results['prediction'] = remappedLabels
        results['confidence'] = pConf

        if batch:
            results.to_csv(
                    'results/'+ dataset +'_'+ str(iteration) +'_results.csv',
                    index=False)
        else: results.to_csv('results/results.csv', index=False)

        measures = DataFrame(columns=['trainAccuracy', 'testAccuracy'])
        measures['trainAccuracy'] = [trainAccuracy]
        if testSplit: measures['testAccuracy'] = [testAccuracy]
        if batch:
            df = read_csv('results/'+ dataset +'_measures.csv', index_col=0)
            df = df.append(measures, ignore_index=True)
            df.to_csv('results/'+ dataset +'_measures.csv')
        else: measures.to_csv('results/measures.csv', index=False)

if __name__ == "__main__":
    main()

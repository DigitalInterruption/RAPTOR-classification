from pandas import DataFrame

from gaussian import main

# define list of experiment tuples: [dataset, iterations]
experiments = []

# run each experiment for the specified number of iterations
for e in experiments:
    print(e)
    dataset = e[0]
    iterations = e[1]

    # create dataframe csv for results to be added to
    df = DataFrame(columns=['trainAccuracy', 'testAccuracy'])
    df.to_csv('results/'+ dataset +'_measures.csv')

    for i in range(iterations):
        main(dataset, i, True)
        print(i)

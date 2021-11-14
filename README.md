# Recommendation-Engines--Books
an attempt at generating book recommendations through various techniques including collaborative filtering

Data can be found here https://www.kaggle.com/ruchi798/bookcrossing-dataset

My goal was to build a collaborative filtering engine from scratch using this data. i started with some more cursory ways of looking at the data and quickly realized that
there were many many editions of a given book, and treating each as separate was going to cause problems. This is probably the most important learning that I will carry over
to future recommendation systems coding- grouping things by the work instead of the specific edition. 

once i fixed that problem, some surface level analysis like the top rated books yielded some interesting results, but I got stuck on trying to get any sensible results out of 
actually creating a collaborative filtering engine. the dataset is simply too sparse. I injected a list of my own book ratings into the dataset as a test, and I couldn't find any salient results. Ultimately most users in the dataset only provided 1 rating, which is where the problem lay. I could have used something such as SVD to solve for the sparsity, but i'm more interested in learning more about different collaborative filtering techniques and ways of employing deep learning to find similarities in users/items. 

i think the best way to provide recommendations from a dataset this sparse are by providing lists rank-ordered by item-based or user-based similarities. But it's hard to learn mroe about these different techniques, let alone how they'd feed into collaborative filtering, with a dataset so sparse. My next step will be to find a dataset that's more populated.

# MovieRecommendation
Movie Recommendation using Movielens 100k Dataset

# Introduction
This was done as a research project. The Precision and recall scores I got were around the 78 range which is on par with the top score on the leaderboard on paperswithcode.com. 

# Objective
The purpose of this repo was not so much to solve the actual problem but to test a methodology of evaluating different hyperparameters using t-tests on the outcomes. I used the smaller movielens dataset ( 100k) because I wanted to experiment with a model search system where I randomly generate using a selection of possible hyperparameters, including architectural choices, and then later analyze the effect of various settings.

# Files of Interest
The model search notebook file shows the actual search.

The model analysis Full set notebook shows some of the results of the experiments run.

The demo shows the system being used in a minimal fashion. This will only run if you have already trained a model and saved it under the same name as the load method in the demo.

The other files are mostly experimental and were just done out of curiosity.

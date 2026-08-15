## Multiclass Classification
## used to predict more than one class
## two strategies - OneVsOne and OneVsAll

## OneVsAll - a binary classifier is trained for each category
## The category with highest score is the final prediction

## OneVsOne - a binary classifier is trained for each pair of category i.e. there are n(n-1)/2 classifiers
## The category which wins the most contests is the final prediction

## OneVsOne is a better strategy for models which scale poorly with increasing dataset sizes


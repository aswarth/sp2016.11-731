python features.py
~/libsvm-3.18/svm-scale -s svm_train.data.ngram5.range svm_train.data.ngram5 > svm_train.data.ngram5.scale
~/libsvm-3.18/svm-scale -r svm_train.data.ngram5.range svm_test.data.ngram5 > svm_test.data.ngram5.scale
~/libsvm-3.18/svm-train -c 32 -g 0.5 svm_train.data.ngram5.scale svm_train.data.ngram5.scale.model
~/libsvm-3.18/svm-predict svm_train.data.ngram5.scale svm_train.data.ngram5.scale.model svm_train.data.ngram5.scale.predictions
~/libsvm-3.18/svm-predict svm_test.data.ngram5.scale svm_train.data.ngram5.scale.model svm_test.data.ngram5.scale.predictions
cat svm_train.data.ngram5.scale.predictions svm_test.data.ngram5.scale.predictions > output.txt

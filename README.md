### [Text Classification with Deep Learning](https://github.com/tankz0r/Bachelor_thesis/blob/master/Thesis/thesis.pdf)

**ABSTRACT**  
The object of study is advertisements at e-commerce platform. The subject of study is classification model for advertisements.  
The aim of the study is to:  
- find the best methods to represent words in vector space;
- study algorithms and methods for text classification;
- build a software for advertisements classification.
  
Software has been created to classify advertisements using their title and descriptions
The methodology is implemented on the basis of Deep Neural Networks: CNN and RNN. I tried different architectures of NNs, special regularizations and common techniques to achive the best result. All experiments were conducted using the Python programming language and framework for quick prototyping of Neural Networks - Keras. 

**SUMMARY**  
Word representation as vectors in the vector space in combination with Deep learning Neural Networks demonstrate high quality classification on textual data. The layers of NN get meaningful information of each class and can be well distinguish one from another. In this thesis was clearly seen that CNNs can perform nearly the same and in some cases even better than RNNs and they are a right choice to work with textual data. CNNs demonstrated high accuracy on unknown data with appropriate speed. However it is necessary to spend some time to pick the right regularization for layers to avoid overfitting problems.  

![selection_039](https://user-images.githubusercontent.com/13698885/45641871-fb6f7500-bab6-11e8-949b-f0a51d1c5840.jpg)

Future work:  
	- train networks with other optimization algorithms.
	- make an assemble of NNs  
	- use different length of words sequences for titles and descriptions  

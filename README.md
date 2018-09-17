### [Text Classification with Deep Learning](https://github.com/tankz0r/Bachelor_thesis/blob/master/Dissertation/dissertation.pdf)

**ABSTRACT**  
The object of study is advertisements at e-commerce platform. The subject of study is classification model for advertisements.  
The aim of the study is:  
  - to develop research methods and algorithms for text transformation into vectors;
	- to study algorithms and methods for text classification;
	- to build a software for advertisements classification
	- item to analyze results  
  
Theoretical and methodological basis of the study is the work of foreign researches in the field of data mining, mathematical modeling, data classification and marketing.  
A software has been created to classify advertisements using their title and descriptions, and present the results of the program on real data.  
The methodology is implemented on the basis of the already known Deep Neural Networks: convolution and recurrent using own developments, which include special architecture of neural networks and use of special regularizations to overcome overfitting problem.   
The software is implemented using the Python programming language and
framework for working with Neural Networks Keras. Recommendations for further research are given.   

**SUMMARY**  
Word representation as vectors in the vector space in combination with Deep learning Neural Networks demonstrate high quality classification of textual data. The layers of NN get meaningful information of each class and can well distinguish one from another. In this thesis we proved that Convolution NNs can perform the same and in some components even better than Recurrent NNs and they are a right choice to work with textual data. Convolution NNs demonstrated high accuracy on unknown data with appropriate speed. However it is necessary to spend some time to pick the right regularization for layers to avoid overfitting problems.  

Future work:  
	- train networks with other training algorithms. For example, it is possible to try SGD or RMSprop with appropriate parameters  
	- make an assemble of neural networks to best use each one`s strong qualities  
	- try to use different words sequences length for titles and descriptions  
	- as the results on categories were not really impressive it is possible to add them into one large called other'  

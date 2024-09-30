- Write your answers to all the questions and challenges
- Conclude with a section named `## Reflection`, in which you answer the following questions:
  - What were the top 3 most challening **AI neural concepts** from doing this lab?
  - What is your comfort level with these concepts now? 
  - What will help you improve your understanding of these concepts? 

  ### Quetions and Answers ###
  1. Why is the os module imported? 
  ans. The `os` module is imported to interact with environment operating system where   python code is running and this `os` module gives for accessing files and directories,managing processes, and performing system tasks.

    reference: Github Copilot

  2. What is pandas? Why is is it imported in this program?
  ans. Pandas is a popular python library used for data manipulation and analysis and it provides data structures like DataFrame and series like a arrays and pandas it also gives extensive of like merging reshaping and cleaning and exploration.

    Pandas in python code `import pandas` as `pd` is used to import the pandas library and `pd` is as alias used for pandas Python community and it has a diverse range of utilites ranging from multiple file formats, and pandas a trusted ally in the data science and machine learning.

  reference: Github Copilot and educative
            (https://www.educative.io/answers/what-is-pandas-in-python)

  3. What is sklearn.datasets? How is it used in in this lab?
  ans. `sklearn.datasets` is a module in the `Scikit-learn` library for python and it gives the datasets for the testing machine learning algorithms, and its loading the datasets and generate and fetching the data sets.

  reference: (https://towardsdatascience.com/how-to-use-scikit-learn-datasets-for-machine-learning-d6493b38eca3)

  4. What is sklearn.model_selection? How is it used in this lab?
  ans. `sklearn.model_selection` is a module in the Scikit-learn library in python it gives the the classes and function for the datasets and evaluating models in the machine learning and it was used for mainly Train and test split and cross-validation and Hyperparameter tuning.

  reference javatpoint (https://www.javatpoint.com/sklearn-model-selection#:~:text=Sklearn's%20model%20selection%20module%20provides,produce%20validation%20and%20learning%20curves.)

  5. What are the use cases of the Sequential? What use case does this lab illustrate?
  ans. The Sequential moldel is a type of model is provided by keras it is used for the bulding the neural network where the layer has one input tensor and one output tensor.
 it was used for the binary classfication and mutli-clas classfication and the regression and the last time series prediction.

 reference:Github copilot

 Challenge 1: Copilot Chat says that "Sequential models are not appropriate when the model has multiple inputs or multiple outputs, or when any layer has multiple inputs or multiple outputs.
 
 What does this mean?
 
 ans. The `Sequential` model in keras is designed for layer has a one input tensor and one output tensor but it was limited.
 the multiple inputs are require more than one variable input but if the model want to take two input tensors it was not supported by the `sequential` model an when it comes to multiple outputs some it may have the more than one output and multi-class classification and also determine the location of the object in the image This would require two output tensors, which is not supported by the Sequential model and the Layers wih Multiple inputs or outputs has some layers are the addition layer and multiple output tensors of the layers are not supported by the `Sequential` model.

 reference: Github copilot

6. What are the key properties of a Dense layer in a Sequential modle?
Ans. A Dense layer in a Keras Sequential model is a fully connected layer with number of neurons and transform inputs and more accurate prediction and shape of the input and the output layers.

reference: Github copilot and keras.io(https://keras.io/api/layers/core_layers/dense/)

Challenge 2: Copilot Chat explains the use of the Dropout layer as a "regularization technique that prevents overfitting in neural networks."

What is overfitting?
ans. Overfitting is when a model learns from the tranning data too well and including noise and perform without quality on the data.

How can overfitting be prevented?
ans. the overfitting prevented by the creating the data from the present data and the addoing a discourage complex models and random nuerons in the duaring the training and if thew training end and performance will be deceare  and more data can help the model generalize for the better.


Challenge 3: Copilot Chat explains that the to_categorical() function is used for "one-hot encoding" of categorical data of target variable.

Find a simple example that explains the need for one-hot encoding.

ans: One-hot encoding it was a process of converting the data variable into a machine learning algorithms to get the better perdictions.

the simple example:
 consider a color has featrure with three categories like a Red, Green, and Blue in machine machnine learning algorithms these strings cannot be used for directly first map them to numerical values like Red=1 and Green=2, Blue=3 it was like a ordered between the colors to avoid this one-hot encoding, which would conevert the color feature into three new binary features and each color ir represented by a binary vector.

references: Github copilot and medium (https://medium.com/@michaeldelsole/what-is-one-hot-encoding-and-how-to-do-it-f0ae272f1179)


Challenge 4: Based on your understanding of the DataFrame data structure, explain why it is considered to be a "dictionary-like container of Series objects."

ans. A `DataFrame` in pandas is often describe as a "dictionary-like container of Series objects" it have the keys and values here keys are column names and the values are Series objects in each column.
 
 A Data frame can be used for column names like in a key

 for example:
     df[`column_name`] it like the Series object in the `column_name` column

     a row in dataframe looks like this df.loc[0]

 each column in a DataFrame is a Series object and it is a array-like object can have the data type and it has the index of the each element  when access a column in Dataframe it a access the Series object and when access a row  like accessing a slice of many Series objects.

 reference: Github Copilot
  
  7. What are the key attributes and methods of the Bunch data type?
  ans. The `Bunch` object in scikit-learn is a container that has a keys as attributes it like a dictionary, but with added benfit you canacces values using dot notation like attribute access 
  for example:
     bunch["value_key"] by a attribute, `bunch.value_key` 

    refernces: Github Copilot and scikit-learn (https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html)

  8.  What are the values of the following key attributes:
      1. data
      2. target
      3. target_names
      4. feature_names
      5. filename
  ans. The kay attributes of the `Bunch` object when it's load in the datasets in the scikit-learn.
       1. data: the data to learn, array of the shape  n_samples * n_features.
       2. target: the classification labels or regression values.
       3. feature_names: The names of the dataset columns.
       4. target_names: the names of target classes.
       5. filename: The physical location of the iris csv dataset or normal dataset location.

       refernces: Github Copilot

   9.  Briefly describe the purpose of each cell. Be specific in terms of implementation decisions related to this particular lab.
   ans. The remaining 6 cells are the typical workflow for building the a supervised learning AI system in a image classification.

    cell: 1 
     `features_train, features_test, target_train, target_test = model_selection.`
     `rain_test_split(features, target, test_size=0.2, random_state=42)`

     the data set into training and testing sets, and 80% used of the data training and 20% data used for the testing and splints generate reproduction result and `random_state=42`
    
    cell:2
     `number_classes = 3`
     this line sets the number of class in the target variable for iris dataset and it has three classes

     `target_train = to_categorical(target_train, number_classes)`
     `target_test = to_categorical(target_test, number_classes)`
     these line are convert the class integers to binary class matrices which is multi-class classfication tasks.

    cell:3
    `model = Sequential()`
    these line initializes a new Sequential models and it will build the model in keras.
    `model.add(InputLayer(input_shape=(4,)))`
    these line add the layers to the model and the input model with 4 input features (sepal length, sepal width, petal length, petal width)
    `model.add(Dense(64, activation='relu'))`
    here dense layers are connected with neurons and `relu` is Rectified Linear Unit.
    `model.add(Dropout(0.2))`
    the dropout layer for prevent overfiting and it randomly sets 20% of the input units to 0 at the during time.
    `model.add(Dense(3, activation='softmax'))`
    this it was a output layer of the model with 3 neurons and the activation function is a softmax it was used for the multi-class classfication.

    cell:4
    `model.compile(loss="categorical_crossentropy", optimizer="adam", metri=["accuracy"])`

    this line compiles the model loss function `categorical_crossentropy` is a multi class classfication and `optimizer` `adam` is a efficient varient Gradient Descent algorithm is nothing but Initialize parameters  with some values and calculate the cost and greatest increase of the function, and the `metrices` valuated by during the training and testing accuracy. 

    cell:5
    `model.fit(features_train, target_train, batch_size=64, epochs=32, verbose=1)`

    here this `model.fit` is nothing but the number of `epochs` mean number of the iteration on dataset and `batch_size` is `64` it means the model gives 64 samples of the data at the training time.
     
    cell:6
    `loss, accuracy = model.evaluate(features_test, target_test)`
    here tis line evaluate the model's performance on the test data

    `print(f"Test loss: {loss}")`
    `print(f"Test accuracy: {accuracy}")`
    and the last these two will print the trained models loss and the accuracy on the test data.

Challenge 5:

What is the total number of parameters in the neural network model designed for this lab? Briefly explained your calculations.
Modify model.fit() call to set aside 20% of the data for validation. Does this change influence the model performance in the end? Justify your answer.

ans. the total parameter is depends on the neural network  model of the structure in  this lab we have the three Dense layers and one input layer and the total parameters is `4675`
 calculations:
   `First Dense Layer`: it has input layer has 4 neurons and the Dense layer has 64 neurons  4*64 = 256 weights and there are also 64 biases one for each neuron in the dense layer and the total dense layer is  256 weights + 64 biases = 320 parameter.
   `Second Dense Layer`: it has 64 neurons in previous layer and 64 neurons in Dense layer so it will be like 64*64 = 4096 weights in second dense laye and biases are the 64 and the total is 4096 weights + 64 biases = 4160.
   `output Dense layer`: there are 64 neurons in the previous layer and 3 neurons in the dense layer so there are 64*3 = 192 weights and 3 biases  and the total is 192 weights + 3 biases =195

   adding all these up the total number of parameter in the models is 320 + 4160 + 195 = 4675

   Modify the model.fir() to set aside 20% of the data validation 
   To set aside 20% of the data for the validation in the trainig and modify the `model.fit()` by adding the `validation_split` in the argument.

   for example:
  ` model.fit(features_train, target_train,batch_size=64, epochs=32, verbose=1, validation_split=0.2)`

  by this change it will influence the performance because with a validation set can help to monitor the models performance on the unseen data during training if the performance on the validation set starts to decrease, the validation is set can be used to tune hyperparameters like the learning rate and batch size number of layers on the model's performance on the validation set, by using  a validation set  and that model is able to generalize well to unseen data, and model may overfit to this data and perform poorly on the unseen data.

  ### Reflection
  1. What were the top 3 most challening AI neural concepts from doing this lab?
  2. What is your comfort level with these concepts now?
  3. What will help you improve your understanding of these concepts?

  ### Answers
    1. What were the top 3 most challening AI neural concepts from doing this lab?
    ans. The 3 Most  challenging of AI neural concepts:
     1. Understanding Dense Layers: 
        Dense layer, also know as fully connected layers are the core of the building  of the blocks  of neural network, each neuron in a denser layer receives input from the neurons of the previous layer.
     2. Activation Function: 
       in this models the `relu` and `softmax` as activation functions, why are they nessary, and how the different activation function and when and to work is the complex.
    3. Model Complilation And Training:
       the process of compiling a model, which includes setting a loss function and optimizer, and metrics, and then training  the  models  which includes like setting the loss function, optimizer and the metrics and model.fit(), involves servals important concepts, like how the loss function quantifies the models performance.

    2. What is your comfort level with these concepts now?
    ans. My comfort level with thses concepts now quite good and i started understand the difference neural networks and symbolic networks  by doing this labs iam gaining the knowledge of the Artifical intelegence and machine learning models.

    3. What will help you improve your understanding of these concepts?
    ans. To improve my understanding i will try to understand each cell in the single lab like line to line i will gain the information and  finding the defination on the each module and function and concepts and also i will also  try to look other datasets information and code snipnets and AI and ML articles.

    







           
    


 









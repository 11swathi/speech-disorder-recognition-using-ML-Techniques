# Speech-disorder-recognition-using-ML-Techniques

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

### Project Structure
This project has :
1. model.py - This contains code fot our Machine Learning model to predict the disorder by using the training and testing audio files.
2. app.py - This contains Flask APIs that receives audio through GUI or API calls, computes the precition based on our model and returns it.

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000




You should be able to view the homepage as below :



![HomePage](https://github.com/11swathi/speech-disorder-recognition-using-ML-Techniques/blob/main/HomePage.png)




Hit Record, then read the given sentence and hit stop.

Now click on upload and hit Predict see what disorder you have. You can also save the recording locally for future use.

If everything goes well, you should  be able to see the predcited disorder on the HTML page!




![Predicted](https://github.com/11swathi/speech-disorder-recognition-using-ML-Techniques/blob/main/MianPage.png)


Predicted Disorders:



![Disorder](https://github.com/11swathi/speech-disorder-recognition-using-ML-Techniques/blob/main/Disorder1.png)



![Disorder](https://github.com/11swathi/speech-disorder-recognition-using-ML-Techniques/blob/main/Disorder.png)

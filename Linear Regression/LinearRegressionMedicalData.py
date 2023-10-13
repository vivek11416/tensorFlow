import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# Read data
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

#Encoding Categorical Vars
insurance = pd.get_dummies(insurance).astype(int)


#Split data



X = insurance.loc[:, insurance.columns != 'charges']
y = insurance['charges']

train_x,test_x,train_y, test_y = train_test_split(X,y, test_size=0.2,random_state=42)




#creating model

model =  tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01),
              metrics=['mae'])

model.fit(train_x,train_y,epochs=50)

pred_y=model.predict(tf.constant(test_x),)



df = pd.DataFrame(list(zip([x for x in range(len(test_y))],test_y,tf.squeeze(pred_y).numpy() )),
               columns =['Ind','Actual', 'Pred'])


plt.scatter(df['Ind'],df['Actual'])
plt.scatter(df['Ind'],df['Pred'])
plt.show()











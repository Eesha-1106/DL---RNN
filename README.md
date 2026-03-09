# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
 to develop a Recurrent Neural Network (RNN) model to predict future stock closing prices. The model is trained on historical stock data and evaluated by comparing its predictions against actual prices on a test set. The key steps involve data preprocessing (scaling and sequence creation), defining the RNN architecture, training the model, and then visualizing the predicted vs. actual prices.
 

## DESIGN STEPS
### STEP 1: 
Data Preprocessing: Load, scale, and sequence stock price data.

### STEP 2: 
Tensor Conversion & DataLoader: Convert data to PyTorch tensors and create

### STEP 3: 
RNN Model Definition: Define the RNN architecture


### STEP 4: 
Model Setup: Initialize model, loss function, and optimizer.


### STEP 5: 
Model Training: Train the RNN model for 20 epochs.

### STEP 6: 
Prediction & Evaluation: Predict test data and plot results.




## PROGRAM

### Name:Eesha Ranka

### Register Number:212224240040

```python
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
    super(RNNModel,self).__init__()
    self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    self.fc=nn.Linear(hidden_size,output_size)
  def forward(self,x):
    out,_=self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out




# Train the Model

def train_model(model,train_loader,criterion,optimizer,epochs=20):
  train_losses=[]
  model.train()
  for epoch in range(epochs):
    total_loss=0
    for x_batch,y_batch in train_loader:
      x_batch,y_batch=x_batch.to(device),y_batch.to(device)
      optimizer.zero_grad()
      output=model(x_batch)
      loss=criterion(output,y_batch)
      loss.backward()
      optimizer.step()
      total_loss+=loss.item()
    train_losses.append(total_loss/len(train_loader))
    print(f"Epoch [{epoch+1}/{epochs}], Loss:{total_loss/len(train_loader):.4f}")
    # Plot training loss
  print('Name:Eesha Ranka')
  print('Register Number:212224240040')
  plt.plot(train_losses, label='Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE Loss')
  plt.title('Training Loss Over Epochs')
  plt.legend()
  plt.show()


```

### OUTPUT

## Training Loss Over Epochs Plot

![WhatsApp Image 2026-03-09 at 9 25 53 AM](https://github.com/user-attachments/assets/d1af20cd-a001-42a5-b72f-f0022a3eb3c1)

## True Stock Price, Predicted Stock Price vs time

<img width="1115" height="684" alt="image" src="https://github.com/user-attachments/assets/04419375-fbac-4bc4-8cf4-ae078fe0aa81" />

### Predictions
<img width="414" height="105" alt="image" src="https://github.com/user-attachments/assets/08f9fbaf-388b-43f5-92de-08a550a2cb29" />


## RESULT
Thus the model has been trained and predicted the stock price.


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 2, batch_first=True)
        self.linear = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])


def lstm_prediction(epochs=30):
    data = np.sin(np.linspace(0, 100, 1000))

    X, y = [], []
    for i in range(len(data) - 50):
        X.append(data[i:i+50])
        y.append(data[i+50])

    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print("Epoch:", epoch+1, "Loss:", loss.item())

    model.eval()
    with torch.no_grad():
        prediction = model(X[-1].unsqueeze(0))

    print("Predicted value:", prediction.item())

    plt.plot(data)
    plt.title("LSTM Time Series Prediction")
    plt.show()

    return prediction.item()

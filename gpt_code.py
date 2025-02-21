import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Parametreleri içeren bir yapılandırma sözlüğü
config = {
    "input_size": 28*28,  # MNIST için giriş boyutu
    "hidden_layers": [128, 64],  # Gizli katmanların büyüklükleri
    "output_size": 10,  # 10 sınıf (0-9)
    "activation": "relu",  # Aktivasyon fonksiyonu
    "dropout": 0.2,  # Dropout oranı
    "optimizer": "adam",  # Optimizasyon algoritması
    "learning_rate": 0.001,  # Öğrenme oranı
    "batch_size": 64,  # Batch boyutu
    "epochs": 5  # Eğitim süresi
}

# Aktivasyon fonksiyonunu seçen yardımcı fonksiyon
def get_activation(activation_name):
    activations = {
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU()
    }
    return activations.get(activation_name, nn.ReLU())

# Esnek model tanımı
class FlexibleNN(nn.Module):
    def __init__(self, config):
        super(FlexibleNN, self).__init__()
        layers = []
        input_size = config["input_size"]

        for hidden_size in config["hidden_layers"]:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(get_activation(config["activation"]))
            layers.append(nn.Dropout(config["dropout"]))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, config["output_size"]))  # Çıkış katmanı
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten işlemi
        return self.model(x)

# Veriyi yükleme ve dönüştürme
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./mnist", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./mnist", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Modeli oluştur
model = FlexibleNN(config)

# Optimizasyonu ayarla
optimizer_dict = {
    "sgd": optim.SGD(model.parameters(), lr=config["learning_rate"]),
    "adam": optim.Adam(model.parameters(), lr=config["learning_rate"]),
    "rmsprop": optim.RMSprop(model.parameters(), lr=config["learning_rate"])
}
optimizer = optimizer_dict[config["optimizer"]]

# Kayıp fonksiyonu
criterion = nn.CrossEntropyLoss()

# Eğitim döngüsü
def train(model, train_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Modeli eğit
train(model, train_loader, optimizer, criterion, config["epochs"])

import torch
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import resnet

def train(model, trainloader, criterion, optimizer) : 
    for epoch in range(2) :
        running_loss = 0.0 
        for i, data in enumerate(trainloader, 0) :
            inputs, outputs = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            predicted_outputs = model(inputs)
            loss = criterion(predicted_outputs, outputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i % 2000 == 1999) :   
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

def test(model, testloader) :
    correct = 0 
    total = 0
    with torch.no_grad() :
        for data in testloader :
            inputs, outputs = data[0].to(device), data[1].to(device)
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs.data, 1)
            total += outputs.size(0)
            correct += (predicted == outputs).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

if(__name__ == "__main__") :

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device\n") 

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root = "./data", train = True, download = True, transform = transform)
    testset = torchvision.datasets.CIFAR10(root = "./data", train = False, download = True, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 2)
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    model = resnet.ResNet152(len(classes), nn.BatchNorm2d).to(device)
    summary(model, (3, 224, 224), device = device.type)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    train(model, trainloader, criterion, optimizer)
    test(model, testloader)


    
    






    



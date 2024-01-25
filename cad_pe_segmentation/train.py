from tqdm import tqdm
from torchmetrics import Dice
import torch.nn as nn
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
num_gpus = torch.cuda.device_count()

def train(epochs,model,train_loader,val_loader):
    
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
    dice = Dice(average='micro').to(device)
    
    train_loss = []
    val_loss = []
    avg_test_loss = []
    avg_iou = []
    avg_iou_val = []

    for epoch in tqdm(range(epochs)):
        model.train()
        for i, (image, label) in enumerate(train_loader):
            #print(image.shape , label.shape)
            image, label = image, label
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            #print(image.shape , label.shape)
            output = model(image)
            # if epoch == 3 and i <=10 :
            #     plt.subplot(1,2,1)
            #     plt.imshow(output[0].permute(1,2,0).cpu().detach().numpy())
            #     plt.subplot(1,2,2)
            #     plt.imshow(label[0].permute(1,2,0).cpu().detach().numpy())
            #     plt.show()
            #print(output.shape)
            loss = loss_fn(output, label)
            label = label.to(torch.int64)
            #print(output , label)
            train_iou = dice(output, label)

            
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            avg_iou.append(train_iou.item())
            
            
        # evaluate on validation set
        model.eval()
        with torch.no_grad():
            for i, (image, label) in enumerate(val_loader):
                image, label = image, label
                
                image, label = image.to(device), label.to(device)
                output = model(image)
                # plt.subplot(1,2,1)
                # plt.imshow(output[0].permute(1,2,0).cpu().detach().numpy())
                # plt.subplot(1,2,2)
                # plt.imshow(label[0].permute(1,2,0).cpu().detach().numpy())
                # plt.show()
                loss = loss_fn(output, label)
                label = label.to(torch.int64)
                val_iou = dice(output, label)

                val_loss.append(loss.item())
                avg_iou_val.append(val_iou.item())
            
            val_loss_epoch = sum(val_loss) / len(val_loss)
            avg_iou_val_epoch = sum(avg_iou_val) / len(avg_iou_val)
        # print average losses and dice coefficients for the epoch
        train_loss_epoch = sum(train_loss) / len(train_loss)
        
        
        avg_iou_epoch = sum(avg_iou) / len(avg_iou)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss_epoch:.5f} | Val Loss: {val_loss_epoch:.5f} |Dice Coefficient: {avg_iou_epoch:.5f} | Val Dice Coefficient: {avg_iou_val_epoch:.5f}")
    # torch.save(model.state_dict(),"/home/akaniyar/Colonoscopy/all_model/Wp_training.pth")
    # torch.save(
    #     {
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "loss": loss_fn,
    #     },
    #     os.path.join("/home/akaniyar/Colonoscopy/all_model/", f"wp_train_unet80.pth"),
    # )
    
if __name__ == '__main__':
    epochs = 10
    train(epochs, model, train_loader)
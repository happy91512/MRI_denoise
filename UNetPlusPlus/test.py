# import cv2
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from UNet import UNet
# from torch.utils.data import Dataset, DataLoader
from train import *



test_set = MRI_dataset(config['test_imgs_path'], masks=True)
# test_loader = DataLoader(test_set, batch_size = config['batch_size'], shuffle=True)
test_loader = DataLoader(test_set, batch_size = 16, shuffle=True)

model.load_state_dict(torch.load('UNetPlusPlus/weight/UNetPlusPlus_1002_best.ckpt'))
model.eval()
i = 0
test_accs = []
for img in tqdm(test_loader):
    i += 1
    DWI_imgs, masks = img
    DWI_imgs = torch.reshape(DWI_imgs, (DWI_imgs.shape[0], 3, 224, 224))
    masks = torch.reshape(masks, (DWI_imgs.shape[0], 1, 224, 224))
            
    DWI_imgs, masks = DWI_imgs.to(torch.float32).to(device), masks.to(torch.float32).to(device)
    with torch.no_grad():
        out = model(DWI_imgs)


    predicted_mask = (out.cpu().detach().numpy() > 0.5).astype(np.uint8)
    predicted_mask = np.squeeze(predicted_mask, 1)
    masks = masks.cpu().detach().numpy().astype(np.uint8)
    masks = np.squeeze(masks, 1)
    # print(masks.shape, predicted_mask.shape)
    acc = iou(predicted_mask, masks)
    
    
    pred_mask = predicted_mask[6]
    pred_mask[pred_mask > 0] = 255
    masks[masks > 0] = 255
    DWI_imgs = DWI_imgs.cpu().detach().numpy().astype(np.uint8)
    # print(DWI_imgs.shape)
    DWI_imgs = np.reshape(DWI_imgs, (DWI_imgs.shape[0], 224, 224, 3))
    DWI_imgs = DWI_imgs[:, :, :, 0]
    fig = plt.figure()
    plt.subplot(221)
    plt.title('src_img')
    plt.imshow(DWI_imgs[6], cmap='gray')

    plt.subplot(222)
    plt.title('ground_truth')
    plt.imshow(masks[6], cmap='gray')

    plt.subplot(223)
    plt.title('skull_removed')
    plt.imshow(np.bitwise_and(DWI_imgs[6], pred_mask), cmap='gray')

    plt.subplot(224)
    plt.title('pred_mask')
    plt.imshow(pred_mask, cmap='gray')
    
    fig.tight_layout()
    fig.savefig(f"UNetPlusPlus/out/{config['name']}/{i}.jpg")
    plt.close()   
    # cv2.imshow("src", imgs[16])
    # cv2.imshow("predicted_mask", pred_mask)
    # cv2.imshow("truth_mask", masks[16])
    # cv2.imshow("skull_removed", np.bitwise_and(imgs[16], pred_mask))
    # if cv2.waitKey(0) == ord('q'):
    #     break
    test_accs.append(acc)

acc = sum(test_accs) / len(test_accs)
print('acc:', acc)


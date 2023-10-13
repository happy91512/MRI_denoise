from UNetPlusPlus.train import *
from tools import get_filenames

test_set = MRI_dataset(config['test_imgs_path'], masks=False)
test_loader = DataLoader(test_set, batch_size = 6, shuffle=True)

model.load_state_dict(torch.load('weight/model0927_best.ckpt'))
model.eval()



for DWI_imgs in tqdm(test_loader):
    DWI_imgs = DWI_imgs.to(torch.float32).to(device)
    with torch.no_grad():
        out = model(DWI_imgs.unsqueeze(1)).squeeze(1)
        # print(imgs.unsqueeze(1).shape)
    predicted_mask = (out.cpu().detach().numpy() > 0.5).astype(np.uint8)

    pred_mask = predicted_mask[0]
    pred_mask[pred_mask > 0] = 255

    DWI_imgs = DWI_imgs.cpu().detach().numpy().astype(np.uint8)
    cv2.imshow("src", DWI_imgs[0])
    cv2.imshow("predicted_mask", pred_mask)

    cv2.imshow("skull_removed", np.bitwise_and(DWI_imgs[0], pred_mask))
    c_in = cv2.waitKey(0)
    if c_in == ord('q'):
        break
    elif c_in == ord('s'):
        cv2.imwrite('./src.jpg', DWI_imgs[0])
        cv2.imwrite('./mask.jpg', pred_mask)
        cv2.imwrite('./out.jpg', np.bitwise_and(DWI_imgs[0], pred_mask))


from train import *
from tools import get_filenames
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[0]

sys.path.append(str(Path(__file__).resolve().parents[1]))

test_set = MRI_dataset(config['test_imgs_path'], masks=False)
test_loader = DataLoader(test_set, batch_size = 16, shuffle=True)

model.load_state_dict(torch.load('UNetPlusPlus/weight/UNetPlusPlus_1002_best.ckpt'))
model.eval()



for DWI_imgs in tqdm(test_loader):
    DWI_imgs = DWI_imgs.to(torch.float32).to(device)
    DWI_imgs = torch.reshape(DWI_imgs, (DWI_imgs.shape[0], 3, 224, 224))
    with torch.no_grad():
        out = model(DWI_imgs)
        # print(imgs.unsqueeze(1).shape)
    
    predicted_mask = (out.cpu().detach().numpy() > 0.5).astype(np.uint8)
    
    predicted_mask = np.squeeze(predicted_mask, 1)
    pred_mask = predicted_mask[0]
    DWI_imgs = DWI_imgs.cpu().detach().numpy().astype(np.uint8)
    show = np.reshape(DWI_imgs[0], (224, 224, 3))
    show = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
    # print(show.shape, pred_mask.shape)

    cv2.imshow("src", show)
    cv2.imshow("skull_removed", show * pred_mask)
    
    pred_mask[pred_mask > 0] = 255
    cv2.imshow("predicted_mask", pred_mask)

    
    c_in = cv2.waitKey(0)
    if c_in == ord('q'):
        break
    # elif c_in == ord('s'):
    #     cv2.imwrite('./src.jpg', DWI_imgs[0])
    #     cv2.imwrite('./mask.jpg', pred_mask)
    #     cv2.imwrite('./out.jpg', np.bitwise_and(DWI_imgs[0], pred_mask))


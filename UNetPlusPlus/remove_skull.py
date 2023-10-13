from train import *
from tools import get_filenames


model.load_state_dict(torch.load('UNetPlusPlus/weight/UNetPlusPlus_1002_best.ckpt'))
model.eval()
dicoms = sorted(get_filenames('no skull data_paired_UNet++/abnormal/DWI', '*.npy'))
for path in tqdm(dicoms):
    DWI_dicom = np.load(path)
    T2_path = path.replace('DWI', 'T2WI', 1).replace('DWI', 'T2')
    T2_dicom = np.load(T2_path)

    DWI_imgs = np.zeros((DWI_dicom.shape[0], 224, 224))
    T2_imgs = np.zeros_like(DWI_imgs)
    for i in range(len(DWI_imgs)):
        d, t = DWI_dicom[i], T2_dicom[i]
        DWI_imgs[i] = cv2.resize(d, (224, 224))
        T2_imgs[i] = cv2.resize(t, (224, 224))
    DWI_imgs = DWI_imgs.astype(np.float32)
    # print(DWI_imgs.shape, T2_imgs.shape)
    img_in = np.zeros((DWI_imgs.shape[0], 224, 224, 3))
   
    for i in range(len(img_in)):
        im = (DWI_imgs[i] - DWI_imgs[i].min()) / (DWI_imgs[i].max() - DWI_imgs[i].min()) * 255
        img_in[i] = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        
    
        # img_in[i] = cv2.cvtColor(img_in[i], cv2.COLOR_GRAY2BGR)
    
    img_in = np.reshape(img_in, (img_in.shape[0], 3, 224, 224))
    img_in = torch.tensor(img_in).to(torch.float32).to(device)
        # T2_imgs[i] = (T2_imgs[i] - T2_imgs[i].min()) / (T2_imgs[i].max() - T2_imgs[i].min()) * 255

    with torch.no_grad():
        out = model(img_in).squeeze(1)
    
    predicted_mask = (out.cpu().detach().numpy() > 0.5).astype(np.uint8)
    
    removed_DWI_imgs = DWI_imgs * predicted_mask
    removed_T2_imgs = T2_imgs * predicted_mask
    # print(removed_DWI_imgs.mean(), removed_T2_imgs.mean())
    c = 12
    D = DWI_imgs[c]
    T = T2_imgs[c]
    D = ((D - D.min()) / (D.max() - D.min())) * 255
    T = ((T - T.min()) / (T.max() - T.min())) * 255
    
    cv2.imshow('DWI', (D * predicted_mask[c]).astype(np.uint8))
    cv2.imshow('T2', (T * predicted_mask[c]).astype(np.uint8))
    i = cv2.waitKey(0)
    if i == ord('q'):
        break
    elif i == ord('s'):
        cv2.imwrite('out.jpg', (T * predicted_mask[c]).astype(np.uint8))

    # np.save(path, removed_DWI_imgs)
    # np.save(T2_path, removed_T2_imgs)
    

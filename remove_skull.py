from UNetPlusPlus.train import *
from tools import get_filenames


model.load_state_dict(torch.load('weight/model0927_best.ckpt'))
model.eval()
dicoms = sorted(get_filenames('no skull data_paired/normal/DWI', '*.npy'))
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

    img_in = torch.tensor(DWI_imgs).to(torch.float32).to(device)
    for i in range(len(img_in)):
        img_in[i] = (img_in[i] - img_in[i].min()) / (img_in[i].max() - img_in[i].min()) * 255
        
        # T2_imgs[i] = (T2_imgs[i] - T2_imgs[i].min()) / (T2_imgs[i].max() - T2_imgs[i].min()) * 255
    
    with torch.no_grad():
        out = model(img_in)

    predicted_mask = (out.cpu().detach().numpy() > 0.5).astype(np.uint8)
    removed_DWI_imgs = DWI_imgs * predicted_mask
    removed_T2_imgs = T2_imgs * predicted_mask
    
    # cv2.imshow('DWI', DWI_imgs[3])
    # cv2.imshow('T2', T2_imgs[3])
    # i = cv2.waitKey(0)
    # if i == ord('q'):
    #     break
    # elif i == ord('s'):
    #     cv2.imwrite('out.jpg', removed_T2_imgs[3])

    np.save(path, removed_DWI_imgs)
    np.save(T2_path, removed_T2_imgs)
    

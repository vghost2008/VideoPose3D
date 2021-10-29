import numpy as np
from numpy.__config__ import show
from common.camera import normalize_screen_coordinates, image_coordinates
import os.path as osp
from predict_on_video.vis_error import update_camera
import wml_utils as wmlu
import img_utils as wmli
import cv2
from demo_toolkit import show_keypoints
from common.camera import *
from common.sem_dataset import *
from common.toolkit import get_offset

global_cam = {
   'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70, # Only used for visualization
        'orientation': [0.1337897777557373, -0.15692396461963654, -0.7571090459823608, 0.6198879480361938],
        'translation': [1935.4517822265625, 4950.24560546875, 1618.0838623046875],
}

keep_ids = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19,25,26,27]
h36m_cameras_intrinsic_params = [
    {
        'id': '54138969',
        'center': [512.54150390625, 515.4514770507812],
        'focal_length': [1145.0494384765625, 1143.7811279296875],
        'radial_distortion': [-0.20709891617298126, 0.24777518212795258, -0.0030751503072679043],
        'tangential_distortion': [-0.0009756988729350269, -0.00142447161488235],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': 70, # Only used for visualization
    },
    {
        'id': '55011271',
        'center': [508.8486328125, 508.0649108886719],
        'focal_length': [1149.6756591796875, 1147.5916748046875],
        'radial_distortion': [-0.1942136287689209, 0.2404085397720337, 0.006819975562393665],
        'tangential_distortion': [-0.0016190266469493508, -0.0027408944442868233],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': -70, # Only used for visualization
    },
    {
        'id': '58860488',
        'center': [519.8158569335938, 501.40264892578125],
        'focal_length': [1149.1407470703125, 1148.7989501953125],
        'radial_distortion': [-0.2083381861448288, 0.25548800826072693, -0.0024604974314570427],
        'tangential_distortion': [0.0014843869721516967, -0.0007599993259645998],
        'res_w': 1000,
        'res_h': 1000,
        'azimuth': 110, # Only used for visualization
    },
    {
        'id': '60457274',
        'center': [514.9682006835938, 501.88201904296875],
        'focal_length': [1145.5113525390625, 1144.77392578125],
        'radial_distortion': [-0.198384091258049, 0.21832367777824402, -0.008947807364165783],
        'tangential_distortion': [-0.0005872055771760643, -0.0018133620033040643],
        'res_w': 1000,
        'res_h': 1002,
        'azimuth': -110, # Only used for visualization
    },
]

h36m_cameras_extrinsic_params = {
    'S1': [
        {
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        },
        {
            'orientation': [0.6157187819480896, -0.764836311340332, -0.14833825826644897, 0.11794740706682205],
            'translation': [1761.278564453125, -5078.0068359375, 1606.2650146484375],
        },
        {
            'orientation': [0.14651472866535187, -0.14647851884365082, 0.7653023600578308, -0.6094175577163696],
            'translation': [-1846.7777099609375, 5215.04638671875, 1491.972412109375],
        },
        {
            'orientation': [0.5834008455276489, -0.7853162288665771, 0.14548823237419128, -0.14749594032764435],
            'translation': [-1794.7896728515625, -3722.698974609375, 1574.8927001953125],
        },
    ],
    'S2': [
        {},
        {},
        {},
        {},
    ],
    'S3': [
        {},
        {},
        {},
        {},
    ],
    'S4': [
        {},
        {},
        {},
        {},
    ],
    'S5': [
        {
            'orientation': [0.1467377245426178, -0.162370964884758, -0.7551892995834351, 0.6178938746452332],
            'translation': [2097.3916015625, 4880.94482421875, 1605.732421875],
        },
        {
            'orientation': [0.6159758567810059, -0.7626792192459106, -0.15728192031383514, 0.1189815029501915],
            'translation': [2031.7008056640625, -5167.93310546875, 1612.923095703125],
        },
        {
            'orientation': [0.14291371405124664, -0.12907841801643372, 0.7678384780883789, -0.6110143065452576],
            'translation': [-1620.5948486328125, 5171.65869140625, 1496.43701171875],
        },
        {
            'orientation': [0.5920479893684387, -0.7814217805862427, 0.1274748593568802, -0.15036417543888092],
            'translation': [-1637.1737060546875, -3867.3173828125, 1547.033203125],
        },
    ],
    'S6': [
        {
            'orientation': [0.1337897777557373, -0.15692396461963654, -0.7571090459823608, 0.6198879480361938],
            'translation': [1935.4517822265625, 4950.24560546875, 1618.0838623046875],
        },
        {
            'orientation': [0.6147197484970093, -0.7628812789916992, -0.16174767911434174, 0.11819244921207428],
            'translation': [1969.803955078125, -5128.73876953125, 1632.77880859375],
        },
        {
            'orientation': [0.1529948115348816, -0.13529130816459656, 0.7646096348762512, -0.6112781167030334],
            'translation': [-1769.596435546875, 5185.361328125, 1476.993408203125],
        },
        {
            'orientation': [0.5916101336479187, -0.7804774045944214, 0.12832270562648773, -0.1561593860387802],
            'translation': [-1721.668701171875, -3884.13134765625, 1540.4879150390625],
        },
    ],
    'S7': [
        {
            'orientation': [0.1435241848230362, -0.1631336808204651, -0.7548328638076782, 0.6188824772834778],
            'translation': [1974.512939453125, 4926.3544921875, 1597.8326416015625],
        },
        {
            'orientation': [0.6141672730445862, -0.7638262510299683, -0.1596645563840866, 0.1177929937839508],
            'translation': [1937.0584716796875, -5119.7900390625, 1631.5665283203125],
        },
        {
            'orientation': [0.14550060033798218, -0.12874816358089447, 0.7660516500473022, -0.6127139329910278],
            'translation': [-1741.8111572265625, 5208.24951171875, 1464.8245849609375],
        },
        {
            'orientation': [0.5912848114967346, -0.7821764349937439, 0.12445473670959473, -0.15196487307548523],
            'translation': [-1734.7105712890625, -3832.42138671875, 1548.5830078125],
        },
    ],
    'S8': [
        {
            'orientation': [0.14110587537288666, -0.15589867532253265, -0.7561917304992676, 0.619644045829773],
            'translation': [2150.65185546875, 4896.1611328125, 1611.9046630859375],
        },
        {
            'orientation': [0.6169601678848267, -0.7647668123245239, -0.14846350252628326, 0.11158157885074615],
            'translation': [2219.965576171875, -5148.453125, 1613.0440673828125],
        },
        {
            'orientation': [0.1471444070339203, -0.13377119600772858, 0.7670128345489502, -0.6100369691848755],
            'translation': [-1571.2215576171875, 5137.0185546875, 1498.1761474609375],
        },
        {
            'orientation': [0.5927824378013611, -0.7825870513916016, 0.12147816270589828, -0.14631995558738708],
            'translation': [-1476.913330078125, -3896.7412109375, 1547.97216796875],
        },
    ],
    'S9': [
        {
            'orientation': [0.15540587902069092, -0.15548215806484222, -0.7532095313072205, 0.6199594736099243],
            'translation': [2044.45849609375, 4935.1171875, 1481.2275390625],
        },
        {
            'orientation': [0.618784487247467, -0.7634735107421875, -0.14132238924503326, 0.11933968216180801],
            'translation': [1990.959716796875, -5123.810546875, 1568.8048095703125],
        },
        {
            'orientation': [0.13357827067375183, -0.1367100477218628, 0.7689454555511475, -0.6100738644599915],
            'translation': [-1670.9921875, 5211.98583984375, 1528.387939453125],
        },
        {
            'orientation': [0.5879399180412292, -0.7823407053947449, 0.1427614390850067, -0.14794869720935822],
            'translation': [-1696.04345703125, -3827.099853515625, 1591.4127197265625],
        },
    ],
    'S11': [
        {
            'orientation': [0.15232472121715546, -0.15442320704460144, -0.7547563314437866, 0.6191070079803467],
            'translation': [2098.440185546875, 4926.5546875, 1500.278564453125],
        },
        {
            'orientation': [0.6189449429512024, -0.7600917220115662, -0.15300633013248444, 0.1255258321762085],
            'translation': [2083.182373046875, -4912.1728515625, 1561.07861328125],
        },
        {
            'orientation': [0.14943228662014008, -0.15650227665901184, 0.7681233882904053, -0.6026304364204407],
            'translation': [-1609.8153076171875, 5177.3359375, 1537.896728515625],
        },
        {
            'orientation': [0.5894251465797424, -0.7818877100944519, 0.13991211354732513, -0.14715361595153809],
            'translation': [-1590.738037109375, -3854.1689453125, 1578.017578125],
        },
    ],
}

def load_data(kp2d_path='data/data_2d_h36m_gt.npz',kp3d_path='data/data_3d_h36m.npz'):
    keypoints = np.load(kp2d_path, allow_pickle=True)
    kps2d = keypoints['positions_2d'].item()
    kps3d = np.load(kp3d_path, allow_pickle=True)['positions_3d'].item()

    return kps2d,kps3d

def update_cam(cam):
    for k, v in cam.items():
        if k not in ['id', 'res_w', 'res_h']:
            cam[k] = np.array(v, dtype='float32')
                
                # Normalize camera frame
    cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
    cam['focal_length'] = cam['focal_length']/cam['res_w']*2
    if 'translation' in cam:
        cam['translation'] = cam['translation']/1000 # mm to meters
    
    # Add intrinsic parameters vector
    cam['intrinsic'] = np.concatenate((cam['focal_length'], #2
                                       cam['center'], #2
                                       cam['radial_distortion'], #3
                                       cam['tangential_distortion'])) #2
    return cam

def get_error(kps2d,kps3d,cam,scale=1.0):
    kps2d = copy.deepcopy(kps2d)
    kps3d = copy.deepcopy(kps3d)

    kps3d = world_to_camera(kps3d,R=cam['orientation'], t=cam['translation'])
    kps3d = kps3d[:,keep_ids,:]
    kps3d[...,-1] = kps3d[...,-1]*scale
    kps2d = kps2d[...,:2]
    #kps2d = kps2d[:,keep_ids,:]
    kps2d[:,SemDataset.COCO_ID_VALID,:] = kps2d[:,SemDataset.H36M_ID_VALID,:]

    proj_2ds = npproject_to_2d(np.array([kps3d]),np.array([cam['intrinsic']]))[0]
    proj_2ds = unnormalize_screen_coordinates(proj_2ds,w=cam['res_w'],h=cam['res_h'])
    proj_2ds[:,SemDataset.COCO_ID_VALID,:] = proj_2ds[:,SemDataset.H36M_ID_VALID,:]
    scores = np.ones(list(proj_2ds.shape[:-1])+[1])
    scores = scores*SemDataset.COCO_MASK[0]
    valid_len = min(proj_2ds.shape[0],kps2d.shape[0]) 
    proj_2ds = proj_2ds[:valid_len]
    kps2d = kps2d[:valid_len]
    offset = get_offset(kps2d/scale,kps3d,cam)
    proj_2ds = proj_2ds-offset
    proj_2ds = proj_2ds[:,SemDataset.COCO_ID_VALID,:]
    proj_2ds = proj_2ds*scale
    kps2d = kps2d[:,SemDataset.COCO_ID_VALID,:]
    errors = np.abs(proj_2ds-kps2d)
    error = np.mean(errors)
    return error

def show_error(scale=1.0):
    #action = 'Directions 1'
    save_dir = "/home/wj/ai/mldata/pose3d/tmp/vis_human3.6_scale"
    video_path = "/home/wj/ai/mldata/human3.6/S6/Videos/Walking.54138969.mp4"
    wmlu.create_empty_dir(save_dir,remove_if_exists=False)
    save_path = osp.join(save_dir,osp.basename(video_path))
    kps2d,kps3d = load_data()
    all_errors = []
    for subject,kps3d_l0 in kps3d.items():
        for action,kps3d_l1 in kps3d_l0.items():
            for i,cam in enumerate(h36m_cameras_intrinsic_params):
                '''if subject != 'S6':
                    continue
                if action != 'Walking':
                    continue
                if i != 0:
                    continue'''
                cam = copy.deepcopy(cam)
                cam_ex = h36m_cameras_extrinsic_params[subject][i]
                cam.update(cam_ex)
                cam = update_cam(cam)
                error = get_error(kps2d[subject][action][i],kps3d_l1,cam,scale=scale)
                all_errors.append(error)
    
    print(f"All errors")
    print(all_errors)
    print("MEAN error:",np.mean(all_errors),"std error:",np.std(all_errors))

if __name__ == "__main__":
    show_error(scale=100.0)
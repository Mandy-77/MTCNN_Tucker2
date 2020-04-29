# MTCNN_Tucker2
Using DFace as the model of MTCNN and do Tucker2 on the ONet

## Installation

```
cd DFace  
conda env create -f path/to/environment.yml
```
## Dataset preparation
[WIDER FACE](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)

[CNN FacePoint](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)

- Create the DFace train data temporary folder, this folder is involved in the following parameter --dface_traindata_store
`mkdir {your dface traindata folder}`

- Generate PNet Train data and annotation file

```
python dface/prepare_data/gen_Pnet_train_data.py --prefix_path /nfs/home/hku_user01/DFace/training_data/WIDER/WIDER_train/images --dface_traindata_store data_/ --anno_file anno_store/anno.txt
```
- Assemble annotation file and shuffle it

`python dface/prepare_data/assemble_pnet_imglist.py`
- Train PNet model

`python dface/train_net/train_p_net.py`
- Generate RNet Train data and annotation file

```
python dface/prepare_data/gen_Rnet_train_data.py --prefix_path /nfs/home/hku_user01/DFace/training_data/WIDER/WIDER_train/images --dface_traindata_store data_/ --pmodel_file /nfs/home/hku_user01/DFace/model_store/pnet_epoch.pt --anno_file anno_store/anno.txt
```
- Assemble annotation file and shuffle it

`python dface/prepare_data/assemble_rnet_imglist.py`
- Train RNet model

`python dface/train_net/train_r_net.py`
- Generate ONet Train data and annotation file

```
python dface/prepare_data/gen_Onet_train_data.py --prefix_path training_data/WIDER/WIDER_train/images/ --dface_traindata_store data_/ --anno_file anno_store/anno.txt --pmodel_file model_store/pnet_epoch.pt --rmodel_file model_store/rnet_epoch.pt
```
- Generate ONet Train landmarks data

```
python dface/prepare_data/gen_landmark_48.py --dface_traindata_store data_ --anno_file training_data/Nvnc/trainImageList.txt --prefix_path training_data/Nvnc
```
- Assemble annotation file and shuffle it

`python dface/prepare_data/assemble_onet_imglist.py`
- Train ONet model

`python dface/train_net/train_o_net.py`

#### Test
`python test_image.py`

#### Reference
[DFace](https://github.com/kuaikuaikim/DFace)

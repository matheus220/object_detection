## Utilisation de l'API de détection d'objet TensorFlow pour détecter de nouveaux objets
Étapes pour créer un modèle qui détecte un objet souhaité. Ce tutoriel est une version très simplifiée du tutoriel disponible à [TensorFlow Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html)

> Il est important de bien suivre les instructions d'installation du tutoriel officiel

### Identification d'objets dans des images

Identifier tous les objets souhaités dans les images du dataset et ajouter un label pour chaque objet à l'aide du logiciel [labelImg](https://pypi.org/project/labelImg/)

### Placer les images (et les fichiers xml) dans le dossier image

Toutes les images et leurs fichiers .xml associés doivent être divisés en deux groupes, le groupe d'entraînement et le groupe de test. Par exemple, 10% des images seront placées dans `/images/train` et les 90% restants dans `/images/test`.

> Les scripts python doivent être exécutés à partir du dossier **base_structure**

###  Conversion de **xml** à **csv**

```sh
python xml_to_csv.py
```

###  Conversion de **csv** à **record**

```sh
python generate_tfrecord.py \
	--csv_input=data/train_labels.csv \
	--output_path=data/train.record \
	--image_dir=images/train/
```

```sh
python generate_tfrecord.py \
	--csv_input=data/test_labels.csv \
	--output_path=data/test.record \
	--image_dir=images/test/
```

### Entraîner le modèle

```sh
python train.py --logtostderr \
	--train_dir=training/ \
	--pipeline_config_path=training/faster_rcnn_inception_v2_coco.config
```

#### Surveiller l’entrainement

```sh
tensorboard --logdir='training'
```
Accéder l'adresse **127.0.0.1:6006** à partir d'un navigateur

### Exporter le modèle

```sh
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/faster_rcnn_inception_v2_coco.config \
    --trained_checkpoint_prefix training/model.ckpt-200000 \
    --output_directory thumb_inference_graph
```

### Utiliser le modèle

> Les deux scripts suivants sont des exemples appliqués au modèle que fait la détection de pouce et donc ils doivent être exécutés à partir du dossier `thumbsup_detection`

#### Détection dans les images de test

```sh
python detection_images.py \
	--model_name=thumb_inference_graph \
	--label_map=thumb_detection.pbtxt \
	--num_classes=2
```

#### Détection dans des images d'une caméra

```sh
python detection_camera.py \
	--model_name=thumb_inference_graph \
	--label_map=thumb_detection.pbtxt \
	--num_classes=2 \
	--time_period=5
```

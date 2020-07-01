

# Semantic segmentation on the TACO Dataset



![alt_text](figs/dataset0)
![alt_text](figs/dataset1)
![alt_text](figs/dataset2)


by: 

* Vera Hoveling, V.T.Hoveling@student.tudelft.nl, 4591941

* Sayra Ranjha S.S.Ranjha@student.tudelft.nl, 4555449

* Maaike Visser, M.E.B.P.Visser@student.tudelft.nl, 4597265


# Introduction

Although Pixar’s animators did surely not intend to lay out a roadmap to dystopia, since the release of WALL-E in 2008, we have steadily kept moving toward the hellscape that was painted in the feature film: a world covered in trash. Projections are that by 2025, we will produce double the amount of garbage that was produced in 2012 [1]:

     (2015, 18 May). U-Net: Convolutional 

(citation from: What a waste: a global review of solid waste management by Hoornweg, Daniel and Bhada-Tata, Perinaz, 2012). 

However, it is not quite time to despair _yet_. Progress was also made on another prominent feature of WALL-E’s world: robots! A plethora of waste-collecting robots and machinery is being developed ([https://mymodernmet.com/trash-collecting-robot-chicago-river/](https://mymodernmet.com/trash-collecting-robot-chicago-river/), [https://www.kickstarter.com/projects/wildmile/trash-cleaning-robot-controlled-by-you](https://www.kickstarter.com/projects/wildmile/trash-cleaning-robot-controlled-by-you), the ocean cleanup, roombas) in attempts to engineer ourselves out of our self-inflicted mess.

With this blog post, we hope to make our own small contribution. We investigate the performance of three popular image segmentation models on the task of segmenting images of various kinds of litter. In particular, we look at whether data augmentation can help these models perform better, without necessitating a larger base dataset.

~~ We present our findings on the application of deep learning computer vision techniques for trash detection. This should enable future WALL-E’s to distinguish trash from treasure and detect litter in varying circumstances.~~

~~This document details our studies of semantic segmentation on a dataset of annotated trash images, the TACO Dataset. We have trained on the dataset with three different models: DeepLab, Mask-RCNN and Unet. We will detail the dataset and it’s intricacies and the training process and results of each model. We conclude with a comparison of the models and some recommendation for further development of the TACO dataset.~~

Code and pre-trained models are available for evaluation purposes. 


# TACO Dataset 

For this project, we use the TACO dataset. It is, in one word, trash. Literally: TACO stands for Trash Annotations in Context. It currently consists of 1500 photos of litter taken in diverse environments, with 4784 total annotations. Together these form, in the words of TACO’s creators, “an open image dataset of waste in the wild”. The dataset was set up without any funding, to enable AI to tackle waste-related environmental concerns. Think of drones surveying trash, anti-littering video surveillance, real-life WALL-Es roaming the streets.



<p id="gdcalert4" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image4.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert5">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image4.png "image_tooltip")


Figure [X]: An example of an image from TACO with the corresponding segmentation mask.

TACO’s annotations are divided in 60 categories, of which 28 super (top) categories. It is important to note that there is a large class imbalance, both among the normal and the super categories. As you can see in Figure [X], there are more than 800 annotations of plastic bags and wrappers, but around 0 of batteries. Large class imbalances pose a challenge for deep learning models, because these models need enough examples of a class to learn a useful representation.



<p id="gdcalert5" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image5.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert6">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image5.png "image_tooltip")


<p id="gdcalert6" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image6.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert7">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image6.png "image_tooltip")


Figure [X]: Number of annotations per (super) class in TACO. Image taken from [http://tacodataset.org/stats](http://tacodataset.org/stats).

~~TACO’s authors note themselve note that “TACO is quite imbalanced but so is the open set.”~~[^1]~~. The “open set” here refers to all the trash present in the real world. The implication is that the real distribution is also not balanced and as such there is no conclude from that that there is no need to impose a fixed number of examples per class. It remains, however, a challenge when training, as one can read in the section on U-Net.~~


## Pre-processing

TACO’s annotations are in the COCO format. However, none of the models we chose to investigate work with COCO images out-of-the-box. Therefore, some preprocessing steps were needed. We:



*   Reduced the resolution of the images to [insert size] to achieve feasible training times.
*   Generated segmentation masks. In the COCO format, masks are given as a string of coordinates. However, our models take in an image tensor.
*   Set class labels for “undefined” areas. Sometimes, an area in a mask can belong to multiple classes at the same time. This can lead to strange behavior, so we mark such areas with the label ‘255’, which is ignored at training time.


# DeepLab

DeepLab is a state-of-the-art convolutional neural network for semantic image segmentation that makes use of atrous (or dilated) convolution to increase the field-of-view of filters without increasing the computational cost or number of parameters **[CITE DEEPLAB]**. 

Specifically, we used a PyTorch implementation of the DeepLabV3 model with a ResNet-50 backbone. Initially, the ResNet-101 backbone was used but due to memory constraints we had to switch to ResNet-50. The PyTorch model is pre-trained on a subset of COCO train2017, using the categories present in the Pascal VOC dataset **[CITE PYTORCH]**. 

The model is fine-tuned on the TACO dataset using SGD with an learning rate of 0.001 and a learning rate policy of 0.001 * (1 - iterations / max iterations)^0.9 as described in **[CITE DEEPLAB]**. A batch size of 2 was used as opposed to the batch size of 16 used in **[CITE DEEPLAB]** due to memory constraints.


### Training

First, DeepLabV3 was fine-tuned on the TACO dataset as-is with no further data augmentation. While the training and validation losses were quite low, the performance on the individual classes left a lot to be desired. The model performs extremely well on the background (IoU of 0.98), and achieved an IoU > 0.5 for the bottle, can, carton, and cup classes, which happen to be the most common classes. The remaining classes had a rather poor performance, with most not even being predicted at all. 

As such, we decided to give random cropping a try. However, as you can see in the learning curve, the validation loss is all over the place. The results on the test set were similarly disappointing, with the mean IoU down to 0.09 and the IoU of each class being significantly worse than before. 

In this case, the validation samples were also randomly cropped. Thinking that this caused the irregular validation loss, we fine-tuned DeepLab again using random cropping, but now leaving the validation images intact. While the validation loss is now lower than before, it is still irregular and not decreasing. The performance on the test set is now slightly better, and it traded the ability to poorly recognize paper for the ability to poorly recognize bottle caps instead. 

Desperate for improvement, we sought out other methods to counter the class imbalance of the TACO dataset. We tried a weighted cross-entropy loss using the inverse class frequencies as weights. This time, the validation loss curve was less irregular than before, but it was still unwilling to decrease. Although the performance on the test set is again worse than that of the first model, it is better than the random crop approaches. It also recognizes four more classes than the original model (bottle cap, lid, other plastic, and straw), where the IoU for bottle cap and other plastic is a respectable 0.14 and 0.16 respectively. While the IoU for other classes deteriorated, the IoU for plastic container is at an all-time high of 0.50.  

 

<p id="gdcalert7" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image7.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert8">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image7.png "image_tooltip")




<p id="gdcalert8" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image8.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert9">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image8.png "image_tooltip")




<p id="gdcalert9" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image9.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert10">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image9.png "image_tooltip")


<p id="gdcalert10" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image10.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert11">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image10.png "image_tooltip")



# Mask-RCNN

[Mask R-CNN](https://arxiv.org/abs/1703.06870) is a convolutional network used for [object instance segmentation](https://towardsdatascience.com/instance-segmentation-using-mask-r-cnn-7f77bdd46abd). It is built on top of [Faster R-CNN](https://arxiv.org/abs/1506.01497). Faster R-CNN generates so-called Regions of Interest (ROIs) and searches for objects within these regions. For each recognized object, Faster R-CNN generates a bounding box. Mask R-CNN expands upon this approach by predicting an object mask as well [cite Mask R-CNN].


## Implementation

We use [Torchvision’s Mask R-CNN implementation](https://github.com/pytorch/vision/blob/master/torchvision/models/detection/mask_rcnn.py) (t-Mask R-CNN). This implementation comes with a number of handy utility methods for training the model, and allows us to use Pytorch functionalities such as custom [DataLoaders](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). Additionally, t-Mask R-CNN comes with a pre-trained ResNet50 backbone. The backbone extracts the feature map on which the ROIs are based. This particular backbone is pre-trained on the [COCO 2017](https://cocodataset.org/) dataset. 


## Training

During preprocessing we already generated segmentation masks from the TACO annotations. However, t-Mask R-CNN requires expects more. In the following, H is the height of the image, W is the width of the image, and M is the number of masks.



*   image: a torch tensor of size (H, W)
*   target: a dict containing the following fields:
    *   boxes: the coordinates of the M bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
    *   labels: the label for each bounding box
    *   image_id: an image identifier
    *   area: the area of each bounding box
    *   iscrowd: instances with iscrowd=True will be ignored during evaluation. iscrowd=False for each image in the TACO dataset
    *   masks: the segmentation mask for each object

boxes, labels, and area are all calculated from the masks, iscrowd is always false, and image_id is given.

While it seems straightforward, training t-Mask R-CNN at first was not an easy feat. The loss of the network would invariably blow up to infinity, without clear cause. Inspecting images did not show a clear cause: there was no visual difference between images that were used in the network right before the loss blew up and images that were used earlier.

After a number of frustrating week, the problem turned out to be threefold:



1. Some images contain masks that are either zero width, height, or both. Since bounding boxes are computed from the masks, this leads to degenerate bounding boxes, which then causes the loss to explode.
2. Masks with label 255 (undefined) were not filtered out, but since there were supposed to be only 29 classes (28 superclasses plus background), when the model received a mask with label 255 it did not know what to do.
3. Some masks contained only the background class, likely because they originally contained very small masks that were lost during the resolution reduction. t-Mask R-CNN was not able to deal with these images.

At first, our solution was to filter out degenerate boxes and masks with label 255 at load time. However, at times this led to case three: images with no masks whatsoever. Therefore, we resorted to computing all the valid TACO images at once, and passing only the valid images to a DataLoader.


## Experiments


### Evaluation metric


### Untrained

As a baseline, it is interesting to see how t-Mask R-CNN performs without any fine-tuning.

[image]


```
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
```


The results are really bad.


### 30 epochs

Next, we train 

Does not give back losses.


```
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.030
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.051
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.031
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.011
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.075
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.049
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.117
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.154
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.155
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.079
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.301
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.140
```



### Augmented 30 epochs


# U-net 


### About U-net

U-Net is a convolutional neural network that was developed for the classification of images in biomedical tasks and designed to work with fewer training images while yielding more precise segmentations. The architecture proved to generalize very well and is also successfully applied in a wide variety of tasks, such as pixel-wise regression, learning 3D segmentations and image segmentation on imagenet. We thus had good hopes for U-net to perform well on the TACO dataset.

For this experiment, we adapted a pytorch U-net implementation that was made for the reproduction project of the course CS4240. The original report and code for that project can be found on [reproducedpapers.org](https://reproducedpapers.org/papers/HCCpp9BNEnUl0z6moLmg#rztA9goB4I5S1YB8Ah4up). The U-net implementation required a few minor adaptations in order to handle the TACO dataset but is otherwise completely the same. 


### Training

Training U-Net for the TACO dataset turned out to be a bumpy road to nowhere. After some initial training rounds, using SGD with a learning rate of 0.0001 and momentum of 0.99, the output seemed to converge at a consistent ‘all is background-class’ prediction for each and every pixel. The learning curve also showed consistent convergence of the validation loss around a loss of about 0.3: 



<p id="gdcalert11" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image11.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert12">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image11.png "image_tooltip")


Unsure if perhaps something had gone wrong in adapting the implementation we tried to overfit the network on a single image, which did work. The network was completely able to reproduce a single output, as can be seen in the image below. The top image is the original label, the bottom is the network prediction after overfitting of a dataset of only this image (due to the unpadded convolutions, the output is cropped to some extent).



<p id="gdcalert12" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image12.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert13">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image12.png "image_tooltip")
 

At first we tried to remedy this issue using different optimizers during training, such as Adadelta, but to no avail. Tweaking the learning rate and the learning rate scheduler did not change the all-background predictions into anything more meaningful either. 

At this point, we started to suspect that the large class imbalance in the dataset, combined with the large variety within classes, was hindering the learning. For example, the largest superclass, ‘Plastic bag & wrapper’, contains garbage bags, potato chips bags, transparent plastic bags and small pieces of wrapper. These are all very different yet belong to the same class, of which not too many samples were present anyway: even though it is the largest superclass, it contains a little over 400 images. In addition, in a lot of images, the annotated parts are very small:



<p id="gdcalert13" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image13.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert14">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image13.png "image_tooltip")


<p id="gdcalert14" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image14.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert15">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image14.png "image_tooltip")


After inspecting such images, one might see why the network would find a comfortable local minimum when predicting “all background” for each sample. We started to suspect that U-Net, with its large number of parameters was a bit too much of a beast for the delicately composed and rather small TADO dataset. We therefore augmented the dataset, using random crops and random rotations. Unfortunately, this did not help either. The validation loss continued to settle for roughly a 0.3 loss and upon inspection, all predictions on the test set would still be “all background”. Learning with data augmentation on a single class did not bring any relief either. We therefore have to conclude that the U-Net architecture, although robust, generalizable and designed for smaller datasets, is an insufficient architecture to capture the intricacies of this small dataset with large variety.


### Conclusion

After eliminating various reasons for the consistent convergence at local minima, we conclude that U-net is not fit for the task of trash detection with the TACO dataset. This is possibly due to its large number of parameters in combination with the small but versatile dataset.


# Discussion & Conclusion

From our training we conclude that for the task of trash detection with the TACO dataset, Mask RCNN is most successful. Unet, despite it’s attractive features, was not able to learn from the dataset.


# Recommendations

From our experiences with the TACO dataset we formulate the following recommendations:


### Add more samples to the dataset

The dataset in its current form is rather small, which is, in essence, relatively easy to remedy: add more samples. However, due to the labor-intensive annotation system, we understand that this is more easier said than done. 


### Balance classes

In order to more efficiently improve the dataset than just brute-forcing in more images, we would recommend adding new annotated samples to the underrepresented classes first. For example, the battery superclass now has 2 samples, which would make that class benefit much more from additional samples than the largest superclass of 400 samples


# Future Work


### Additional models

There are many more network architectures that could possibly be suitable for learning trash detection with the TACO dataset. We have only tried three, with mixed results, so further studies would be interesting. We’d recommend the following models for further investigation: YoloV4, Shift-invariant CNN, SSD and RetinaNet.


### Object tracking 

In order to enable a real WALL-E, one would ideally extend the segmentation to object tracking. It would be an interesting direction for future work to investigate Deep SORT for this purpose, which has proven to be very fast and can achieve frame rates up to 16 fps!

Deeplab


<table>
  <tr>
   <td>
   </td>
   <td>Mean IoU
   </td>
   <td>Frequency weighted IoU
   </td>
   <td>Mean Accuracy
   </td>
   <td>Pixel Accuracy
   </td>
  </tr>
  <tr>
   <td>Normal
   </td>
   <td><strong>0.20387399306541715</strong>
   </td>
   <td><strong>0.9651476099846004</strong>
   </td>
   <td><strong>0.2515695525111232</strong>
   </td>
   <td><strong>0.980938604472134</strong>
   </td>
  </tr>
  <tr>
   <td>Random crop
   </td>
   <td>0.09274474882048157
   </td>
   <td>0.9393082019789044
   </td>
   <td>0.11035002293630625
   </td>
   <td>0.9667554270737567
   </td>
  </tr>
  <tr>
   <td>Random crop (no crop on val)
   </td>
   <td>0.1463793855147662
   </td>
   <td>0.9525940638522274
   </td>
   <td>0.18911111717251386
   </td>
   <td>0.9730163166143371
   </td>
  </tr>
  <tr>
   <td>Weighted loss
   </td>
   <td>0.15872279081703652
   </td>
   <td>0.9527966536230024
   </td>
   <td>0.23470755680019448
   </td>
   <td>0.9691538754546601
   </td>
  </tr>
</table>


IoU per class:


<table>
  <tr>
   <td>
   </td>
   <td>Normal
   </td>
   <td>Random crop
   </td>
   <td>Random crop (no crop on val)
   </td>
   <td>Weighted loss
   </td>
  </tr>
  <tr>
   <td><strong>Background</strong>
   </td>
   <td>0.98497
   </td>
   <td>0.971271
   </td>
   <td>0.979263
   </td>
   <td>0.980206
   </td>
  </tr>
  <tr>
   <td><strong>Aluminium foil</strong>
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Battery
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Blister pack
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td><strong>Bottle</strong>
   </td>
   <td>0.735247
   </td>
   <td>0.395908
   </td>
   <td>0.516769
   </td>
   <td>0.421527
   </td>
  </tr>
  <tr>
   <td><strong>Bottle cap</strong>
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0.027382
   </td>
   <td>0.146449
   </td>
  </tr>
  <tr>
   <td><strong>Broken glass</strong>
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td><strong>Can</strong>
   </td>
   <td>0.555532
   </td>
   <td>0.080671
   </td>
   <td>0.438712
   </td>
   <td>0.319014
   </td>
  </tr>
  <tr>
   <td><strong>Carton</strong>
   </td>
   <td>0.56346
   </td>
   <td>0.126214
   </td>
   <td>0.332544
   </td>
   <td>0.293334
   </td>
  </tr>
  <tr>
   <td><strong>Cup</strong>
   </td>
   <td>0.253962
   </td>
   <td>0.041346
   </td>
   <td>0.037928
   </td>
   <td>0.092904
   </td>
  </tr>
  <tr>
   <td>Food waste
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Glass jar
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td><strong>Lid</strong>
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0.075601
   </td>
  </tr>
  <tr>
   <td><strong>Other plastic</strong>
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0.16305
   </td>
  </tr>
  <tr>
   <td><strong>Paper</strong>
   </td>
   <td>0.087936
   </td>
   <td>0.047733
   </td>
   <td>0
   </td>
   <td>0.017705
   </td>
  </tr>
  <tr>
   <td><strong>Paper bag</strong>
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td><strong>Plastic bag & wrapper</strong>
   </td>
   <td>0.484086
   </td>
   <td>0.122013
   </td>
   <td>0.279493
   </td>
   <td>0.20816
   </td>
  </tr>
  <tr>
   <td><strong>Plastic container</strong>
   </td>
   <td>0.284191
   </td>
   <td>0.0000661310055219389
   </td>
   <td>0.174681315815212
   </td>
   <td>0.503665423817259
   </td>
  </tr>
  <tr>
   <td><strong>Plastic gloves</strong>
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td><strong>Plastic utensils</strong>
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Pop tab
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td><strong>Rope & strings</strong>
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Scrap metal
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Shoe
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td>Squeezable tube
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td><strong>Straw</strong>
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0.081429
   </td>
  </tr>
  <tr>
   <td><strong>Styrofoam piece</strong>
   </td>
   <td>0.331968
   </td>
   <td>0.162418
   </td>
   <td>0.287194
   </td>
   <td>0.030135
   </td>
  </tr>
  <tr>
   <td><strong>Unlabeled litter</strong>
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
  <tr>
   <td><strong>Cigarette</strong>
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
   <td>0
   </td>
  </tr>
</table>



<!-- Footnotes themselves at the bottom. -->
## Notes

[^1]:
     [https://www.reddit.com/r/MachineLearning/comments/cjekzr/p_taco_trash_annotations_in_context_dataset/](https://www.reddit.com/r/MachineLearning/comments/cjekzr/p_taco_trash_annotations_in_context_dataset/)

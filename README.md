# CarND-Vehicle-Detection

The goal of the project is to write a software pipeline to detect vehicles in a video using sliding window classification approach.
Udacity provides two sets of 64x64 images for binary classification:
* [8792 images of vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip)
* [9666 images of non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip)

Final vehicle detection pipeline should correctly track vehicles on the road and eliminate false positive detections, which can be dangerous for a real Self-Driving Car for several reasons.

Here I describe most valuable pats of the code according to Udacity's project rubric:
* Histogram of Oriented Gradients (HOG) usage
* Sliding Window Search implementation
* Video application

## Histogram of Oriented Gradients (HOG) usage

The heart of the pipeline is Histogram of Oriented Gradients or HOG, which is implemented in scikit image package as [`skimage.feature.hog`](http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog).

Here is a wrapper code snippet used in pipeline:
```
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```

Essential parameters here are `orient`, `pix_per_cell` and `cell_per_block` which have to be defined to work well for this task.

Another tricky thing is `img` -- this must be 1-channel input and this can be either grayscaled image or a particular color channel from a color space.

To handle this parameter search I created another snippet of code which gets whole input dataset of raw images, converts it into particular color space and uses one of desired channels.

```
def get_hog_data(X, colorspace, orient, pixpercell, cellperblock, featuresize):
    Xhog = np.zeros((X.shape[0], featuresize), dtype=np.float32)
    
    for idx in range(X.shape[0]):
        if colorspace == 'gray':
            x = cv2.cvtColor(X[idx], cv2.COLOR_RGB2GRAY)
        else:
            hls = cv2.cvtColor(X[idx], cv2.COLOR_RGB2HLS)
            x = hls[:,:,['h', 'l', 's'].index(colorspace)]
        
        f = get_hog_features(x, orient, pixpercell, cellperblock)
        Xhog[idx] = f
        
    return Xhog
```

To actually pick HOG params, I used grid search for tihs parameters together with different classifier.
This allowed me to answer two questions at once:
* what classifier works best for this task
* what HOG params to use

With next evaluation snippet of code I performed several examinations.
```
def examine_hog_params_and_classifiers(colorspace, orient, pixpercell, cellperblock):
    # get vector size, doesnt matter which color space to use
    sample_feature = get_hog_features(cv2.cvtColor(sample_image, cv2.COLOR_RGB2GRAY), orient, pixpercell, cellperblock)
    vector_size = sample_feature.shape[0]
    
    # transform raw pixel data into hog features
    X_train = get_hog_data(X_TRAIN, colorspace, orient, pixpercell, cellperblock, vector_size)
    X_test =  get_hog_data(X_TEST,  colorspace, orient, pixpercell, cellperblock, vector_size)

    # get scaler from training data
    X_scaler = StandardScaler().fit(X_train)

    # transform all X data with scaler
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)
    
    print('colorspace {}, orient {}, pixpercell {}, cellperblock {}'.format(colorspace, orient, pixpercell, cellperblock))
    for c in classifiers:
        clf = c()
        clf.fit(X_train, Y_TRAIN)
        print(c, clf.score(X_test, Y_TEST))
```

Search space included next values to examine:
* `classifiers`: LinearSVC, DecisionTreeClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
* `colorspace`: ['gray', 'h', 'l', 's']  -- e.g. grayscale and channels of HLS color space
* `orient`: [6, 8, 10, 12]
* `pixpercell`: [4, 6, 8, 10, 12, 14, 16, 18, 20]
* `cellperblock`: [1, 2, 3, 4]

After grid search experiments I finished with next results:
* L-channel gives best results for HOG features, noteworthy is grayscale is also good
* GradientBoostingClassifier works best among other classifiers
* orient value of 10 gives best results
* more pixpercell produced more accuracy; best results I got with 16
* cells per block value of 2 gives best results

Below is an example of images with chosen HOG parameters (`HOG_ORIENT = 10`, `HOG_PPC = 16`, `HOG_CPB = 2`)
![Vehicles](readme_files/hog-vehicle.png)
![Non-vehicles](readme_files/hog-nonvehicle.png)

Well actually this looks not very descriptive for a human eye, but works very well :)

Key points:
* HOG parameters such color space, orientation, and others were choosen by grid search to get highest classification results
* GradientBoostingClassifier was choosen in same way
* [`sklearn.preprocessing.StandardScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) used to normalize data

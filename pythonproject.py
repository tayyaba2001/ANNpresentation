
    "import tensorflow as tf\n",
    "from tensorflow.keras.layers import Conv2D, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense\n",
    "from tensorflow.keras.models import Model, load_model\n",
    "from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import f1_score\n",
    "from sklearn.utils import shuffle\n",
    "import cv2\n",
    "import imutils\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import time\n",
    "from os import listdir\n",
    "\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def crop_brain_contour(image, plot=False):\n",
    "    \n",
    "    #import imutils\n",
    "    #import cv2\n",
    "    #from matplotlib import pyplot as plt\n",
    "    \n",
    "    # Convert the image to grayscale, and blur it slightly\n",
    "    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\n",
    "    gray = cv2.GaussianBlur(gray, (5, 5), 0)\n",
    "\n",
    "\n",
    "    # Threshold the image, then perform a series of erosions +\n",
    "    # dilations to remove any small regions of noise\n",
    "    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]\n",
    "    thresh = cv2.erode(thresh, None, iterations=2)\n",
    "    thresh = cv2.dilate(thresh, None, iterations=2)\n",
    "\n",
    "    # Find contours in thresholded image, then grab the largest one\n",
    "    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
    "    cnts = imutils.grab_contours(cnts)\n",
    "    c = max(cnts, key=cv2.contourArea)\n",
    "   # Find the extreme points\n",
    "    extLeft = tuple(c[c[:, :, 0].argmin()][0])\n",
    "    extRight = tuple(c[c[:, :, 0].argmax()][0])\n",
    "    extTop = tuple(c[c[:, :, 1].argmin()][0])\n",
    "    extBot = tuple(c[c[:, :, 1].argmax()][0])\n",
    "    \n",
    "    # crop new image out of the original image using the four extreme points (left, right, top, bottom)\n",
    "    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            \n",
    "\n",
    "    if plot:\n",
    "        plt.figure()\n",
    "\n",
    "        plt.subplot(1, 2, 1)\n",
    "        plt.imshow(image)\n",
    "        \n",
    "        plt.tick_params(axis='both', which='both', \n",
    "                        top=False, bottom=False, left=False, right=False,\n",
    "                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)\n",
    "        \n",
    "        plt.title('Original Image')\n",
    "            \n",
    "        plt.subplot(1, 2, 2)\n",
    "        plt.imshow(new_image)\n",
    "\n",
    "        plt.tick_params(axis='both', which='both', \n",
    "                        top=False, bottom=False, left=False, right=False,\n",
    "                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)\n",
    "\n",
    "        plt.title('Cropped Image')\n",
    "        \n",
    "        plt.show()\n",
    "    \n",
    "    return new_image"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_data(dir_list, image_size):\n",
    "    \"\"\"\n",
    "    Read images, resize and normalize them. \n",
    "    Arguments:\n",
    "        dir_list: list of strings representing file directories.\n",
    "    Returns:\n",
    "        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)\n",
    "        y: A numpy array with shape = (#_examples, 1)\n",
    "    \"\"\"\n",
    "\n",
    "    # load all images in a directory\n",
    "    X = []\n",
    "    y = []\n",
    "    image_width, image_height = image_size\n",
    "    \n",
    "    for directory in dir_list:\n",
    "        for filename in listdir(directory):\n",
    "            # load the image\n",
    "           #   CHANGE UNDER STATEMENT\n",
    "            image = cv2.imread(directory + '/' + filename)\n",
    "            # crop the brain and ignore the unnecessary rest part of the image\n",
    "            image = crop_brain_contour(image, plot=False)\n",
    "            # resize image\n",
    "            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)\n",
    "            # normalize values\n",
    "            image = image / 255.\n",
    "            # convert image to numpy array and append it to X\n",
    "            X.append(image)\n",
    "            # append a value of 1 to the target array if the image\n",
    "            # is in the folder named 'yes', otherwise append 0.\n",
    "            if directory[-3:] == 'yes':\n",
    "                y.append([1])\n",
    "            else:\n",
    "                y.append([0])\n",
    "\n",
    "    X = np.array(X)\n",
    "    y = np.array(y)\n",
    "    \n",
    "    # Shuffle the data\n",
    "    X, y = shuffle(X, y)\n",
    "    \n",
    "    print(f'Number of examples is: {len(X)}')\n",
    "    print(f'X shape is: {X.shape}')\n",
    "    print(f'y shape is: {y.shape}')\n",
    "    \n",
    "    return X, y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "from os import listdir\n",
    "from os.path import join, isfile\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "augmented_no = 'C:/Users/Student/Documents/ANN/b435/b4356/augmented data/no'\n",
    "augmented_yes = 'C:/Users/Student/Documents/ANN/b435/b4356/augmented data/yes'\n",
    "\n",
    "IMG_WIDTH, IMG_HEIGHT = (240, 240)\n",
    "\n",
    "X, y = load_data([augmented_yes, augmented_no], (IMG_WIDTH, IMG_HEIGHT))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_sample_images(X, y, n=50):\n",
    "    \"\"\"\n",
    "    Plots n sample images for both values of y (labels).\n",
    "    Arguments:\n",
    "        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)\n",
    "        y: A numpy array with shape = (#_examples, 1)\n",
    "    \"\"\"\n",
    "    \n",
    "    for label in [0,1]:\n",
    "        # grab the first n images with the corresponding y values equal to label\n",
    "        images = X[np.argwhere(y == label)]\n",
    "        n_images = images[:n]\n",
    "        \n",
    "        columns_n = 10\n",
    "        rows_n = int(n/ columns_n)\n",
    "\n",
    "        plt.figure(figsize=(20, 10))\n",
    "        i = 1 # current plot        \n",
    "        for image in n_images:\n",
    "            plt.subplot(rows_n, columns_n, i)\n",
    "            plt.imshow(image[0])\n",
    "            \n",
    "            # remove ticks\n",
    "            plt.tick_params(axis='both', which='both', \n",
    "                            top=False, bottom=False, left=False, right=False,\n",
    "                           labelbottom=False, labeltop=False, labelleft=False, labelright=False)\n",
    "            \n",
    "            i += 1\n",
    "        \n",
    "        label_to_str = lambda label: \"Yes\" if label == 1 else \"No\"\n",
    "        plt.suptitle(f\"Brain Tumor: {label_to_str(label)}\")\n",
    "        plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_sample_images(X, y)  \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ex_img = cv2.imread(\"C:/Users/Student/Documents/ANN/b435/b4356/yes/Y1.jpg\")\n",
    "ex_new_img = crop_brain_contour(ex_img, True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def split_data(X, y, test_size=0.2):\n",
    "       \n",
    "    \"\"\"\n",
    "    Splits data into training, development and test sets.\n",
    "    Arguments:\n",
    "        X: A numpy array with shape = (#_examples, image_width, image_height, #_channels)\n",
    "        y: A numpy array with shape = (#_examples, 1)\n",
    "    Returns:\n",
    "        X_train: A numpy array with shape = (#_train_examples, image_width, image_height, #_channels)\n",
    "        y_train: A numpy array with shape = (#_train_examples, 1)\n",
    "        X_val: A numpy array with shape = (#_val_examples, image_width, image_height, #_channels)\n",
    "        y_val: A numpy array with shape = (#_val_examples, 1)\n",
    "        X_test: A numpy array with shape = (#_test_examples, image_width, image_height, #_channels)\n",
    "        y_test: A numpy array with shape = (#_test_examples, 1)\n",
    "    \"\"\"\n",
    "    \n",
    "    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size)\n",
    "    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5)\n",
    "    \n",
    "    return X_train, y_train, X_val, y_val, X_test, y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print (\"number of training examples = \" + str(X_train.shape[0]))\n",
    "print (\"number of development examples = \" + str(X_val.shape[0]))\n",
    "print (\"number of test examples = \" + str(X_test.shape[0]))\n",
    "print (\"X_train shape: \" + str(X_train.shape))\n",
    "print (\"Y_train shape: \" + str(y_train.shape))\n",
    "print (\"X_val (dev) shape: \" + str(X_val.shape))\n",
    "print (\"Y_val (dev) shape: \" + str(y_val.shape))\n",
    "print (\"X_test shape: \" + str(X_test.shape))\n",
    "print (\"Y_test shape: \" + str(y_test.shape))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Nicely formatted time string\n",
    "def hms_string(sec_elapsed):\n",
    "    h = int(sec_elapsed / (60 * 60))\n",
    "    m = int((sec_elapsed % (60 * 60)) / 60)\n",
    "    s = sec_elapsed % 60\n",
    "    return f\"{h}:{m}:{round(s,1)}\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_f1_score(y_true, prob):\n",
    "    # convert the vector of probabilities to a target vector\n",
    "    y_pred = np.where(prob > 0.5, 1, 0)\n",
    "    \n",
    "    score = f1_score(y_true, y_pred)\n",
    "    \n",
    "    return score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_model(input_shape):\n",
    "    \"\"\"\n",
    "    Arugments:\n",
    "        input_shape: A tuple representing the shape of the input of the model. shape=(image_width, image_height, #_channels)\n",
    "    Returns:\n",
    "        model: A Model object.\n",
    "    \"\"\"\n",
    "    # Define the input placeholder as a tensor with shape input_shape. \n",
    "    X_input = Input(input_shape) # shape=(?, 240, 240, 3)\n",
    "    \n",
    "    # Zero-Padding: pads the border of X_input with zeroes\n",
    "    X = ZeroPadding2D((2, 2))(X_input) # shape=(?, 244, 244, 3)\n",
    "    \n",
    "    # CONV -> BN -> RELU Block applied to X\n",
    "    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)\n",
    "    X = BatchNormalization(axis = 3, name = 'bn0')(X)\n",
    "    X = Activation('relu')(X) # shape=(?, 238, 238, 32)\n",
    "    \n",
    "    # MAXPOOL\n",
    "    X = MaxPooling2D((4, 4), name='max_pool0')(X) # shape=(?, 59, 59, 32) \n",
    "    \n",
    "    # MAXPOOL\n",
    "    X = MaxPooling2D((4, 4), name='max_pool1')(X) # shape=(?, 14, 14, 32)\n",
    "    \n",
    "    # FLATTEN X \n",
    "    X = Flatten()(X) # shape=(?, 6272)\n",
    "    # FULLYCONNECTED\n",
    "    X = Dense(1, activation='sigmoid', name='fc')(X) # shape=(?, 1)\n",
    "    \n",
    "    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.\n",
    "    model = Model(inputs = X_input, outputs = X, name='BrainDetectionModel')\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "IMG_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = build_model(IMG_SHAPE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# tensorboard\n",
    "log_file_name = f'brain_tumor_detection_cnn_{int(time.time())}'\n",
    "tensorboard = TensorBoard(log_dir=f'logs/{log_file_name}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# checkpoint\n",
    "# unique file name that will include the epoch and the validation (development) accuracy\n",
    "filepath=\"cnn-parameters-improvement-{epoch:02d}-{val_accuracy:.2f}\"\n",
    "# save the model with the best validation (development) accuracy till now\n",
    "checkpoint = ModelCheckpoint(\"models/{}.model\".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "start_time = time.time()\n",
    "\n",
    "model.fit(x=X_train, y=y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])\n",
    "\n",
    "end_time = time.time()\n",
    "execution_time = (end_time - start_time)\n",
    "print(f\"Elapsed time: {hms_string(execution_time)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "start_time = time.time()\n",
    "\n",
    "model.fit(x=X_train, y=y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])\n",
    "\n",
    "end_time = time.time()\n",
    "execution_time = (end_time - start_time)\n",
    "print(f\"Elapsed time: {hms_string(execution_time)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "start_time = time.time()\n",
    "\n",
    "model.fit(x=X_train, y=y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])\n",
    "\n",
    "end_time = time.time()\n",
    "execution_time = (end_time - start_time)\n",
    "print(f\"Elapsed time: {hms_string(execution_time)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import absl.logging\n",
    "absl.logging.set_verbosity(absl.logging.ERROR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "start_time = time.time()\n",
    "\n",
    "model.fit(x=X_train, y=y_train, batch_size=32, epochs=3, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])\n",
    "\n",
    "end_time = time.time()\n",
    "execution_time = (end_time - start_time)\n",
    "print(f\"Elapsed time: {hms_string(execution_time)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "start_time = time.time()\n",
    "\n",
    "model.fit(x=X_train, y=y_train, batch_size=32, epochs=5, validation_data=(X_val, y_val), callbacks=[tensorboard, checkpoint])\n",
    "\n",
    "end_time = time.time()\n",
    "execution_time = (end_time - start_time)\n",
    "print(f\"Elapsed time: {hms_string(execution_time)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "history = model.history.history"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for key in history.keys():\n",
    "    print(key)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_metrics(history):\n",
    "    \n",
    "    train_loss = history['loss']\n",
    "    val_loss = history['val_loss']\n",
    "    train_acc = history['accuracy']\n",
    "    val_acc = history['val_accuracy']\n",
    "    \n",
    "    # Loss\n",
    "    plt.figure()\n",
    "    plt.plot(train_loss, label='Training Loss')\n",
    "    plt.plot(val_loss, label='Validation Loss')\n",
    "    plt.title('Loss')\n",
    "    plt.legend()\n",
    "    plt.show()\n",
    "    \n",
    "    # Accuracy\n",
    "    plt.figure()\n",
    "    plt.plot(train_acc, label='Training Accuracy')\n",
    "    plt.plot(val_acc, label='Validation Accuracy')\n",
    "    plt.title('Accuracy')\n",
    "    plt.legend()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_metrics(history) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model = load_model(filepath=\"C:/Users/Student/Documents/ANN/b435/b4356/models/cnn-parameters-improvement-23-0.91.model\" )\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model.metrics_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "loss, acc = best_model.evaluate(x=X_test, y=y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print (f\"Test Loss = {loss}\")\n",
    "print (f\"Test Accuracy = {acc}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_test_prob = best_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "f1score = compute_f1_score(y_test, y_test_prob)\n",
    "print(f\"F1 score: {f1score}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_val_prob = best_model.predict(X_val)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "f1score_val = compute_f1_score(y_val, y_val_prob)\n",
    "print(f\"F1 score: {f1score_val}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def data_percentage(y):\n",
    "    \n",
    "    m=len(y)\n",
    "    n_positive = np.sum(y)\n",
    "    n_negative = m - n_positive\n",
    "    \n",
    "    pos_prec = (n_positive* 100.0)/ m\n",
    "    neg_prec = (n_negative* 100.0)/ m\n",
    "    \n",
    "    print(f\"Number of examples: {m}\")\n",
    "    print(f\"Percentage of positive examples: {pos_prec}%, number of pos examples: {n_positive}\") \n",
    "    print(f\"Percentage of negative examples: {neg_prec}%, number of neg examples: {n_negative}\") "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# the whole data\n",
    "data_percentage(y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Training Data:\")\n",
    "data_percentage(y_train)\n",
    "print(\"Validation Data:\")\n",
    "data_percentage(y_val)\n",
    "print(\"Testing Data:\")\n",
    "data_percentage(y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}

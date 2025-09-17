1.Project Overview:
    ->Face Mask Detection App is an application that detects whether a person is wearing a mask or not.
      Face Mask Detection App built using TensorFlow, OpenCV, and Streamlit.
      It trains a deep learning model with MobileNetV2 on annotated face mask datasets. The app allows users to upload an image, 
      detects faces using OpenCV’s Haar cascade, and predicts mask presence in real-time. It provides a simple, 
      interactive UI for testing mask compliance.

2.Folder Structure:
    FACEMASK:-
        |-  app.py (Streamlit app)
        |-  mask_detector.py (Script to train model)
        |-  model(mask_detector.h5) -(Trained model)
        |-  readme.txt (Project description)
        |-  requirements.txt (Python packages required)
        \--dataset  (data)
                |- annotations (XML annotation files)
                |- images   (Images corresponding to annotations)


3.Libraries & Environment Setup:
  a.Libraries -
    -TensorFlow
    -NumPy
    -OpenCV
    -Pandas & Scikit-learn
    -Streamlit
    -Pillow
    -opencv-python
    -Matplotlib & Seaborn
    -Scikit-learn


  b.Setup Instructions: 
     pip install -r requirements.txt or pip install streamlit==1.22.0 opencv-python==4.9.0.80 Pillow==10.0.0 numpy==1.24.3 tensorflow scikit-learn matplotlib seaborn

     Note:- If the dataset is not available in project folder, you can download from this link https://drive.google.com/drive/folders/1qwDwo4iZUdrG7ewz4mPNixke-81PwFsg?usp=share_link
     



4.How to Run the Code:
   
    a. Web Interface: 
     -> If model (mask_detector.h5) is not available in the folder structure, then follow these steps:
         -pip install -r requirements.txt or pip install streamlit==1.22.0 opencv-python==4.9.0.80 Pillow==10.0.0 numpy==1.24.3 tensorflow scikit-learn matplotlib seaborn
         -python mask_detector.py
         -streamlit run app.py  
           
     -> If model (mask_detector.h5) is available in the folder structure, then follow these steps:
        -streamlit run app.py  

        - This will open a browser window with a user-friendly interface to upload image and see whether a person is wearing a mask or not.

      b. Example:
        -pip install -r requirements.txt
         Or manually install:
         pip install streamlit==1.22.0 opencv-python==4.9.0.80 Pillow==10.0.0 numpy==1.24.3 tensorflow scikit-learn matplotlib seaborn

        - python model.py
        - streamlit run app.py  

    c. File need configured before running
        - mask_detector.h5

5.Input and Output:
   a. Input:
        Upload an image(.png,.jpg,.jpeg)  

    b. Output:
           The output of this project is an image with detected faces highlighted by bounding boxes.
           For each face, the system predicts whether the person is with_mask or without_mask,
           displaying the label above the bounding box.
           Green box → Face with a mask 
           Blue box → Face without a mask

    
6. How to Test with a New Dataset

    Upload new dataset:-
       - Add your new data in datasetm folder.

    Update dataset path in mask_detector.py:-
       - Change the BASE_DIR, IMG_DIR, and ANNOT_DIR variables to point to your new dataset folder.

    Train the model:-
       -Open a terminal in your project folder and run:
       -python mask_detector.py
        This will train the model on your new dataset.
        After training completes, the model will be saved as mask_detector.h5.

    Run the Streamlit app to test:-
      -streamlit run app.py
    Upload images in the app to see mask detection results with bounding boxes and labels.

    
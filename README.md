# Install env for python
```python
  pip install -r requirements.txt
```

# Download model file detect face from GGDrive then move it into Models
  ```
  https://drive.google.com/file/d/1aBwUsMZE-FYbcaBiBtQAA0Xm1s1sZgAP/view?usp=sharing
  ```
# Create directory
  Dataset/FaceData/raw --> store image
  Dataset/FaceData/processed --> store info

# Init data
face:
```python
   python src/init_data_face.py
```
result:
```python
  python src/init_data_result.py
```

# Add data of employee into train_data
```python
  python src/make_data_train.py name'
```
#### name is name of new employee after save face_image into raw


# Run Face-recognition process with webcam
```python
  python src/face_rec_cam.py
```
#### Result of process was saved in 'Dataset/FaceData/processed/result.json'

# Remove employee
```python
  python src/delete_member.py name
```
#### name is name of employee

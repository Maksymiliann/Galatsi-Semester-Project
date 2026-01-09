# Galatsi-Semester-Project

## Project structure

<!-- TREE_START -->

```text
Galatsi-Semester-Project/
├── .github/
├── Dataset/
├── LightGlue/
├── Main code/
├── Results/
├── runs/
├── scripts/
├── Src/
├── .gitattributes
├── .gitignore
├── EPFL_Report_Semester_Project_2_compressed.pdf
├── logs.txt
├── README.md
├── requirements.txt
├── yolo11l-obb.pt
├── yolo11m-obb.pt
├── yolo11n-obb.pt
├── yolo11x-obb.pt
└── yolo11x.pt
```

<!-- TREE_END -->

1) tracking.py
  - takes a video or image as input as well as the desired YOLO model 
  - stabilizes the video
  - detects and track the different vehicles
  - assignes their state (parked, driving)
  - exports a txt file with the OBB position, state, class, confidence score and ID

2) parking_prediction_dwell_state_multiples4.py
   - takes txt files and first frame as input as well as some parameters
   - projects all into a reference frame
   - creates a parking heatmap
   - predicts parking location
   - extend the parking places in the lenght direction
   - exports a mask of the parking locations

3) post_precessing2.py
   - takes the mask as input
   - close the small gaps
   - 
4) cleaning.py
   - takes the mask as input and removes the small parking locations

5) zones3.py
   - takes a mask as input
   - creates zones based on the parking location using DBscan
    
6) zones.py
   - takes a mask with zones as input
   - creates an ID for each zone

(Optional) 
  Padding.py
  - takes the txt files as input
  - does a padding on parked vehicles to have a better analysis later

7) occupancy_analysis_per_zones3.py
   - takes the txt files as well as the mask and the ID mask, and a reference frame and the first frame on the analyzed video
   - analyse the occupancy of the different parking zones based on the txt files
   - the ref image is to project the analysed video into the ref frame
   - gets a few CSV files with data
  
8) csv_reader.py
   - takes the csv as input
   - read them and plot some graphs

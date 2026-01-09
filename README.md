# Galatsi-Semester-Project

## Project structure

<!-- TREE_START -->

```text
Galatsi-Semester-Project/
├── .github/
│   └── workflows/
├── Dataset/
│   ├── Galatsi_Data_Semester_Project_archive/
│   ├── Galatsi_Data_Semester_Project_processing/
│   └── Galatsi_Data_Semester_Project_archive.zip
├── LightGlue/
│   ├── .github/
│   ├── assets/
│   ├── lightglue/
│   ├── lightglue.egg-info/
│   ├── .flake8
│   ├── .gitattributes
│   ├── .gitignore
│   ├── benchmark.py
│   ├── demo.ipynb
│   ├── LICENSE
│   ├── pyproject.toml
│   ├── README.md
│   └── requirements.txt
├── Main code/
│   ├── 1_tracking.py
│   ├── 2_parking_prediction_dwell_state_multiples4.py
│   ├── 3_post_processing2.py
│   ├── 4_cleaning.py
│   ├── 4_post_processing3.py
│   ├── 5_zones3.py
│   ├── 6_1_optional_padding.py
│   ├── 6_zones.py
│   ├── 7_occupancy_analysis_per_zone3.py
│   ├── 8_csv_reader.py
│   └── EPFL_Report_Semester_Project_2_compressed.pdf
├── Results/
│   ├── feature_matching/
│   ├── Ground_truth/
│   ├── Images/
│   ├── occupancy_analysis/
│   ├── parking_detection/
│   ├── TXT/
│   ├── TXT10/
│   ├── TXT_0004/
│   ├── TXT_0004_padded/
│   ├── TXT_0005/
│   ├── TXT_0005_padded/
│   ├── TXT_0006/
│   ├── TXT_0006_padded/
│   ├── TXT_0312_D2_S3_S1/
│   ├── TXT_0312_D2_S3_S1_padded/
│   ├── TXT_0314_D2_S4_S1/
│   ├── TXT_0314_D2_S4_S1_padded/
│   ├── TXT_0319_D2_S5_S1/
│   ├── TXT_0319_D2_S5_S1_padded/
│   └── Video/
├── runs/
│   └── obb/
├── scripts/
│   └── update_readme_tree.py
├── Src/
│   ├── cleaning.py
│   ├── csv_reader.py
│   ├── data_analysis.ipynb
│   ├── data_time_processing.py
│   ├── feature_matching_image.py
│   ├── feature_matching_image_openCV.py
│   ├── feature_matching_image_openCV2.py
│   ├── feature_matching_lightglue.py
│   ├── feature_matching_lightglue2.py
│   ├── feature_matching_video.py
│   ├── mask_quality_analysis.py
│   ├── new_test_clean_density.py
│   ├── occupancy_analyis_per_zone2.py
│   ├── occupancy_analysis.py
│   ├── occupancy_analysis_fast.py
│   ├── occupancy_analysis_faster.py
│   ├── occupancy_analysis_mult.py
│   ├── occupancy_analysis_per_zone.py
│   ├── occupancy_analysis_per_zone3.py
│   ├── occupancy_analysis_very_fast.py
│   ├── occupancy_analysis_with_feature_tracking.py
│   ├── padding.py
│   ├── parking_detection_dwell_state_ratio.py
│   ├── parking_detection_gauss_splat.py
│   ├── parking_prediction_dwell_mix.py
│   ├── parking_prediction_dwell_state.py
│   ├── parking_prediction_dwell_state_clip.py
│   ├── parking_prediction_dwell_state_multiples.py
│   ├── parking_prediction_dwell_state_multiples2.py
│   ├── parking_prediction_dwell_state_multiples3.py
│   ├── parking_prediction_dwell_state_multiples4.py
│   ├── parking_prediction_dwell_test.py
│   ├── parking_prediction_GMM_EM.py
│   ├── parking_prediction_GMM_EM_fast.py
│   ├── patch_inference.ipynb
│   ├── patch_inference_model.py
│   ├── patch_inference_model_test_myself.py
│   ├── post_processing.py
│   ├── post_processing2.py
│   ├── post_processing3.py
│   ├── read_json.py
│   ├── see_id.py
│   ├── see_id2.py
│   ├── take_frame.py
│   ├── test.ipynb
│   ├── test_line.py
│   ├── test_parking.py
│   ├── test_patch_counting.py
│   ├── test_patch_not_counting.py
│   ├── test_txt_record.py
│   ├── tracking.py
│   ├── yolo11m-obb.pt
│   ├── yolo11m.pt
│   ├── yolo11n-obb.pt
│   ├── yolo11n.pt
│   ├── yolo11x-obb.pt
│   ├── yolo11x.pt
│   ├── zones.py
│   ├── zones2.py
│   ├── zones3.py
│   └── zones4.py
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

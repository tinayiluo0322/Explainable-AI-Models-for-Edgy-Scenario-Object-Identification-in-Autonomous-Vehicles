# Explainable-AI-Models-for-Edgy-Scenario-Object-Identification-in-Autonomous-Vehicles
## Luopeiwen Yi

# Overview

## Research Question  
The primary goal of this project is to evaluate the prediction accuracy and explainability of AI models for object identification in challenging or edge-case scenarios within the context of autonomous driving. Specifically, this research explores how well pre-trained image classification models—**ResNet50** and **VGG16**—perform under conditions like night driving, rain, snow, fog, and broken infrastructures. It also examines how explainability techniques, including **LIME**, **Grad-CAM**, and **Anchor Explanations**, clarify their predictions.  

In this study, 'object identification' refers to the classification of entire images representing scenes or objects, rather than detecting and localizing objects within the image.  

**Hypothesis**: Edge-case scenarios challenge both model performance and explainability, requiring improved methods to handle uncertainty in real-world conditions.

## Importance of the Topic  
Autonomous vehicles rely heavily on object identification systems for safe and effective operation. Poor visibility, adverse weather conditions, or damaged infrastructure can cause failures in these systems. As these conditions pose significant safety risks, this project bridges the gap between performance and transparency by improving the explainability of AI models in these challenging scenarios. The findings could inform safer, more reliable AI for autonomous vehicles.

## Prior Work in the Field  
While significant advances have been made in Explainable AI (XAI) methods such as LIME and Grad-CAM, most research focuses on ideal conditions. Anchor Explanations have been underexplored in object classification tasks but offer unique insights by identifying specific areas or conditions necessary to maintain a prediction. This project addresses a gap by evaluating models in edge-case scenarios, creating a novel dataset to assess explainability in real-world conditions.

## Methodology

1. **Data Collection**  
   - Collect 8 edge-case images from the internet, depicting scenarios like night driving, snow, fog, rain, and broken cars and road signs.  

2. **Modeling**  
   - Use pre-trained **ResNet50** and **VGG16** for image classification.  

3. **Explainability Evaluation**  
   - Apply **LIME**, **Grad-CAM**, and **Anchor Explanations** to understand why and how the models make decisions.  

4. **Visual Output Analysis**  
   - Visualize and interpret explainability results for edge-case images.  

## Data Sources  
- **Custom Dataset**: 8 images sourced online, representing edge-case scenarios (e.g., night driving, snow, fog, etc.). Located in [EdgyData](./EdgyData) directory.

## Instructions to Run the Project  

To reproduce the experiments and visual outputs, follow these steps:

### Clone the Repository  
```bash
git clone https://github.com/tinayiluo0322/Explainable-AI-Models-for-Edgy-Scenario-Object-Identification-in-Autonomous-Vehicles.git
cd Explainable-AI-Models-for-Edgy-Scenario-Object-Identification-in-Autonomous-Vehicles
```

### Google Colab Notebooks  
- Each experiment (e.g., **Car in Fog**, **Night Pedestrian**) is documented in separate **Google Colab notebooks** included in the repository.  
- Open the relevant notebook for the specific experiment you wish to run.  
- The notebooks are well-commented and include all necessary steps, from loading the dataset to generating explainability visualizations.  

### Dependencies  
- Ensure you have the required dependencies installed.  
  - Dependencies are listed in the Colab notebooks, and you can install them using the provided commands within the notebook.  
  - Alternatively, install them using a `requirements.txt` file.  

### Run Experiments  
1. **Dataset Upload**:  
   - The dataset is located in the `EdgyData` folder within the repository.  
   - Upload the dataset files to the Colab environment when prompted.  

2. **Execute Cells**:  
   - Follow the instructions in the notebook.  
   - Execute each cell sequentially to preprocess the data, run the model, and generate explainability outputs.  

## Unique Contributions  
This project’s uniqueness lies in its:  
1. **Edge-case dataset**: A novel dataset specifically tailored for evaluating object identification in autonomous vehicles.  
2. **Explainability focus**: Assessing widely used XAI methods (LIME, Grad-CAM, Anchor Explanations) in difficult conditions.  
3. **Benchmarking robustness**: Providing insights into model transparency under edge-case scenarios, offering a foundation for future research.  

## Final Deliverables  
1. **Manuscript**: Summarizing the methodology, visual outputs, and insights.  
2. **Video Presentation**: Explaining the project and key findings.  
3. **Colab Notebook**: A documented workflow for reproducibility, including data preprocessing, modeling, and explainability visualizations.

## Summary of XAI Methods for Object Identification  

| **XAI Method**       | **Explanation Mechanism**                                                                 | **Application on Object Identification**                                                                                      | **Visual Output**                                                                                                   |
|-----------------------|-------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| **LIME**             | Perturbs input data and observes changes in predictions to build local surrogate models.  | Highlights areas in an image (e.g., superpixels) critical to the model's decision.                                           | Overlay of superpixels showing important areas for object classification.                                          |
| **Grad-CAM**         | Uses gradients of the model's output with respect to the final convolutional layer.       | Identifies important pixel regions contributing to the model's classification.                                               | Heatmap superimposed on the original image, with warm colors indicating important regions.                         |
| **Anchor**            | Constructs rules (anchors) that specify conditions sufficient for a prediction.           | Identifies the specific parts of an image or conditions that ensure the model's classification remains consistent.           | Highlighted areas (anchors) showing conditions that strongly influenced the model's decision.                      |

# Comparative Analysis of ResNet50 and VGG16 with XAI Methods for Edgy Dataset

## Broken Road Sign Experiment

The broken road sign experiment demonstrates that both ResNet50 and VGG16 successfully identified the road sign in the image, showcasing their capability to accurately classify traffic signs. However, the effectiveness of the explanations provided by the different XAI methods—LIME, Grad-CAM, and Anchor—varied significantly in terms of their interpretability, precision, and coverage, offering valuable insights into the strengths and weaknesses of each approach.

LIME provided localized explanations by highlighting the road sign as the central feature influencing the top-1 prediction. While it effectively grayed out irrelevant areas such as the background building, its interpretations for the top-5 predictions were inconsistent and sometimes misleading. For instance, in alternative predictions like "parking meter" or "traffic light," LIME highlighted irrelevant areas such as the parking lot or scattered features of the image. This inconsistency indicates that while LIME is intuitive and model-agnostic, its reliance on perturbation and linear approximations limits its ability to capture complex model behaviors and nuanced decision boundaries.

Grad-CAM, on the other hand, consistently highlighted regions centered around the road sign for both models but failed to provide detailed or class-specific insights. For top-5 predictions, it showed little variation in the regions of interest, making it difficult to differentiate between classes or understand the unique features contributing to each prediction. While Grad-CAM is computationally efficient and useful for coarse visual explanations, its lack of granularity and focus on similar regions across predictions limits its interpretability.

Anchor stood out as the most precise and interpretable method in this experiment. For both ResNet50 and VGG16, it highlighted the road sign's shape and the distinct outline of the word "STOP" on the sign, clearly indicating that text was the primary factor influencing the "traffic sign" prediction. The anchor precision of 1.00 demonstrated the method's reliability, as the highlighted features were sufficient to guarantee the correct prediction. However, with an anchor coverage of 0.50, the explanation applied to only half of the dataset instances, suggesting that while Anchor excels in specific scenarios, its generalizability may require further refinement.

In summary, while all three XAI methods contributed unique perspectives to model interpretability, Anchor proved to be the most reliable and insightful for this experiment, offering precise and intuitive explanations tied directly to the model's decision-making process. LIME provided flexible but sometimes inconsistent interpretations, and Grad-CAM offered less differentiation and clarity across predictions. These findings underscore the importance of selecting XAI methods based on their strengths and weaknesses to ensure meaningful and actionable insights, particularly in edge-case scenarios like this broken road sign experiment.

### **ResNet50 Prediction: Traffic Sign** 

![ResNet50_LIME_RoadSign](https://github.com/user-attachments/assets/2e720529-d870-4c17-8067-c76b653b228f)

![ResNet50_Grad_Cam_RoadSign](https://github.com/user-attachments/assets/702d6669-f61d-46aa-974d-68cccb9180fd)

![ResNet50_Anchor_RoadSign](https://github.com/user-attachments/assets/904b281d-e8bb-46c1-b632-da80a5b4e9aa)

### **VGG16 Prediction: Traffic Sign**

![VGG16_LIME_RoadSign](https://github.com/user-attachments/assets/b7b9f171-f487-4a10-aeb6-9ab1581ecf58)

![VGG16_GradCam_RoadSign](https://github.com/user-attachments/assets/e9f2e26c-4336-435d-a492-e345bd32872e)

![VGG16_Anchor_RoadSign](https://github.com/user-attachments/assets/7e8fa9fe-f46b-4828-96e3-87235208e7b3)

## Collide Broken Car Experiment

The broken car experiment demonstrates the strengths and weaknesses of ResNet50 and VGG16 in accurately predicting and interpreting a complex image involving a damaged car. ResNet50 correctly identified the object as a "taxicab," while VGG16 labeled it as a "station wagon." Both predictions were reasonable, as they accurately categorized the object as a car, despite contextual distractions such as visible damage and the presence of a background bus.

**Model Prediction Accuracy:**  
Both ResNet50 and VGG16 successfully identified the car as the primary object in the image, showing their robustness in recognizing overall vehicle shapes and features.

**XAI Method Comparison:**

1. **LIME:**  
   LIME provided a granular understanding of the models' focus for both predictions. For ResNet50, it highlighted the car's top regions, such as the front and side windows, for the "taxicab" prediction, while alternative predictions like "limousine" and "minibus" shifted attention to side features or the background bus. Similarly, for VGG16, LIME emphasized the car's top features for "station wagon" but shifted focus to the bus for other predictions. While LIME offered detailed insights into localized feature importance, it exposed the models' tendency to inconsistently rely on irrelevant background elements, reducing its interpretability in complex scenes.

2. **Grad-CAM:**  
   Grad-CAM provided a broader, less specific focus for both models. For ResNet50, it highlighted the car for predictions like "taxicab" but shifted to the background bus for predictions like "minibus." Similarly, for VGG16, the focus varied across predictions but lacked clarity, making it difficult to pinpoint the exact features influencing the decisions. This generality limited Grad-CAM’s utility in explaining the models’ reasoning for this experiment.

3. **Anchor:**  
   Anchor offered the most precise and interpretable explanations. For ResNet50, it focused on the car's upper features, such as the windows and wipers, while also highlighting parts of the background bus, achieving a precision of 0.99. For VGG16, Anchor emphasized the car’s entire shape and some background bus features, with perfect precision (1.00) but zero coverage (0.00). While Anchor’s precision demonstrated its reliability, the low coverage revealed its instance-specific nature, limiting its generalizability across similar images.

**Comparison of Models:**  
ResNet50 and VGG16 showed comparable performance in predicting the car's class, leveraging general vehicle shapes and features. However, both models relied on background elements like the bus for alternative predictions, suggesting susceptibility to contextual distractions. This reliance was consistently reflected in their explanations, particularly in LIME and Anchor evaluations.

**Key Insights:**  
The broken car experiment highlights the importance of using multiple XAI methods to uncover different facets of model interpretability. While LIME offered localized feature importance, its inconsistency across predictions revealed the models' biases toward irrelevant features. Grad-CAM’s broad focus lacked specificity, making it less useful for detailed analysis. Anchor provided the most precise insights, directly linking features to predictions, but its low coverage emphasized the need for improved generalization. Overall, both models demonstrated strong object recognition capabilities but require enhanced contextual reasoning to handle complex scenes more effectively.

### **ResNet50 Prediction: Taxicab** 

![R_LIME_BrokenCAR](https://github.com/user-attachments/assets/62e4d43e-350c-4eef-90fc-786ed510a48e)

![R_GradCam_BrokenCar](https://github.com/user-attachments/assets/4fff01b9-5e18-4fa0-b957-662f4c16534d)

![R_Anchor_Broken_Car](https://github.com/user-attachments/assets/b50ff92f-3c74-4355-aabb-b3482d605ee4)

### **VGG16 Prediction: Station Wagon** 

![V_LIME_Broken_Car](https://github.com/user-attachments/assets/c61985d9-db90-4a00-ad87-3bb2e700c28f)

![V_GradCam_BrokenCar](https://github.com/user-attachments/assets/59096787-1ba1-4069-8cf7-f4f5484c334e)

![V_Anchor_Broken_Car](https://github.com/user-attachments/assets/ac5f25eb-8b32-48bf-8d84-3a312d0b6430)

## Car in Fog Experiment

The car in fog experiment demonstrates the challenges ResNet50 and VGG16 face in interpreting visually ambiguous scenes with minimal contextual information. ResNet50 predicted "fire engine," likely due to its focus on the car's bright red backlight, while VGG16 predicted "traffic light," mistaking the same feature for the lights typically associated with a traffic signal. Both models relied heavily on the brightest element in the image, failing to consider the overall structure of the car or the foggy environment, leading to misclassification.

**XAI Method Comparison:**

1. **LIME:**  
   LIME provided insights into the models' focus on the car’s red backlight across predictions. For ResNet50, LIME highlighted the backlight and parts of the car for predictions such as "fire engine" and "tow truck," while shifting attention to the background for "semi-trailer truck" and "traffic light." Similarly, VGG16's LIME interpretation mirrored this focus, demonstrating both models' tendency to rely on the most visually prominent feature while inconsistently addressing other parts of the image. This inconsistency highlights the models' struggle to integrate the scene's context into their decision-making.

2. **Grad-CAM:**  
   Grad-CAM offered limited interpretability for both models. For ResNet50, the focus was centered on the car and its immediate surroundings across all predictions, showing little variation and providing minimal differentiation between predictions. VGG16 displayed slightly more variation, with attention shifting across different parts of the car and the background. However, the focus remained scattered and non-intuitive, failing to explain the reasoning behind specific predictions effectively.

3. **Anchor:**  
   Anchor provided the most precise explanations but revealed the models' over-reliance on specific features. For ResNet50, Anchor highlighted the car’s backlight and a portion of the sky, achieving a high precision of 0.98 but with zero coverage, indicating that the explanation was highly instance-specific. For VGG16, Anchor similarly focused on the car’s backlight for the "traffic light" prediction, with a precision of 0.95 and a low coverage of 0.06. These results emphasize both models’ reliance on the most visually striking feature—the red backlight—while lacking broader generalization.

**Comparison of Models:**  
Both ResNet50 and VGG16 demonstrated similar weaknesses in handling the foggy scene, with their predictions driven primarily by the car's red backlight. ResNet50's focus resulted in a misclassification of "fire engine," while VGG16 associated the backlight with "traffic light," reflecting its bias toward identifying light sources. Neither model effectively incorporated the structural or contextual information needed to identify the car correctly.

**Key Insights:**  
This experiment underscores the limitations of ResNet50 and VGG16 in interpreting challenging environments like foggy, low-visibility scenes. Among the XAI methods, Anchor provided the clearest explanations but revealed the models' reliance on narrow, instance-specific reasoning. LIME highlighted inconsistent attention patterns, while Grad-CAM failed to provide meaningful insights due to its broad and scattered focus. To improve performance in such scenarios, future models should prioritize the integration of contextual reasoning and better handling of ambiguous visual cues to enhance both prediction accuracy and interpretability.

### **ResNet50 Prediction: Fire Engine** 

![R_LIME_CarFOG](https://github.com/user-attachments/assets/93b91c61-73a0-4707-b4f5-27ba0e464031)

![R_GradCam_CarFog](https://github.com/user-attachments/assets/58a4a3df-068c-46bf-8f60-949ce9dadd7c)

![R_Anchor_CarFog](https://github.com/user-attachments/assets/656e4d0b-feb4-4bee-a1ea-ae4b6590c2dc)

### **VGG16 Prediction: Traffic Light** 

![V_LIME_CarFog](https://github.com/user-attachments/assets/ff70f413-2587-46db-b58a-04029ee4ae24)

![V_GradCAM_CarFOG](https://github.com/user-attachments/assets/fb0c5e11-8981-4a41-a9f6-97932a094228)

![V_Anchor_CarFOG](https://github.com/user-attachments/assets/42be2a21-5142-4d7d-a482-d6529b4e07d2)

## Scooter in Fog Experiment

The scooter in fog experiment demonstrates that both ResNet50 and VGG16 were able to correctly classify the image of a pedestrian driving a motor scooter. This highlights the models' ability to recognize the concept of a scooter, even under challenging conditions such as fog. However, the interpretability provided by XAI methods revealed varying levels of clarity and utility in understanding the models' decision-making processes.

**XAI Method Comparison:**
- **LIME:** LIME struggled to provide clear insights for both models. The explanations focused on fragmented, small regions of the image, graying out the majority of the scene. While these regions were consistent across predictions like "scooter," "scuba diver," and "oxygen mask," it remained unclear how such similar areas could result in drastically different predictions. This inconsistency undermined the utility of LIME in explaining the models' reasoning.  
- **Grad-CAM:** Grad-CAM also faced interpretability challenges, with both models showing little variation in focus across predictions. The highlighted areas consistently centered on the pedestrian driving the scooter but failed to differentiate between features contributing to correct predictions like "scooter" and other potential predictions. This lack of variation limited Grad-CAM's ability to provide meaningful insights.  
- **Anchor:** Anchor offered the most intuitive and precise explanations for both models. It highlighted the shape of the pedestrian driving the scooter as the key feature influencing the correct "motor scooter" prediction. For ResNet50, the Anchor precision was 0.98, and for VGG16, it was 0.95, demonstrating high reliability. However, the anchor coverage of 0.00 for both models indicated that the explanations were highly instance-specific, limiting their generalizability.

**Key Insights:**  
This experiment underscores the strengths and limitations of ResNet50 and VGG16 in interpreting challenging scenes. While both models successfully recognized the "scooter," XAI methods revealed gaps in their reasoning and focus. Anchor provided the clearest and most intuitive explanations, though its lack of generalization highlights the need for models to better integrate context and environmental cues for broader applicability. Improving interpretability and consistency in reasoning across predictions remains a crucial area for future development in model design and XAI methods.

### **ResNet50 Prediction: Scooter** 

![R_LIME_Motor_Scooter](https://github.com/user-attachments/assets/32ad5077-aeb4-4e44-9f7c-ee239caddc47)

![R_GradCam_MotorScooter](https://github.com/user-attachments/assets/8270cffa-1e4a-4773-abf3-d90c816876cd)

![R_Anchor_Scooter](https://github.com/user-attachments/assets/4af83502-d11c-4506-849c-f80ea24f13c2)

### **VGG16 Prediction: Scooter** 

![V_LIME_Scooter](https://github.com/user-attachments/assets/08eba96a-f1e1-4ae9-881b-69535cf0037d)

![V_GradCam_Scooter](https://github.com/user-attachments/assets/4d8b8760-738e-4f3b-aa09-c7be252d57cf)

![V_Anchor_Scooter](https://github.com/user-attachments/assets/4460b9e8-11b3-423a-96d3-e352b1555812)

## Night Pedestrian and Car Experiment

The night pedestrian and car experiment highlights the challenges faced by ResNet50 and VGG16 in interpreting complex nighttime scenes dominated by extreme contrasts, such as bright car headlights against a dark background. ResNet50 predicted "spotlight," likely due to its focus on the car's headlights, while VGG16 predicted "snowplow," influenced by a combination of the car's lights and background shapes. Both models failed to recognize the pedestrian or the car as central elements of the scene, revealing their limitations in understanding context in low-light scenarios.

**Model Prediction Accuracy:**  
ResNet50 and VGG16 both misclassified the image, with ResNet50 associating the scene with a "spotlight" and VGG16 interpreting it as a "snowplow." These predictions reflect each model's reliance on specific visual cues—ResNet50 focused heavily on the bright headlights, while VGG16 was influenced by a composite of car lights and background shapes. Neither model demonstrated the ability to capture the broader context or the central pedestrian and car in the scene.

**XAI Method Comparison:**

1. **LIME:**  
   LIME revealed the fragmented focus of both models in this experiment. For ResNet50, the model's "spotlight" prediction emphasized the car's lights and other bright elements in the image, while alternative predictions like "fountain" and "torch" highlighted background lights. Similarly, for VGG16, the "snowplow" prediction ignored the central objects and focused on background features, while other predictions followed a pattern similar to ResNet50, emphasizing inconsistent and often irrelevant elements. LIME exposed the models' confusion and lack of cohesive attention across the scene.

2. **Grad-CAM:**  
   Grad-CAM provided limited clarity for both models, as it consistently focused on broad, vague areas of the image. For ResNet50, it centered on the car's headlights and surrounding areas across predictions without offering meaningful differentiation. VGG16's Grad-CAM explanations were similarly diffuse, focusing predominantly on background elements for multiple predictions like "snowplow," "parking meter," and "traffic light." This lack of specificity highlighted the weaknesses of Grad-CAM in providing actionable insights for complex scenes.

3. **Anchor:**  
   Anchor explanations were the most precise and interpretable, offering distinct insights into each model's decision-making process. For ResNet50, Anchor showed that the "spotlight" prediction was based almost entirely on the car's lights, achieving high precision (1.00) but with limited generalizability (coverage: 0.12). For VGG16, Anchor revealed that the "snowplow" prediction relied on a combination of the car's lights and various background shapes, forming a composite resembling a snowplow. While Anchor precision for VGG16 was also perfect (1.00), the coverage of 0.00 indicated the explanation applied solely to this specific instance, underscoring the model's reliance on narrow, instance-specific reasoning.

**Comparison of Models:**  
Both ResNet50 and VGG16 struggled to interpret the scene correctly, but their approaches differed. ResNet50 relied heavily on the car's lights, leading to a singular focus that drove its prediction of "spotlight." In contrast, VGG16 incorporated a broader range of visual elements, including background shapes, resulting in the misclassification of "snowplow." These differences highlight the models’ respective tendencies to overemphasize either dominant features or contextual elements without fully integrating the scene's key objects.

**Key Insights:**  
This experiment underscores the limitations of ResNet50 and VGG16 in understanding contextually rich but visually challenging environments, such as nighttime scenes with extreme contrasts. Among the XAI methods, Anchor provided the most reliable and precise explanations, revealing the models' reliance on specific features but also their lack of generalizability. LIME exposed the fragmented and inconsistent focus of both models, while Grad-CAM struggled to provide meaningful insights due to its broad, undifferentiated attention. To improve performance in such scenarios, future models should incorporate mechanisms to better integrate global and local features, ensuring a more holistic understanding of complex scenes.

### **ResNet50 Prediction: Spotlight** 

![R_LIME_Night_PC](https://github.com/user-attachments/assets/673a7471-9254-4534-8a04-8916c01acb61)

![R_GradCam_PC](https://github.com/user-attachments/assets/d091fd86-d19c-41b1-83df-4faeb072cf6c)

![R_Anchor_PC](https://github.com/user-attachments/assets/f545090d-8bbc-48ed-81c9-8b8f86e6ef62)

### **VGG16 Prediction: Snowplow** 

![V_LIME_PC](https://github.com/user-attachments/assets/dc3bc8d7-3aac-4f9f-b9f6-5fd4ee9f194c)

![V_GradCam_PC](https://github.com/user-attachments/assets/e11d010b-c031-424b-9366-8d4c24bec6b3)

![V_Anchor_PC](https://github.com/user-attachments/assets/a9ddd378-eb75-4b15-8c18-58053ea1c58c)

## Night Raining Pedestrian Experiment

The night raining pedestrian experiment highlights the challenges both ResNet50 and VGG16 face in correctly interpreting complex nighttime scenes with multiple overlapping elements, such as pedestrians with umbrellas, traffic lights, and other contextual features. Both models incorrectly predicted the label "carousel," likely due to the bright lights and patterns in the scene, which misled the models into associating the image with features of a carousel.

**Model Prediction Accuracy:**  
Both ResNet50 and VGG16 failed to correctly classify the image, showing similar tendencies to focus on visually striking but contextually irrelevant elements. Their shared prediction of "carousel" underscores their difficulty in accurately distinguishing key features in complex, low-light environments with multiple overlapping objects.

**XAI Method Comparison:**

1. **LIME:**  
   LIME revealed that both models were confused by the complexity of the image, focusing on random elements such as the traffic light, parts of the pedestrians, and umbrellas while graying out other areas. This inconsistent focus across predictions (e.g., "torch," "umbrella," "vase") demonstrated the models' struggle to identify which features were most relevant, reducing interpretability. While LIME provided localized feature importance, its scattered explanations highlighted the models' inability to process the image cohesively.

2. **Grad-CAM:**  
   Grad-CAM provided varying insights for the two models. For ResNet50, the attention was consistently focused on the center of the image across predictions, offering little differentiation and failing to clarify the reasoning behind specific predictions. In contrast, VGG16's Grad-CAM interpretation offered slightly more detail, with "carousel" focusing on the center of the image, including the pedestrians and traffic light, and "umbrella" shifting attention specifically to the umbrellas. This difference suggests that VGG16 exhibited a marginally better ability to localize relevant features, but Grad-CAM's overall explanatory power remained limited.

3. **Anchor:**  
   Anchor explanations provided the most precise insights for both models. For ResNet50, Anchor highlighted the shapes of the pedestrians with umbrellas and the traffic light as key features, achieving perfect precision (1.00) but no coverage (0.00), indicating that the explanation was highly instance-specific. Similarly, for VGG16, Anchor also highlighted the pedestrians with umbrellas and the traffic light, with slightly lower precision (0.98) and similarly low coverage (0.00). These results suggest that both models relied heavily on specific visual patterns in this instance but lacked generalizability across similar scenes.

**Comparison of Models:**  
ResNet50 and VGG16 displayed similar interpretability and prediction performance in this experiment, with both struggling to contextualize the scene accurately. While VGG16's Grad-CAM offered slightly more differentiation, the models' reliance on irrelevant or misleading elements, such as bright lights and umbrella shapes, was consistently revealed across all XAI methods.

**Key Insights:**  
This experiment underscores the limitations of ResNet50 and VGG16 in handling complex, multi-object nighttime scenes, where contextual reasoning is critical. Among the XAI methods, Anchor provided the most precise explanations, but its low coverage revealed a lack of generalization. LIME exposed the models' inconsistent focus, while Grad-CAM offered limited interpretability, with VGG16 performing slightly better than ResNet50. Future improvements should focus on enhancing models' ability to prioritize meaningful features and context in challenging visual environments.

### **ResNet50 Prediction: Carousel** 

![R_LIME_Night_Rain](https://github.com/user-attachments/assets/0a4cf627-590c-4eac-af62-dac13c178eb1)

![R_GradCam_Night_Rain](https://github.com/user-attachments/assets/1367c6e1-361a-4cfa-837a-4a500bc9fa2f)

![R_Anchor_Night_Rain](https://github.com/user-attachments/assets/df3593a3-8d6d-4eaf-8ecc-4e10dd052a15)

### **VGG16 Prediction: Carousel** 

![V_LIME_Night_Rain](https://github.com/user-attachments/assets/fdd21a85-c790-43d0-8103-c7e87dfdd002)

![V_GradCam_Night_Rain](https://github.com/user-attachments/assets/96cde9ee-97ba-4e22-9ac2-d366205f82f3)

![V_Anchor_Night_Rain](https://github.com/user-attachments/assets/aff34500-e20f-4ebd-965b-0463c58569b9)

## Raining Pedestrian Experiment

The raining pedestrian experiment highlights the challenges both ResNet50 and VGG16 face in accurately identifying the pedestrian amidst a complex scene. In this case, the pedestrian's head was completely covered by the umbrella, leaving only the clothing and the umbrella visible. ResNet50 incorrectly predicted "umbrella," focusing on visual patterns resembling an umbrella, while VGG16 misclassified the image as "kimono," likely influenced by the textures and patterns in the clothing and umbrella. These misclassifications underscore limitations in both models’ ability to contextualize and distinguish visually similar elements.

**LIME Evaluation:**  
LIME provided local interpretations that revealed both models’ focus on surface-level features. For ResNet50, LIME highlighted parts of the umbrella and human clothing for the top-1 prediction "umbrella" but showed inconsistent attention across other predictions, such as focusing on the background fence for "picket fence" or the sky for "pinwheel." Similarly, for VGG16, LIME highlighted clothing and umbrella patterns for "kimono," while focusing predominantly on clothing for alternative predictions like "abaya" and "suit." These results demonstrate LIME’s utility in visualizing decision patterns but also highlight its susceptibility to instability and limited ability to reflect broader contextual understanding.

**Grad-CAM Evaluation:**  
Grad-CAM showed a lack of interpretability for ResNet50, providing broad, uniform attention across the image without meaningful differentiation for the top-1 and other predictions. In contrast, Grad-CAM for VGG16 displayed some variation, focusing on both the umbrella and clothing for "kimono" and primarily on clothing for other predictions. While this demonstrates slightly better contextual sensitivity, Grad-CAM still failed to provide actionable insights into the models’ decision-making processes, particularly for complex scenarios.

**Anchor Evaluation:**  
Anchor explanations offered the most precise and interpretable insights for both models. For ResNet50, Anchor highlighted the umbrella patterns and a small portion of the clothing, showing high precision (0.98) but limited generalizability with coverage (0.25). For VGG16, Anchor emphasized parts of the umbrella and most of the clothing, indicating the model treated these features as a combined entity. While the precision (0.96) was similarly high, the coverage (0.00) revealed that the explanation applied to an extremely narrow instance set, underscoring a critical limitation in its broader applicability.

**Key Observations**:
The complete occlusion of the pedestrian’s head by the umbrella was a critical factor influencing both models’ predictions. This absence of a key human feature left the models relying solely on the umbrella and clothing, leading to errors. ResNet50 focused heavily on the umbrella, while VGG16 prioritized the clothing and treated it as part of a larger "kimono-like" structure. These misinterpretations underscore the limitations of current models in handling occlusions and complex visual contexts.

**Overall Insights**:
ResNet50 and VGG16 demonstrated comparable weaknesses in their predictions, relying heavily on specific visual cues without adequately contextualizing the broader scene. Among the XAI methods, Anchor provided the most precise and interpretable insights, directly linking specific features to predictions but with limited coverage. LIME offered flexible visualizations but struggled with consistency and relevance across predictions. Grad-CAM, while efficient, lacked meaningful focus and differentiation, contributing the least to understanding the models' decision processes.

### **ResNet50 Prediction: Umbrella** 

![R_LIME_Unbrella_Bro](https://github.com/user-attachments/assets/2be6ae3c-a4e5-4f10-8e58-6d764f058a20)

![R_GradCam_Unbrella_Bro](https://github.com/user-attachments/assets/b591402c-1523-4b3d-842a-7b92348368ed)

![R_Anchor_Unbrella_Bro](https://github.com/user-attachments/assets/fb07f78e-7da3-45c5-b7fe-14c48feb492e)

### **ResNet50 Prediction: Kimono** 

![V_LIME_Unbrella_Bro](https://github.com/user-attachments/assets/94db8199-c72a-40d6-a11f-5062a255e4a8)

![V_GradCam_Unbrella_bro](https://github.com/user-attachments/assets/d36d5c5d-b015-4837-a4da-82d74f606616)

![V_Anchor_unbrella_bro](https://github.com/user-attachments/assets/e5999d15-0e25-4729-b47e-fedd229d85ee)

## Car in Snow Experiment

The car in snow experiment highlights the challenges ResNet50 and VGG16 face in accurately interpreting snow-covered scenes, where object shapes and surrounding snow create visual ambiguities. ResNet50 predicted "bobsleigh," likely due to its focus on the car's snow-covered front, which resembles a bobsleigh, while VGG16 predicted "military aircraft," associating the car's elongated shape and uncovered features with the silhouette of a warplane. Both models struggled to recognize the car in its snow-obscured state, instead relying on shape-based reasoning that led to misclassification.

**XAI Method Comparison:**  
- **LIME:** LIME revealed that both models emphasized shape-based features to make predictions. For ResNet50, LIME highlighted the car's front for "bobsleigh" and shifted attention to the surrounding snow and other features for predictions like "snowplow" and "station wagon." Similarly, for VGG16, LIME focused on the car's uncovered parts and made predictions such as "snowmobile," "bobsleigh," and "tank," demonstrating confusion in the scene’s interpretation.  
- **Grad-CAM:** Grad-CAM provided limited insights for both models. For ResNet50, it focused broadly on the car and snow without significant differentiation between predictions. For VGG16, Grad-CAM highlighted different parts of the car but lacked depth in explaining the reasoning behind specific predictions like "military aircraft," reducing its overall interpretability.  
- **Anchor:** Anchor offered the most precise explanations for both models. For ResNet50, it highlighted the car, wall, and surrounding snow as contributing to the "bobsleigh" prediction, achieving high precision (0.98) but with zero generalizability (coverage: 0.00). For VGG16, Anchor emphasized the car’s shape and iron sheet-like features for the "military aircraft" prediction, with similar reliability (precision: 0.99) but limited coverage (0.06). These explanations underscored each model's reliance on specific shape-based elements rather than broader contextual reasoning.

**Key Insights:**  
This experiment highlights the reliance of both models on shape-based reasoning in snow-obscured scenes, leading to significant misclassification. Among the XAI methods, Anchor provided the most precise explanations, while LIME exposed the fragmented focus of the models, and Grad-CAM offered minimal differentiation. To improve performance in such scenarios, future models must incorporate better contextual reasoning and environmental awareness, enabling them to distinguish objects in complex, snow-covered environments more accurately.

### **ResNet50 Prediction: Bobsleigh** 

![R_LIME_SnowCar](https://github.com/user-attachments/assets/90afc4f6-d460-40ec-9674-1d894b40434c)

![R_GradCam_SnowCar](https://github.com/user-attachments/assets/b1fd6601-4f03-4cfa-8c74-1dfbc75f3ef4)

![R_Anchor_SnowCar](https://github.com/user-attachments/assets/7419492f-9263-4a5a-8d62-2a5f6430bc77)

### **VGG16 Prediction: Military Aircraft** 

![V_LIME_SnowCar](https://github.com/user-attachments/assets/4e734bc4-b8a0-4fde-9243-10b79f11d3b0)

![V_GradCam_SnowCar](https://github.com/user-attachments/assets/1fd8f0b5-5c85-4e10-b559-eb24e2c5c507)

![V_Anchor_SnowCar](https://github.com/user-attachments/assets/1e510a86-1c9f-4db8-af0b-651d397bc376)

# Conclusion

The results of this study highlight the capabilities and limitations of ResNet50 and VGG16 in identifying objects in edge-case scenarios, with varying performance influenced by environmental conditions and object visibility. 

**Model Performance:**  
Both models performed well in scenarios where the object's general shape and key features were intact, such as broken road signs, broken cars, and motor scooters in fog. In these cases, the models relied on essential features like text for road signs or the overall shape and key details of cars and scooters. However, performance significantly degraded in scenarios where objects were obscured or distorted (e.g., pedestrians in rain with umbrellas covering the head, cars partially covered in snow) or captured under challenging conditions like nighttime, fog, or a cluttered scene with multiple overlapping objects. These scenarios revealed the models’ inability to accurately interpret visual contexts, resulting in significant misclassification.

**XAI Method Analysis:**  
The evaluation of LIME, Grad-CAM, and Anchor Explanations exposed clear differences in their interpretability and reliability across the experiments:

### XAI Method Performance Summary  

| **Aspect**         | **LIME**                                                                                                     | **Grad-CAM**                                                                                     | **Anchor**                                                                                                    |
|--------------------|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **Interpretability** | Local explanations are intuitive but often inconsistent and fragmented, focusing on irrelevant areas.       | Broad focus lacks specificity and provides minimal differentiation between predictions.          | Precise and clear explanations, directly linking features to predictions.                                    |
| **Strengths**       | Works well in identifying local feature importance; model-agnostic and flexible across domains.             | Efficient and directly tied to CNN feature maps; visually intuitive in highlighting key areas.   | Highly reliable in identifying critical features; intuitive if-then rules for specific predictions.          |
| **Weaknesses**      | Struggles with nonlinear decision boundaries; inconsistent and oversimplified explanations.                 | Limited to coarse visualizations; fails to provide actionable insights in complex scenarios.      | Coverage is often extremely low, meaning generalizability across instances is limited.                       |
| **Best Use Cases**   | Useful for analyzing feature importance in relatively simple and well-defined scenarios.                   | Effective for initial visualization of attention in CNNs but lacks depth for nuanced analysis.    | Ideal for understanding critical features driving specific predictions; excels in edge-case scenarios.       |

**Key Insights:**  

1. **LIME**:  
   LIME provided flexible and localized visualizations, but its interpretations were often inconsistent, focusing on irrelevant areas or failing to explain how similar highlighted features led to drastically different predictions. These limitations were particularly evident in scenarios with multiple overlapping objects or poorly visible scenes, reducing LIME's reliability in edge-case scenarios.

2. **Grad-CAM**:  
   Grad-CAM consistently highlighted broad regions of the image but lacked differentiation and clarity. For example, in foggy or dark scenarios, it focused on the car and its surrounding areas but failed to explain how specific features contributed to distinct predictions. While efficient, Grad-CAM proved the least useful in providing actionable insights for edge-case analyses.

3. **Anchor**:  
   Anchor Explanations excelled in providing precise and interpretable insights, directly linking specific features, such as the shape of a scooter or the text on a road sign, to the models’ predictions. However, its low coverage indicated that these explanations were often highly instance-specific, limiting their ability to generalize across similar images. Despite this limitation, Anchor was the most effective method for understanding model decision-making in challenging scenarios.

### Conclusion Table  

| **Scenario**                     | **Model**  | **Prediction**            | **LIME Evaluation**                                                                                   | **Grad-CAM Evaluation**                                                                        | **Anchor Evaluation**                                                                                                 |
|----------------------------------|------------|---------------------------|-------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| **Broken Road Sign**              | ResNet50   | Traffic Sign              | Highlighted text on the road sign but inconsistently focused on irrelevant areas for other predictions.| Highlighted the road sign uniformly without class-specific insights.                           | Precisely focused on the road sign’s text and shape; high precision (1.00) but limited coverage (0.50).                |
|                                  | VGG16      | Traffic Sign              | Similar to ResNet50; inconsistent focus on road sign and irrelevant background.                        | Similar broad focus; minimal differentiation between predictions.                              | Clear focus on the road sign; high precision (1.00) but low coverage (0.50).                                         |
| **Broken Car**                    | ResNet50   | Taxicab                   | Focused on the car’s top and shifted attention to the background for alternative predictions.          | Broad focus on car and bus; little clarity on feature differentiation.                         | Highlighted car features and parts of the bus; high precision (0.99), no coverage (0.00).                              |
|                                  | VGG16      | Station Wagon             | Focused on similar features as ResNet50; inconsistent for background elements.                        | Scattered focus on the car and bus; lacked meaningful variation between predictions.            | Clear focus on the car and bus; high precision (1.00), no coverage (0.00).                                           |
| **Scooter in Fog**                | ResNet50   | Scooter                   | Focused on fragmented parts of the scooter but struggled with consistent explanations across predictions.| Centered on the scooter but lacked variation in focus.                                          | Clear focus on the scooter’s shape and pedestrian; high precision (0.98), no coverage (0.00).                          |
|                                  | VGG16      | Scooter                   | Similar fragmented focus to ResNet50; inconsistently highlighted scooter features.                    | Focused on the scooter and pedestrian but failed to clarify the reasoning behind predictions.   | Highlighted scooter and pedestrian with clear rationale; high precision (0.95), no coverage (0.00).                    |
| **Night Pedestrian and Car**      | ResNet50   | Spotlight                 | Focused on fragmented bright elements like the headlights and background lights; lacked cohesion.      | Broad focus on the headlights and car; minimal differentiation between predictions.             | Precisely highlighted the car’s headlights but with limited generalization. Precision (1.00), coverage (0.12).         |
|                                  | VGG16      | Snowplow                  | Similar fragmented focus to ResNet50; emphasized irrelevant background areas.                         | Focused primarily on the car and background without meaningful variation.                       | Clear focus on headlights and background but limited to the specific instance. Precision (1.00), coverage (0.00).      |
| **Car in Fog**                    | ResNet50   | Fire Engine               | Highlighted the car's backlight but inconsistently shifted to irrelevant areas for other predictions.  | Consistently focused on the car and immediate surroundings with little variation.               | Highlighted car backlight and sky; high precision (0.98), no coverage (0.00).                                         |
|                                  | VGG16      | Traffic Light             | Mirrored ResNet50 by focusing on the car’s backlight, struggling to differentiate between predictions. | Scattered focus on car and background lights with minimal interpretability.                     | Highlighted car backlight and parts of the background; high precision (0.95), low coverage (0.06).                     |
| **Night Raining Pedestrian**      | ResNet50   | Carousel                  | Focused on scattered elements like umbrellas and traffic lights; inconsistent and fragmented.          | Uniform focus on the center of the image, offering little differentiation.                      | Highlighted umbrellas and traffic lights with precision (1.00), no coverage (0.00).                                   |
|                                  | VGG16      | Carousel                  | Similar fragmented focus to ResNet50, emphasizing irrelevant bright elements.                        | Focused on pedestrians and umbrellas but lacked clear reasoning.                                | Highlighted umbrellas and pedestrians; high precision (0.98), no coverage (0.00).                                     |
| **Raining Pedestrian**            | ResNet50   | Umbrella                  | Highlighted parts of the umbrella and clothing but inconsistently shifted attention to irrelevant features.| Broad and uniform focus on the umbrella; lacked meaningful differentiation between predictions.  | Highlighted umbrella patterns and a portion of clothing; high precision (0.98), low coverage (0.25).                   |
|                                  | VGG16      | Kimono                    | Similar to ResNet50; focused on clothing and umbrella but with limited context.                       | Slightly better focus on clothing and umbrella, but still lacked actionable insights.            | Highlighted clothing and umbrella as a single entity; high precision (0.96), no coverage (0.00).                       |
| **Car in Snow**                   | ResNet50   | Bobsleigh                 | Focused on the car’s front, resembling a bobsleigh, but shifted attention to snow for other predictions.| Broad focus on car and snow; lacked clarity in feature differentiation.                         | Highlighted car and surrounding snow; high precision (0.98), no coverage (0.00).                                       |
|                                  | VGG16      | Military Aircraft         | Focused on uncovered parts of the car; misinterpreted them as military aircraft features.              | Similar broad focus on car’s shape and surrounding snow.                                        | Highlighted car shape and uncovered features; high precision (0.99), low coverage (0.06).                              |

### Final Insights  

ResNet50 and VGG16 performed similarly across edge-case scenarios, demonstrating reasonable classification capabilities in clear conditions but significant challenges under occlusions, poor visibility, and overlapping objects. Among the XAI methods, Anchor consistently provided the most precise and interpretable explanations, though its limited coverage underscored its instance-specific nature. LIME offered flexible, localized visualizations but struggled with consistency in complex scenarios. Grad-CAM, while efficient, lacked clarity and differentiation, making it the least effective for edge-case interpretability. Future research should prioritize developing models and XAI methods that better integrate contextual reasoning and enhance generalization to ensure robust performance in real-world conditions.

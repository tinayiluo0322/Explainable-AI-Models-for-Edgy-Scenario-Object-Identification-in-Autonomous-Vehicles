# Explainable-AI-Models-for-Edgy-Scenario-Object-Identification-in-Autonomous-Vehicles
# Overview

---

## 1. Research Question  
The primary goal of this project is to evaluate the explainability of AI models for object identification in challenging or edge-case scenarios within the context of autonomous driving. Specifically, this research explores how well pre-trained image classification models—**ResNet50** and **VGG16**—perform under conditions like night driving, rain, snow, fog, and broken traffic signs. It also examines how explainability techniques, including **LIME**, **Grad-CAM**, and **Anchor Explanations**, clarify their predictions.  

In this study, 'object identification' refers to the classification of entire images representing scenes or objects, rather than detecting and localizing objects within the image.  

**Hypothesis**: Edge-case scenarios challenge both model performance and explainability, requiring improved methods to handle uncertainty in real-world conditions.

---

## 2. Importance of the Topic  
Autonomous vehicles rely heavily on object identification systems for safe and effective operation. Poor visibility, adverse weather conditions, or damaged infrastructure can cause failures in these systems. As these conditions pose significant safety risks, this project bridges the gap between performance and transparency by improving the explainability of AI models in these challenging scenarios. The findings could inform safer, more reliable AI for autonomous vehicles.

---

## 3. Prior Work in the Field  
While significant advances have been made in Explainable AI (XAI) methods such as LIME and Grad-CAM, most research focuses on ideal conditions. Anchor Explanations have been underexplored in object classification tasks but offer unique insights by identifying specific areas or conditions necessary to maintain a prediction. This project addresses a gap by evaluating models in edge-case scenarios, creating a novel dataset to assess explainability in real-world conditions.

---

## 4. General Approach  

1. **Data Collection**  
   - Collect 8 edge-case images from the internet, depicting scenarios like night driving, snow, fog, rain, and broken cars and road signs.  

2. **Modeling**  
   - Use pre-trained **ResNet50** and **VGG16** for image classification.  

3. **Explainability Evaluation**  
   - Apply **LIME**, **Grad-CAM**, and **Anchor Explanations** to understand why and how the models make decisions.  

4. **Visual Output Analysis**  
   - Visualize and interpret explainability results for edge-case images.  

---

## 5. Data Sources  
- **Custom Dataset**: 8 images sourced online, representing edge-case scenarios (e.g., night driving, snow, fog, etc.).

---

## 6. Unique Contributions  
This project’s uniqueness lies in its:  
1. **Edge-case dataset**: A novel dataset specifically tailored for evaluating object identification in autonomous vehicles.  
2. **Explainability focus**: Assessing widely used XAI methods (LIME, Grad-CAM, Anchor Explanations) in difficult conditions.  
3. **Benchmarking robustness**: Providing insights into model transparency under edge-case scenarios, offering a foundation for future research.  

---

## 7. Final Deliverables  
1. **Slides**: Summarizing the methodology, visual outputs, and insights.  
2. **Video Presentation**: Explaining the project and key findings.  
3. **Colab Notebook**: A documented workflow for reproducibility, including data preprocessing, modeling, and explainability visualizations.


---

## Summary of XAI Methods for Object Identification  

| **XAI Method**       | **Explanation Mechanism**                                                                 | **Application on Object Identification**                                                                                      | **Visual Output**                                                                                                   |
|-----------------------|-------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| **LIME**             | Perturbs input data and observes changes in predictions to build local surrogate models.  | Highlights areas in an image (e.g., superpixels) critical to the model's decision.                                           | Overlay of superpixels showing important areas for object classification.                                          |
| **Grad-CAM**         | Uses gradients of the model's output with respect to the final convolutional layer.       | Identifies important pixel regions contributing to the model's classification.                                               | Heatmap superimposed on the original image, with warm colors indicating important regions.                         |
| **Anchor**            | Constructs rules (anchors) that specify conditions sufficient for a prediction.           | Identifies the specific parts of an image or conditions that ensure the model's classification remains consistent.           | Highlighted areas (anchors) showing conditions that strongly influenced the model's decision.                      |

### LIME

LIME is a technique that explains the predictions of any black-box machine learning model by approximating its behavior locally (near the instance of interest) using a simple interpretable model like linear regression or decision trees. The goal is to explain why a model made a particular prediction for a specific data point, rather than understanding the global behavior of the model. LIME approximates the decision function around the instance of interest using a linear model or some other simple interpretable model.

#### Summary of Pros and Cons for LIME

| **Aspect**              | **Pros**                                                                 | **Cons**                                                                                          |
|-------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Interpretability**     | Provides **feature importance scores** that are intuitive and easy to understand | Explanations can be **inconsistent** across settings or experiments, leading to **instability**    |
| **Model Agnosticism**    | **Flexible** – can be applied to any black-box model, even if the model is replaced | Can struggle with **nonlinear decision boundaries** due to its use of linear approximations         |
| **Flexibility**          | Works well with **tabular data, text, and images**                        | **Defining the neighborhood** and optimizing the kernel width can be difficult                     |
| **Efficiency**           | Provides **local explanations** without requiring a global model to be retrained | Can be **computationally expensive** when a large number of perturbations is required              |
| **Local Approximation**  | Approximates the **local decision boundary** around the instance, making it simple | May provide **oversimplified explanations**, missing complex model behavior                        |
| **Bias**                 | Can work with various data types and models, remaining **intuitive**      | Can be **fooled** or used to **hide biases**, as the explanation may focus on irrelevant features  |

#### Lime Image Explainer 

[lime 0.1 documentation](https://lime-ml.readthedocs.io/en/latest/lime.html#module-lime.lime_image)

The LIME Image Explainer (Local Interpretable Model-agnostic Explanations) is a technique designed to explain the predictions of black-box models for image classification tasks. It works by perturbing the original image—typically by segmenting the image into superpixels (small regions of similar pixels) and selectively altering these regions to create multiple perturbed versions of the image. The model’s predictions for these perturbed images are then used to train an interpretable local surrogate model, such as a linear classifier, which approximates the model's behavior in the vicinity of the original image. LIME highlights the most important regions of the image that contributed to the model's prediction, providing human-understandable visual explanations for complex models like neural networks. This approach is especially useful for understanding why certain parts of an image strongly influenced the model’s classification decision.

#### Summary of Pros and Cons for LIME Image Explainer

| **Aspect**              | **Pros**                                                                 | **Cons**                                                                                          |
|-------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Interpretability**     | Provides **visual explanations** by highlighting important image regions, making it intuitive | Explanations can be **inconsistent** across different image perturbations, leading to **instability** |
| **Model Agnosticism**    | **Flexible** – can be applied to any black-box image classifier, regardless of the model architecture | May struggle with highly **nonlinear decision boundaries** due to its linear approximations        |
| **Flexibility**          | Works well with **various image classification models**                   | Requires **careful tuning** of the perturbation process (e.g., superpixel size)                     |
| **Efficiency**           | Provides **local explanations** for specific images without retraining the model | Can be **computationally expensive** when a large number of image perturbations are required       |
| **Local Approximation**  | Approximates the **local decision boundary** of the model for each image   | May lead to **oversimplified explanations**, failing to capture complex model behaviors            |
| **Bias**                 | Provides **intuitive visual cues** for feature importance in images        | Can be **fooled** or manipulated to highlight **irrelevant regions**, leading to biased explanations |


### Grad-CAM 

Grad-CAM (Gradient-weighted Class Activation Mapping) is a visualization technique used to interpret the decisions of convolutional neural networks (CNNs), especially in image classification tasks. Unlike methods such as LIME that approximate the model's behavior using surrogate models, Grad-CAM uses the internal gradients of the model to produce a coarse localization map of the important regions in an image that contributed to the model’s prediction. It achieves this by backpropagating the gradients of the target class score with respect to the feature maps of the final convolutional layer. These gradients are then weighted and aggregated to highlight the regions in the image most relevant to the prediction.

#### Summary of Pros and Cons for Grad-CAM

| **Aspect**              | **Pros**                                                                 | **Cons**                                                                                          |
|-------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Interpretability**     | Produces intuitive heatmaps, showing regions of interest in the image. | Heatmaps can sometimes be too coarse to pinpoint specific features.                              |
| **Handling Complexity**  | Handles complex and nonlinear models effectively by leveraging gradients. | Limited to analyzing the final convolutional layer, potentially missing deeper interactions.       |
| **Model Dependency**     | Provides insights directly tied to the model’s internal structure.      | Cannot be applied to non-CNN architectures or requires modification for non-visual data.          |
| **Efficiency**           | Relatively efficient since it reuses gradients from backpropagation.   | Computational cost increases for high-resolution images or deeper networks.                       |
| **Visual Output**        | Highlights key regions responsible for predictions, aiding decision-making. | May focus on multiple irrelevant regions if gradients are noisy or poorly calibrated.             |

#### Grad-CAM in Practice

Grad-CAM is particularly useful for understanding and debugging CNN-based models in critical applications such as healthcare and autonomous vehicles. For example, in medical imaging, Grad-CAM can identify whether the model is focusing on clinically relevant regions, such as the lungs in chest X-rays, when diagnosing pneumonia. Similarly, in autonomous driving, it can highlight important features like road signs or pedestrians. By providing visually interpretable feedback on the model's predictions, Grad-CAM helps build trust in AI systems and enables researchers and practitioners to diagnose issues or refine model performance.


### Anchors Explaination

Anchors is a type of if-then rule used to explain individual predictions made by machine learning models. Unlike LIME, which approximates complex decision boundaries using local linear models, Anchors provide decision rules that "anchor" the prediction to certain features or conditions. When these conditions hold true, the model’s prediction remains consistent with high probability, even if other features change.

#### Summary of Pros and Cons for Anchors

| **Aspect**              | **Pros**                                                       | **Cons**                                                                                          |
|-------------------------|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Interpretability**     | Rules are easy to understand—if-then format is intuitive      | Requires careful parameter tuning, which can be challenging                                        |
| **Handling Complexity**  | Works well with nonlinear or complex models                   | Perturbation function needs to be custom-designed for each specific use case                       |
| **Model Agnosticism**    | Can be used with any model, model agnostic        | Coverage is hard to define and measure in some domains, making it difficult to compare             |
| **Efficiency**           | Highly efficient search algorithms like beam search           | Imbalanced data can result in biased perturbation spaces unless careful mitigation is used         |

#### AnchorImage 

[alibi.explainers.anchors.anchor_image module documentation](https://docs.seldon.io/projects/alibi/en/latest/api/alibi.explainers.anchors.anchor_image.html)

The AnchorImage explainer is an extension of the Anchors approach designed specifically for image classification tasks, providing highly interpretable, rule-based explanations for predictions made by complex models such as deep neural networks. Instead of relying on global model behavior or linear approximations, AnchorImage identifies if-then rules that are based on superpixels, or segmented regions of the image. These anchors represent key areas of the image that, when present, strongly influence the model’s prediction. The method generates a set of perturbed images by altering different regions, and it tests whether the model’s prediction remains consistent when the anchor region is unchanged. This allows AnchorImage to provide a highly stable and visual explanation by highlighting the specific areas of the image responsible for the model’s decision, making it an effective tool for understanding predictions in complex models like convolutional neural networks (CNNs).

#### Summary of Pros and Cons for AnchorImage Explainer

| **Aspect**              | **Pros**                                                                 | **Cons**                                                                                          |
|-------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **Interpretability**     | Provides easy-to-understand **if-then rules** based on image superpixels, making visual explanations intuitive | Requires **careful parameter tuning** for perturbations, such as defining superpixels and thresholds |
| **Handling Complexity**  | Works well with **nonlinear and complex models**, such as deep neural networks | Perturbation function and superpixel segmentation need to be **custom-designed** for each image dataset |
| **Model Agnosticism**    | Can be used with **any image classification model**, remaining model-agnostic | **Coverage** can be hard to measure, and it is difficult to define in certain domains               |
| **Efficiency**           | Uses **efficient search algorithms** like beam search to find anchor regions | **Computationally expensive** when generating and testing a large number of image perturbations     |
| **Stability**            | Provides **stable explanations** by identifying key superpixels that consistently influence predictions | **Imbalanced data** can lead to anchors focusing on irrelevant regions or failing to generalize     |

# Comparative Analysis of ResNet50 and VGG16 with XAI Methods for Edgy Dataset

## Broken Road Sign Experiment

The broken road sign experiment demonstrates that both ResNet50 and VGG16 successfully identified the road sign in the image, showcasing their capability to accurately classify traffic signs. However, the effectiveness of the explanations provided by the different XAI methods—LIME, Grad-CAM, and Anchor—varied significantly in terms of their interpretability, precision, and coverage, offering valuable insights into the strengths and weaknesses of each approach.

LIME provided localized explanations by highlighting the road sign as the central feature influencing the top-1 prediction. While it effectively grayed out irrelevant areas such as the background building, its interpretations for the top-5 predictions were inconsistent and sometimes misleading. For instance, in alternative predictions like "parking meter" or "traffic light," LIME highlighted irrelevant areas such as the parking lot or scattered features of the image. This inconsistency indicates that while LIME is intuitive and model-agnostic, its reliance on perturbation and linear approximations limits its ability to capture complex model behaviors and nuanced decision boundaries.

Grad-CAM, on the other hand, consistently highlighted regions centered around the road sign for both models but failed to provide detailed or class-specific insights. For top-5 predictions, it showed little variation in the regions of interest, making it difficult to differentiate between classes or understand the unique features contributing to each prediction. While Grad-CAM is computationally efficient and useful for coarse visual explanations, its lack of granularity and focus on similar regions across predictions limits its interpretability.

Anchor stood out as the most precise and interpretable method in this experiment. For both ResNet50 and VGG16, it highlighted the road sign's shape and the distinct outline of the word "STOP" on the sign, clearly indicating that text was the primary factor influencing the "traffic sign" prediction. The anchor precision of 1.00 demonstrated the method's reliability, as the highlighted features were sufficient to guarantee the correct prediction. However, with an anchor coverage of 0.50, the explanation applied to only half of the dataset instances, suggesting that while Anchor excels in specific scenarios, its generalizability may require further refinement.

In summary, while all three XAI methods contributed unique perspectives to model interpretability, Anchor proved to be the most reliable and insightful for this experiment, offering precise and intuitive explanations tied directly to the model's decision-making process. LIME provided flexible but sometimes inconsistent interpretations, and Grad-CAM offered less differentiation and clarity across predictions. These findings underscore the importance of selecting XAI methods based on their strengths and weaknesses to ensure meaningful and actionable insights, particularly in edge-case scenarios like this broken road sign experiment.

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

## Scooter in Fog Experiment

The scooter in fog experiment demonstrates that both ResNet50 and VGG16 were able to correctly classify the image of a pedestrian driving a motor scooter (Predicted class: 670). This highlights the models' ability to recognize the concept of a scooter, even under challenging conditions such as fog. However, the interpretability provided by XAI methods revealed varying levels of clarity and utility in understanding the models' decision-making processes.

**XAI Method Comparison:**
- **LIME:** LIME struggled to provide clear insights for both models. The explanations focused on fragmented, small regions of the image, graying out the majority of the scene. While these regions were consistent across predictions like "scooter," "scuba diver," and "oxygen mask," it remained unclear how such similar areas could result in drastically different predictions. This inconsistency undermined the utility of LIME in explaining the models' reasoning.  
- **Grad-CAM:** Grad-CAM also faced interpretability challenges, with both models showing little variation in focus across predictions. The highlighted areas consistently centered on the pedestrian driving the scooter but failed to differentiate between features contributing to correct predictions like "scooter" and other potential predictions. This lack of variation limited Grad-CAM's ability to provide meaningful insights.  
- **Anchor:** Anchor offered the most intuitive and precise explanations for both models. It highlighted the shape of the pedestrian driving the scooter as the key feature influencing the correct "motor scooter" prediction. For ResNet50, the Anchor precision was 0.98, and for VGG16, it was 0.95, demonstrating high reliability. However, the anchor coverage of 0.00 for both models indicated that the explanations were highly instance-specific, limiting their generalizability.

**Key Insights:**  
This experiment underscores the strengths and limitations of ResNet50 and VGG16 in interpreting challenging scenes. While both models successfully recognized the "scooter," XAI methods revealed gaps in their reasoning and focus. Anchor provided the clearest and most intuitive explanations, though its lack of generalization highlights the need for models to better integrate context and environmental cues for broader applicability. Improving interpretability and consistency in reasoning across predictions remains a crucial area for future development in model design and XAI methods.

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

## Car in Snow Experiment

The car in snow experiment highlights the challenges ResNet50 and VGG16 face in accurately interpreting snow-covered scenes, where object shapes and surrounding snow create visual ambiguities. ResNet50 predicted "bobsleigh," likely due to its focus on the car's snow-covered front, which resembles a bobsleigh, while VGG16 predicted "military aircraft," associating the car's elongated shape and uncovered features with the silhouette of a warplane. Both models struggled to recognize the car in its snow-obscured state, instead relying on shape-based reasoning that led to misclassification.

**XAI Method Comparison:**  
- **LIME:** LIME revealed that both models emphasized shape-based features to make predictions. For ResNet50, LIME highlighted the car's front for "bobsleigh" and shifted attention to the surrounding snow and other features for predictions like "snowplow" and "station wagon." Similarly, for VGG16, LIME focused on the car's uncovered parts and made predictions such as "snowmobile," "bobsleigh," and "tank," demonstrating confusion in the scene’s interpretation.  
- **Grad-CAM:** Grad-CAM provided limited insights for both models. For ResNet50, it focused broadly on the car and snow without significant differentiation between predictions. For VGG16, Grad-CAM highlighted different parts of the car but lacked depth in explaining the reasoning behind specific predictions like "military aircraft," reducing its overall interpretability.  
- **Anchor:** Anchor offered the most precise explanations for both models. For ResNet50, it highlighted the car, wall, and surrounding snow as contributing to the "bobsleigh" prediction, achieving high precision (0.98) but with zero generalizability (coverage: 0.00). For VGG16, Anchor emphasized the car’s shape and iron sheet-like features for the "military aircraft" prediction, with similar reliability (precision: 0.99) but limited coverage (0.06). These explanations underscored each model's reliance on specific shape-based elements rather than broader contextual reasoning.

**Key Insights:**  
This experiment highlights the reliance of both models on shape-based reasoning in snow-obscured scenes, leading to significant misclassification. Among the XAI methods, Anchor provided the most precise explanations, while LIME exposed the fragmented focus of the models, and Grad-CAM offered minimal differentiation. To improve performance in such scenarios, future models must incorporate better contextual reasoning and environmental awareness, enabling them to distinguish objects in complex, snow-covered environments more accurately.

# Conclusion 

Final Conclusion
Both ResNet50 and VGG16 correctly identify broken upside-down road sign, broken cars, motor scooters in fog, mostly because the generally shape and key features of the object (text for road sign and car shape and key features for car and motor scooter) is kept untacked and the environment that the image is captured is good and clear under daylight. However, when the shape of the object is significantly covered or distorted (such as pedestrian in rain with umbrella covering the head and half of the car is covered in snow )and when the picture is dark at night or object is hard to recognize in blurry situation (such as car in fog and at night) and when different objects are mixed together in one image (like car and pedestrian together in one image at night or several pedestrian with the street light), both models has significant difficulties in correctly identifying and classifying objects in the image.

XAI method conclusion (Grad-Cam the worst, Anchor the best) Generate a table
---

### Explainability Metrics Benchmark Table  

| **Dataset**       | **Model**  | **XAI Technique**      | **Explainability Analysis Case Study** |
|--------------------|------------|------------------------|-----------------------------------------|
| **Edge-Case Data** | ResNet50   | LIME                   | TBD                                     |
| **Edge-Case Data** | ResNet50   | Grad-CAM               | TBD                                     |
| **Edge-Case Data** | ResNet50   | Anchor                 | TBD                                     |
| **Edge-Case Data** | VGG16      | LIME                   | TBD                                     |
| **Edge-Case Data** | VGG16      | Grad-CAM               | TBD                                     |
| **Edge-Case Data** | VGG16      | Anchor                 | TBD                                     | 



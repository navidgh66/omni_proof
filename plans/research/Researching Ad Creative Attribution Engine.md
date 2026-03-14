# **Creative-Level Causal Attribution in Digital Advertising: An Exhaustive Technical Investigation**

## **Executive Summary**

The digital advertising ecosystem stands at a critical inflection point where the granularity of measurement must align with the granularity of optimization. For the past decade, the industry has relied on aggregate performance metrics—Click-Through Rates (CTR), Return on Ad Spend (ROAS), and Conversion Rates (CVR)—to judge the success of marketing campaigns. While these metrics provide a retrospective accounting of *what* happened, they are fundamentally incapable of explaining *why* it happened. This epistemic gap is most acute in the domain of **Creative Strategy**. Advertisers universally acknowledge that "creative is the new targeting," yet the tools to measure creative effectiveness remain stuck in a correlational paradigm. A video ad featuring a dog might outperform one featuring a cat, but without rigorous causal inference, it is impossible to determine if the "dog" element caused the lift, or if the ad platform's delivery algorithm simply routed that specific asset to a higher-intent audience segment.  
This report presents a comprehensive technical investigation into the feasibility, architecture, and methodological requirements for building an **Open-Source Engine for Creative-Level Causal Attribution**. It moves beyond the superficial layer of "creative analytics"—which typically amounts to tagging assets and averaging their performance—to propose a rigorous framework rooted in **Causal Machine Learning (CausalML)**. The objective is to isolate the **Average Treatment Effect (ATE)** and **Conditional Average Treatment Effect (CATE)** of specific, granular creative elements (e.g., visual hooks, semantic sentiment, color palettes, audio cues) on downstream business KPIs, while robustly controlling for the massive confounding variablesintroduced by algorithmic ad delivery systems.  
The investigation synthesizes findings from over 170 distinct research artifacts, spanning open-source code repositories, academic literature on heterogeneous treatment effects, whitepapers from proprietary ad-tech platforms, and adjacent domains such as recommendation system explainability.  
**Key Findings:**

1. **The "Compound Treatment" Problem:** An advertisement is not a single variable but a high-dimensional bundle of concurrent treatments (text, image, motion, sound). Isolating the effect of one component requires advanced decomposition techniques. The most promising methodological path involves a two-stage **Double Machine Learning (DML)** framework: first, estimating the "orthogonalized signal" of the total intervention to strip out targeting bias; and second, projecting these residuals onto a lower-dimensional feature space to recover element-level causal lifts.  
2. **The Algorithmic Confounder:** The most significant obstacle to causal inference in advertising is **Selection Bias**. Modern ad platforms (Meta, Google, TikTok) do not assign creatives randomly; they use "bandit" algorithms to exploit early performance signals. This creates a violation of the *ceteris paribus* condition. A naive comparison of Creative A vs. Creative B is mathematically invalid because they were likely shown to fundamentally different populations. The report details how **Inverse Probability Weighting (IPW)** and **Propensity Score** estimation, using proxies like CPM (Cost Per Mille) and placement data, are essential to re-weight the data and simulate a randomized controlled trial (RCT).  
3. **Tooling Maturity:** While no single "turnkey" open-source tool exists for this specific use case, the component technologies are mature. **CausalML** (Uber) and **EconML** (Microsoft) provide the necessary statistical estimators (S/T/X-Learners). **YOLOv8**, **PaddleOCR**, and **CLIP** provide the computer vision capabilities to unstructured media into structured covariates. The challenge lies in the *orchestration*—building a pipeline that feeds extracted visual features into causal estimators while handling the non-stationarity of "creative fatigue."  
4. **Creative Fatigue as a Causal Variable:** Performance decay is deterministic. Research from Meta demonstrates that creative responsiveness follows a power-law decay function (Effect \\propto (1+N)^{-\\lambda}). An effective engine must model "freshness" as a dynamic covariate, distinguishing between the intrinsic quality of an ad and its current state of saturation.

The report is structured to guide a technical architect through the entire lifecycle of building this engine: from the selection of open-source libraries (Section 1\) and the digestion of academic theory (Section 2), to the analysis of proprietary gaps (Section 3), the definition of the causal architecture (Section 4), and the strategy for data acquisition and synthetic benchmarking (Section 5).

## **1\. Existing Open-Source Projects & Tools**

The first dimension of our investigation focuses on the existing software landscape. The goal is to identify whether an open-source "Creative Causal Engine" already exists, or if one must be assembled from general-purpose libraries. The conclusion is definitively the latter: while there are powerful engines for *causal inference* and powerful engines for *feature extraction*, the intersection—**Creative Attribution**—remains a whitespace that requires custom orchestration.

### **1.1 General-Purpose Causal Inference Engines**

These libraries constitute the mathematical "kernel" of the proposed system. They implement the statistical estimators required to calculate lift.  
**CausalML (Uber)**

* **Repo/Source:**  
* **Core Functionality:** CausalML is a Python package specifically designed for **Uplift Modeling** and estimating **Conditional Average Treatment Effects (CATE)**. It provides a unified interface for various meta-learners, including S-Learner, T-Learner, X-Learner, and R-Learner.  
* **Relevance to Creative Attribution:** High. The "uplift" paradigm—identifying which user segment responds best to a treatment—maps directly to the creative problem (e.g., "Which users respond to 'User Generated Content' style ads?").  
* **Strengths:** It is optimized for large-scale datasets typical of advertising logs. The implementation of **Tree-based algorithms** (Uplift Random Forests) allows for the automatic discovery of heterogeneous segments, which is critical when we don't know *a priori* which audience will prefer a specific creative element.  
* **Actionable Application:** The **X-Learner** implementation in CausalML is particularly robust for datasets with unbalanced treatment assignment, a common scenario in ad auctions where "winning" creatives get 90% of the impressions and "losing" creatives get 10%.

**EconML (Microsoft)**

* **Repo/Source:**  
* **Core Functionality:** EconML focuses on **structural causal models** and **Double Machine Learning (DML)**. It is built to combine the flexibility of machine learning (for prediction) with the rigor of econometrics (for causal identification).  
* **Relevance to Creative Attribution:** Very High. DML is arguably the most appropriate methodology for handling the "nuisance parameters" of ad targeting. In an ad auction, the probability of exposure (propensity) is a complex function of user features. DML allows us to use an arbitrary ML model (e.g., a Gradient Boosted Tree) to learn this propensity function and "partial it out," leaving the pure causal effect of the creative.  
* **Strengths:** Extensive support for **Orthogonal Forests** and **Deep IV** (Instrumental Variables), which are powerful for high-dimensional confounding adjustment.

**DoWhy (PyWhy/Microsoft)**

* **Repo/Source:**  
* **Core Functionality:** DoWhy is a library for causal *reasoning* rather than just estimation. It forces the user to explicitly define a **Causal Graph (DAG)** and provides a "Refutation API" to test the robustness of causal claims (e.g., "What if I add a random common cause? Does the effect disappear?").  
* **Relevance:** Medium-High. While less performant for massive data processing than CausalML, it is essential for the *design phase* of the engine. It ensures that the assumptions regarding confounders (e.g., "Does 'Time of Day' affect both 'Creative Choice' and 'Conversion'?") are explicitly modeled and tested.

**OpenASCE (Ant Group)**

* **Repo/Source:**  
* **Core Functionality:** The "Open All-Scale Causal Engine" is designed for industrial-scale applications like Alipay. It includes novel implementations for **causal representation learning** and tree-based methods optimized for distributed systems.  
* **Relevance:** Emerging. It addresses the scalability bottleneck. Ad logs can generate terabytes of data daily. OpenASCE’s focus on distributed computing and "large-scale" estimators makes it a strong candidate for the production deployment of the engine, particularly for the **attribution** (root cause analysis) of conversion events.

**PeLICAn (Neustar)**

* **Repo/Source:**  
* **Core Functionality:** PeLICAn (Private Learning and Inference for Causal Attribution) is a framework focused on **privacy-preserving** attribution. It uses sensitivity analysis to attribute outcomes to sequences of marketing events.  
* **Relevance:** Niche but Critical. As privacy regulations (GDPR, CCPA) and platform changes (SKAdNetwork, Privacy Sandbox) restrict access to user-level data, the architecture of PeLICAn—which operates on aggregated or obfuscated data—provides a blueprint for a "future-proof" engine that does not rely on persistent user IDs.

### **1.2 Marketing Mix Modeling (MMM) Tools**

While MMM typically operates at the channel level (TV vs. Facebook), recent open-source innovations are pushing these tools toward granular, creative-aware modeling.  
**Robyn (Meta)**

* **Repo/Source:**  
* **Methodology:** Semi-automated MMM using **Ridge Regression** and evolutionary algorithms (Nevergrad) for hyperparameter optimization.  
* **Creative Decomposition Capability:** Limited but hackable. Robyn allows for "context variables" and organic variables. A sophisticated user can effectively "hack" Robyn by feeding creative features (e.g., "Spend\_Video\_Humor", "Spend\_Static\_Product") as separate media channels.  
* **Limitation:** It relies on **Ridge Regression** to handle multicollinearity. While Ridge helps stability, it is not a causal estimator in the potential outcomes sense. It does not account for selection bias (who saw the ad), only for the correlation of spend with outcome. It assumes that the "spend" signal is the primary driver, which is true for media weight but not for creative *quality*.

**Meridian (Google)**

* **Repo/Source:**  
* **Methodology:** **Bayesian Hierarchical Modeling**.  
* **Creative Decomposition Capability:** Meridian’s Bayesian nature allows for the injection of **priors**. If we have external experimental evidence (e.g., a Brand Lift study) that "Faces" increase lift by 10%, we can set this as a prior in the model.  
* **Advantage:** It explicitly supports **Reach & Frequency (R\&F)** data. This is critical for modeling **Creative Fatigue**. Unlike Ridge Regression, which sees "Spend," Meridian can see "How many times did the average user see this creative?" This allows for the estimation of saturation curves at the creative level.

### **1.3 Feature Extraction Libraries (The "Eyes" and "Ears")**

An attribution engine is blind without the ability to "see" the ad.  
**AdDownloader & Visual Analysis**

* **Source:**  
* **Function:** An open-source Python package to retrieve ads from the Meta Ad Library and perform analysis using **BLIP** (Bootstrapping Language-Image Pre-training) for captioning and visual QA.  
* **Relevance:** This is a key "ingestion" component. It demonstrates how to automate the retrieval of creative assets for analysis.

**HuggingFace Transformers & Document Intelligence**

* **Source:**  
* **Function:** Models like **LayoutLM** and **Donut** (Document Understanding Transformer) are designed to understand text *in spatial context*.  
* **Relevance:** For static ads, the *position* of the Call to Action (CTA) matters as much as the text. Is the button top-left or bottom-right? These models can extract structured layout features ("CTA\_Position: Bottom-Right") which serve as treatment variables.

**Summary of Section 1:** There is no "CreativeCausalEngine" repo. However, the stack is clear:

1. **Ingestion:** AdDownloader / Custom Scrapers.  
2. **Vision:** YOLO / CLIP / LayoutLM (Open Source Models).  
3. **Inference:** CausalML (X-Learner) or EconML (DML).  
4. **Privacy:** PeLICAn architecture.

## **2\. Academic Research & Papers**

To build an engine that goes beyond "pseudo-science," we must ground the methodology in rigorous academic theory. The literature provides solutions to the two hardest problems in this domain: **Compound Treatments** (an ad is a bundle of features) and **Fatigue** (the effect changes over time).

### **2.1 The Theory of Compound Marketing Interventions**

The foundational text for this domain is the research on **"Estimating the Effects of Component Treatments in Compound Marketing Interventions"**. This work addresses the exact problem of creative attribution: digital ads are "compound interventions" containing multiple simultaneous components (image, headline, body text, CTA, promotion).  
**The Methodological Breakthrough:** Standard A/B testing fails here because the number of combinations is combinatorial. Testing every headline with every image is impossible. The paper proposes a **Double Machine Learning (DML)** framework to decompose these effects from observational data.  
**The Algorithm (The "Orthogonalization" Process):**

1. **Stage 1 (Debiasing):** The goal is to strip out the targeting bias.  
   * Let Y be the outcome (conversion).  
   * Let T be the compound treatment (the specific ad ID).  
   * Let X be the user covariates (targeting).  
   * Train a model g(X) to predict the probability of seeing ad T (Propensity).  
   * Train a model h(X) to predict the outcome Y from X alone.  
   * Calculate Residuals: Y\_{res} \= Y \- h(X) and T\_{res} \= T \- g(X).  
   * *Insight:* These residuals represent the "surprise" outcome and "surprise" treatment assignment that cannot be explained by user targeting.  
2. **Stage 2 (Decomposition):**  
   * Project the residuals (Y\_{res}) onto the **component features** (C) of the creative (e.g., C \= {Has\_Face, Is\_Video, Color\_Red}).  
   * This projection recovers the causal lift of each *element*.

**Relevance:** This is the blueprint for the engine. It proves that we do not need to run randomized tests for every single feature; we can recover element-level lift from the "messy" logs of ad campaigns if we rigorously control for targeting bias first.

### **2.2 Heterogeneous Treatment Effects (HTE) & Meta-Learners**

Research by Künzel et al. (2019) on the **X-Learner** is critical for the "sparse data" problem in creative analytics.

* **The Problem:** In advertising, the "Control" group is often ill-defined (who is the control for a specific Nike ad? People who saw a generic shoe ad? Or no ad?). Furthermore, the treatment group for a specific creative variant might be small.  
* **The Solution:** The X-Learner is designed for unbalanced designs. It estimates the treatment effect by training two separate models (one for treated, one for control) and then "crossing" them to impute counterfactuals for every data point.  
* **Application:** This allows the engine to output **Conditional Average Treatment Effects (CATE)**. Instead of saying "Red Backgrounds work," the engine can say "Red Backgrounds increase CTR by 0.5% for iOS users but decrease it by 0.2% for Android users." This level of granularity is essential for *personalization*.

### **2.3 Creative Fatigue: The Causal Decay Function**

Research from Meta’s analytics team provides a causal formulation for **Creative Fatigue**.

* **The Finding:** Creative performance is non-stationary. It decays as a function of **Frequency** (number of exposures).  
* **The Decay Law:** The reduction in click likelihood follows a power law: Effect \\propto (1+N)^{-\\lambda}, where N is the prior exposure count.  
* **Causal Validation:** Meta validated this using "dose-dependent" experiments. They injected fresh creatives into ad sets with varying levels of accumulated fatigue. The lift from the new creative was proportional to the fatigue level of the old one, confirming a causal link.  
* **Implication for the Engine:** The engine cannot treat "Creative Quality" as a static constant. It must compute a **"Freshness-Adjusted"** score. The causal model must include cumulative\_impressions as a covariate or interaction term.

### **2.4 Deep Feature Extraction & Causal Representation**

Recent work in **Deep Causal Representation Learning** attempts to bridge the gap between "Deep Learning" (which finds correlations) and "Causal Inference" (which seeks structure).

* **DeepFEIVR:** This framework uses **Instrumental Variable (IV)** regression with deep neural networks. It extracts features from images (like MRI scans in the paper, but applicable to ads) that are causally related to the outcome, filtering out "spurious" features that are merely correlated.  
* **Relevance:** Standard CNNs might learn that "Snow" predicts "Sales" (because winter coats sell in winter). A Causal Representation Learner would try to disentangle the "Snow" (context) from the "Product Image" (causal driver).

## **3\. Industry Approaches & Proprietary Platforms**

To build a competitive open-source engine, we must understand the "state of the art" in the proprietary world. The industry is currently bifurcated between **Pre-Testing Tools** (simulations) and **In-Flight Analytics** (measurement).

### **3.1 The "Creative Quality Score" Approach (CreativeX / VidMob)**

Platforms like **CreativeX** and **VidMob** have industrialized the feature extraction pipeline.

* **Methodology:** They rely heavily on **Computer Vision (CV)** to tag every frame of a video. They check for "Best Practices" (e.g., "Brand logo in first 3 seconds," "Human face present," "Framing for mobile").  
* **The "Creative Quality Score" (CQS):** They aggregate these checks into a single score.  
* **The Gap:** Their approach is often **normative**, not **causal**. They assume that "Best Practices" are universally valid. If the engine detects a logo in the first 3 seconds, it assigns a high score. However, for a specific "mystery" campaign, revealing the logo early might *hurt* curiosity and CTR.  
* **Opportunity:** An open-source engine can improve on this by learning the **empirical weights** of these features for *each specific advertiser*, rather than relying on global "best practices."

### **3.2 The Attention Prediction Approach (Neurons / Dragonfly AI)**

* **Methodology:** These tools use models trained on eye-tracking data to generate **Saliency Maps**. They predict *where* a user will look.  
* **Limitation:** Attention \\neq Conversion. A user might stare at a confusing element (high saliency) but fail to click (low conversion).  
* **Opportunity:** Saliency maps should be treated as a **feature input** for the causal engine, not the final output. The causal engine can test the hypothesis: "Does high saliency on the CTA button causally drive conversion?"

### **3.3 The MMM Platforms (Robyn & Meridian)**

As discussed in Section 1, **Robyn** (Meta) and **Meridian** (Google) are the giants of attribution.

* **Industry Posture:** They are positioning MMM as the "source of truth" in a post-cookie world.  
* **The Creative Blindspot:** Both tools struggle with the *granularity* of creative. They model "Channel" (e.g., Facebook) or "Campaign" (e.g., Summer Sale). They do not natively handle the thousands of unique creative IDs and their visual features.  
* **Integration Point:** The proposed open-source engine should sit *upstream* of these MMMs. It should aggregate creative performance into a "Creative Quality Index" which is then fed into Robyn/Meridian as a context variable. This allows the MMM to adjust the channel's efficiency based on the quality of the creative running at that time.

## **4\. Methodological Landscape: The Engine Architecture**

This section details the theoretical and architectural blueprint for the engine. It combines the components identified above into a cohesive system.

### **4.1 Feature Extraction Pipeline (The "Decomposition Layer")**

Before any causal math can occur, the unstructured creative asset must be transformed into a structured vector of covariates (X\_{creative}).  
**A. Visual Decomposition (Computer Vision)**

* **Object Detection:** Use **YOLOv8** to generate a count vector of objects.  
  * *Output:* \[person: 1, car: 0, dog: 1, laptop: 0...\]  
* **Layout Analysis:** Use **LayoutLM** or simple OCR bounding boxes to determine the spatial composition.  
  * *Features:* CTA\_Position\_X, CTA\_Position\_Y, Text\_Area\_Ratio.  
* **Face & Emotion:** Use **DeepFace**.  
  * *Features:* Face\_Count, Dominant\_Emotion (Happy/Surprise/Neutral), Demographic\_Group.  
* **Visual Embeddings:** Use **CLIP** (Contrastive Language-Image Pre-Training).  
  * *Method:* Extract the 512-dimensional embedding vector.  
  * *Cluster:* Perform K-Means clustering on these embeddings to identify "Visual Archetypes" (e.g., Cluster 1 \= "Minimalist/Pastel", Cluster 2 \= "High Contrast/Neon"). Use Cluster ID as a categorical treatment variable.

**B. Textual & Audio Decomposition (NLP)**

* **OCR:** Use **PaddleOCR** or **Google Vision API** to extract text overlays.  
* **Hook Analysis:** Analyze the first 3 seconds of transcript/overlay.  
  * *Model:* Fine-tune a **BERT** classifier on ad copy to identify "Hook Types" (e.g., "Problem/Solution," "Social Proof," "Question").  
* **Sentiment:** Use **RoBERTa** to score sentiment (Positive/Negative/Urgent).

**C. Metadata Features**

* **Motion Score:** Calculate **Optical Flow** (pixel displacement between frames) to measure "energy" or "pacing."  
* **Audio Features:** Use **YAMNet** to tag audio events (Music, Speech, Silence).

### **4.2 Causal Inference Engine (The "Reasoning Layer")**

Once we have the feature vector X\_{creative} and the performance logs (User u, Creative c, Outcome y, Confounders Z), we apply causal estimation.  
**Technique 1: Double Machine Learning (DML) for Feature Attribution** This is the workhorse for estimating the Average Treatment Effect (ATE) of specific features.

* **Step 1 (Propensity Model):** Train a classifier (e.g., XGBoost) to predict the presence of a feature (e.g., "Has\_Face") based on the confounders (Audience, Time, Spend).  
  * P(Has\\\_Face | Z).  
* **Step 2 (Outcome Model):** Train a regressor to predict the outcome (e.g., Conversion Rate) based on the confounders.  
  * E\[Conversion | Z\].  
* **Step 3 (Residualization):**  
  *   
  *   
* **Step 4 (Causal Estimation):** Regress Residual\_{Conv} on Residual\_{Face}.  
  * The resulting coefficient \\beta is the causal lift of having a face, *purged of targeting bias*.

**Technique 2: Inverse Probability Weighting (IPW) for Campaign Comparison** To compare two distinct creative concepts (Ad A vs. Ad B) that were shown to different audiences:

* Calculate the propensity score e(Z) \= P(Ad\\\_A | Z).  
* Weight each observation of Ad A by 1/e(Z) and each observation of Ad B by 1/(1-e(Z)).  
* This creates a "pseudo-population" where Ad A and Ad B appear to be assigned randomly, allowing for a direct comparison of their means.

**Technique 3: Causal Forests for Heterogeneity**

* Use the **CausalForest** implementation in **EconML** or **OpenASCE**.  
* The forest splits the data not to maximize *purity* (like a normal Random Forest) but to maximize the *difference in treatment effect*.  
* *Output:* It identifies the rules that define heterogeneity (e.g., "Faces work well when Device=Mobile and Time=Weekend, but poorly when Device=Desktop").

### **4.3 Handling Confounders: The "Proxy" Strategy**

The engine faces a "Hidden Confounder" problem: we do not see the proprietary "User ID" or the platform's internal "User Quality Score." We must use **Proxies**.

* **CPM (Cost Per Mille):** In an auction, the price paid (CPM) is a direct proxy for the demand/quality of the user. If the platform shows an ad to a high-intent user, the CPM is higher. Including CPM as a confounder in the DML models is essential to control for audience quality.  
* **Placement ID:** "Facebook Feed" vs. "Audience Network" implies vastly different user intent. This must be a categorical confounder.  
* **Device & OS:** Proxy for socio-economic status.

## **5\. Datasets & Benchmarks**

Building this system requires data. Since ad performance data is highly sensitive, the open-source community relies on public proxies and synthetic generation.

### **5.1 Public Datasets**

1. **AdImageNet:**  
   * **Description:** A dataset of 9,000+ programmatic ad creatives with metadata (dimensions, extracted text).  
   * **Usage:** Ideal for pre-training the *Feature Extraction* pipeline. It helps the vision models learn the "visual language" of ads (which differs from natural scenes).  
   * **Limitation:** No performance labels (clicks/conversions).  
2. **CausalML Example Datasets:**  
   * **Description:** Synthetic datasets included in the Uber CausalML repo.  
   * **Usage:** Validating the *statistical correctness* of the inference engine.  
3. **Kaggle Marketing Datasets:**  
   * **Description:** Campaign-level performance data.  
   * **Usage:** Testing the DML pipeline on "macro" variables, though lacking the granular creative assets.

### **5.2 Synthetic Data Generation (The "Digital Twin" Approach)**

To rigorously benchmark the engine, we must generate **Synthetic Data** where the "Ground Truth" causal effects are known.  
**Workflow:**

1. **Asset Generation:** Use a Generative AI model (Stable Diffusion) to create 1,000 synthetic ad variations (varying text size, background color, objects).  
2. **Truth Definition:** Define a hidden causal rule (e.g., Rule: "Blue Background" adds \+0.05 to CTR).  
3. **Confounding Simulation:** Simulate an "Ad Server" that introduces bias (e.g., If "Blue Background", show to "Low Income" users).  
4. **Data Generation:** Generate a log of 1 million impressions based on these rules.  
5. **Benchmarking:** Feed this log into the engine. Does the engine recover the \+0.05 lift for "Blue Background"? Or does it get confused by the "Low Income" bias?  
   * *Tools:* **Syntho**, **SDV**, or custom Python scripts using numpy to simulate the Structural Causal Model (SCM).

## **6\. Key Challenges & Open Problems**

### **6.1 Multicollinearity (The "Style" Problem)**

Creative elements are highly correlated. A "Luxury" ad might simultaneously use "Serif Font," "Black Background," and "Slow Motion."

* **The Issue:** Mathematically, it is difficult to assign partial credit to "Serif Font" if it *always* appears with "Black Background."  
* **Solution:**  
  * **Feature Grouping:** Aggregate collinear features into "Themes" and attribute lift to the Theme first.  
  * **Principal Component Analysis (PCA):** Reduce the feature space before causal estimation.  
  * **Recommendation:** The engine should flag these clusters and suggest **"De-correlation Experiments"** (e.g., "Run a test with Serif Font on a *White* Background").

### **6.2 Positivity Violation (The "Bad Ad" Problem)**

Causal inference requires **Positivity** (Overlap): every user type must have a non-zero probability of seeing every ad type.

* **The Issue:** Ad platforms kill bad ads quickly. We rarely see "Bad Creatives" shown to "Good Audiences" (High CPM).  
* **Solution:**  
  * **Exploration Filtering:** Restrict the analysis to the "Learning Phase" (first \~5000 impressions) of campaigns where the bandit algorithm is still exploring.  
  * **Clipping:** Use "Trimmed IPW" to discard data points with extreme propensity scores (where the probability of exposure was near 0).

### **6.3 Non-Stationarity (Creative Fatigue)**

As detailed in Section 2.3, the effect of a creative is not constant.

* **The Issue:** A DML model assumes a static Treatment Effect (\\tau).  
* **Solution:** Use **Time-Varying Effect Models**. The engine must calculate a "Fatigue Index" (f \= (1+N)^{-\\lambda}) and include it as an interaction term. The output should be: "This creative has a base quality of X, but is currently performing at X-Y due to fatigue."

## **7\. Adjacent Domains to Draw From**

### **7.1 Recommendation Systems (RecSys) Explainability**

The question "Why did this ad work?" is isomorphic to "Why did Netflix recommend this movie?"

* **Technique:** **Counterfactual Explanations**.  
* **Application:** The engine can use **Perturbation Analysis**. "If we perturbed the 'Color' feature from Red to Blue, how much would the predicted CTR change?" This is used to generate human-readable explanations ("This ad worked primarily because of the 'Red' background").

### **7.2 Uplift Modeling**

* **Technique:** **Class Transformation Method** (used in CausalML).  
* **Application:** Standard ads measurement looks at *response*. Uplift modeling looks at the *difference* in response between Treatment and Control. This is vital for distinguishing between "Sure Things" (users who would buy anyway) and "Persuadables" (users who buy *only because* of the creative).

## **8\. Conclusion**

The construction of an **Open-Source Engine for Creative-Level Causal Attribution** is a formidable but solvable engineering challenge. It requires a synthesis of three distinct disciplines: **Computer Vision** (to see the ad), **Causal Inference** (to reason about the ad), and **Ad Tech Domain Knowledge** (to model the auction dynamics).  
**The Blueprint for the Builder:**

1. **Do not reinvent the math:** Use **CausalML** or **EconML** as the core estimator. Focus your engineering effort on the *pipeline*.  
2. **The "Compound Treatment" DML framework** is the correct architectural choice. It is the only method rigorous enough to handle the high-dimensional, collinear nature of creative features.  
3. **Data is the bottleneck.** Invest heavily in **Synthetic Data Generation** to validate your causal estimators before deploying them on live ad spend.  
4. **Respect the Algorithm.** You cannot ignore the ad platform's delivery logic. You must model the **Propensity Score** using proxies like CPM and Placement to strip out the massive selection bias that plagues this industry.

By following this roadmap, it is possible to move the industry from the "Art of Creative" to the "Science of Creative Attribution."

## **Key Tables & Comparisons**

### **Table 1: Comparative Analysis of Causal Libraries for Creative Attribution**

| Library | Primary Methodology | Creative Attribution Readiness | Best Use Case |
| :---- | :---- | :---- | :---- |
| **CausalML** (Uber) | Meta-learners (X/T-Learner), Uplift Trees | **High** | Estimating **CATE** (Heterogeneity)—finding which audience segment prefers which creative element. |
| **EconML** (Microsoft) | Double Machine Learning (DML), Orthogonal Forests | **Very High** | Handling **High-Dimensional Confounding**. Best for isolating the "pure" lift of a visual tag while controlling for complex targeting. |
| **DoWhy** (PyWhy) | Structural Causal Models (SCM), Refutation | **Medium** | **Model Validation**. Use it to define the causal graph and stress-test assumptions before running the heavy estimators. |
| **Robyn** (Meta) | Ridge Regression (MMM) | **Low** (requires hacking) | **Budget Allocation**. Good for determining *how much* to spend on "Video" vs "Static", but poor for granular element analysis. |
| **OpenASCE** (Ant) | Tree-based, Large-scale Distributed | **High** (Emerging) | **Scale**. Best if processing terabytes of log data where standard Python libraries might bottleneck. |

### **Table 2: Feature Extraction Pipeline Components**

| Modality | Target Features | Recommended Open-Source Model |
| :---- | :---- | :---- |
| **Visual Objects** | Persons, Products, Logos, Vehicles | **YOLOv8** (Ultralytics) |
| **Visual Style** | Aesthetic, Mood, Semantic Concepts | **CLIP** (OpenAI/HuggingFace) |
| **Text (In-Image)** | Keywords, CTA presence, Text Area | **PaddleOCR** |
| **Faces** | Emotion, Age, Gender, Count | **DeepFace** (VGG-Face) |
| **Layout** | Spatial composition, White space | **LayoutLM** |
| **Motion** | Pacing, Velocity, Cuts-per-minute | **OpenCV** (Optical Flow) |
| **Audio** | Music genre, Voiceover presence | **YAMNet** / **OpenL3** |

#### **Works cited**

1\. Estimating Marketing Component Effects: Double Machine Learning ..., https://pubsonline.informs.org/doi/10.1287/mksc.2022.1401 2\. Causal Inference \- Kaiser Permanente Division of Research, https://divisionofresearch.kaiserpermanente.org/research/biostatistics/causal-inference/ 3\. How should we estimate inverse probability weights with possibly misspecified propensity score models? | Political Science Research and Methods, https://www.cambridge.org/core/journals/political-science-research-and-methods/article/how-should-we-estimate-inverse-probability-weights-with-possibly-misspecified-propensity-score-models/7590026E7B84B8BBE8329745D3E6615F 4\. CausalML Package Example Dataset \- Kaggle, https://www.kaggle.com/datasets/vikasmalhotra08/causalml-package-example-dataset 5\. (PDF) AdDownloader: Automating the retrieval of advertisements and their media content from the Meta Online Ad Library \- ResearchGate, https://www.researchgate.net/publication/381575085\_AdDownloader\_Automating\_the\_retrieval\_of\_advertisements\_and\_their\_media\_content\_from\_the\_Meta\_Online\_Ad\_Library 6\. Creative Fatigue: How advertisers can improve performance by managing repeated exposures | by Analytics at Meta | Medium, https://medium.com/@AnalyticsAtMeta/creative-fatigue-how-advertisers-can-improve-performance-by-managing-repeated-exposures-e76a0ea1084d 7\. uber/causalml: Uplift modeling and causal inference with machine learning algorithms \- GitHub, https://github.com/uber/causalml 8\. About CausalML, https://causalml.readthedocs.io/en/latest/about.html 9\. Metalearners for estimating heterogeneous treatment effects using machine learning \- PNAS, https://www.pnas.org/doi/10.1073/pnas.1804597116 10\. Fisher-Schultz Lecture: Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments, With an \- MIT Economics, https://economics.mit.edu/sites/default/files/2025-07/2025.07%20Fisher-Schultz%20Lecture.pdf 11\. attribution-modeling · GitHub Topics, https://github.com/topics/attribution-modeling 12\. Open-All-Scale-Causal-Engine/OpenASCE: OpenASCE ... \- GitHub, https://github.com/Open-All-Scale-Causal-Engine/OpenASCE 13\. neustar/pelican: Private Learning and Inference for Causal ... \- GitHub, https://github.com/neustar/pelican 14\. Mastering Marketing Mix Models: From Theory to Practice with Meta's Robyn, https://akshaykapoor020.medium.com/mastering-marketing-mix-models-from-theory-to-practice-with-metas-robyn-7117e9dc0485 15\. An Analyst's Guide to MMM | Robyn \- GitHub Pages, https://facebookexperimental.github.io/Robyn/docs/analysts-guide-to-MMM/ 16\. Key Features | Robyn \- GitHub Pages, https://facebookexperimental.github.io/Robyn/docs/features/ 17\. How Facebook Robyn Works \- Recast, https://getrecast.com/facebook-robyn/ 18\. Clarification Questions · Issue \#214 · facebookexperimental/Robyn \- GitHub, https://github.com/facebookexperimental/Robyn/issues/214 19\. ROBYN UNDER THE HOOD \- Deep Dive into Meta's MMM Library Robyn | Medium, https://ridhima-kumar0203.medium.com/robyn-under-the-hood-6a06730ed9a5 20\. MMM Unified Schema | Meridian \- Google for Developers, https://developers.google.com/meridian/docs/user-guide/mmm-unified-schema 21\. Google Meridian: What you need to know, https://www.impressiondigital.com/blog/google-meridian-what-you-need-to-know/ 22\. Google Meridian MMM: The 2025 Guide for Marketers \- Eliya, https://www.eliya.io/blog/media-mix-modeling/google-meridian-mmm 23\. Introduction to Meridian Demo | Google for Developers, https://developers.google.com/meridian/notebook/meridian-getting-started 24\. Daily Papers \- Hugging Face, https://huggingface.co/papers?q=Document%20extraction 25\. Comparing Meta-Learners for Estimating Heterogeneous Treatment Effects and Conducting Sensitivity Analyses \- MDPI, https://www.mdpi.com/2297-8747/30/6/139 26\. Deep causal feature extraction and inference with neuroimaging genetic data \- PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC11193942/ 27\. Causal Inference Meets Deep Learning: A Comprehensive Survey \- PMC \- NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC11384545/ 28\. Unlocking Creative Impact in Marketing Mix Models \- VidMob, https://vidblog.vidmob.com/vidmob-resources/unlocking-creative-impact-in-marketing-mix-models 29\. How Creative Data is Transforming MMM \- VidMob, https://vidblog.vidmob.com/blog/how-creative-data-tranforms-mmm 30\. Shifting perspectives on influencer marketing across the funnel, https://go.impact.com/rs/280-XQP-994/images/WARC-impact.com-Influencer-marketing.pdf 31\. Vision and convolutional transformers for Alzheimer's disease diagnosis: a systematic review of architectures, multimodal fusion and critical gaps \- PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC12764722/ 32\. Proceedings of the 2nd Workshop on Ontology Learning and Population \- ACL Anthology, https://aclanthology.org/W06-05.pdf 33\. Relationship-Extractor-NLP/preprocessing\_All\_Data.ipynb at main \- GitHub, https://github.com/sadam-99/Relationship-Extractor-NLP/blob/main/preprocessing\_All\_Data.ipynb 34\. Targeting resources efficiently and justifiably by combining causal machine learning and theory \- Frontiers, https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.1015604/full 35\. AI Fairness in Data Management and Analytics: A Review on Challenges, Methodologies and Applications \- MDPI, https://www.mdpi.com/2076-3417/13/18/10258 36\. Regression-based proximal causal inference | American Journal of Epidemiology | Oxford Academic, https://academic.oup.com/aje/article/194/7/2030/7775568 37\. PeterBrendan/AdImageNet · Datasets at Hugging Face, https://huggingface.co/datasets/PeterBrendan/AdImageNet 38\. Marketing Campaign Performance Dataset \- Kaggle, https://www.kaggle.com/datasets/manishabhatt22/marketing-campaign-performance-dataset 39\. Synthetic Data \- An Explainer \- IAB Australia, https://iabaustralia.com.au/guideline/synthetic-data-an-explainer/ 40\. Synthetic Data Generation for Digital Marketers and Brand Strategists \- NextBrain AI, https://nextbrain.ai/blog/synthetic-data-generation-for-digital-marketers-and-brand-strategists 41\. Introducing the Synthetic Data Generator \- Build Datasets with Natural Language, https://huggingface.co/blog/synthetic-data-generator 42\. New accurate, explainable, and unbiased machine learning models for recommendation with implicit feedback., https://ir.library.louisville.edu/cgi/viewcontent.cgi?article=5128\&context=etd 43\. Learning Causal Explanations for Recommendation \- CEUR-WS.org, https://ceur-ws.org/Vol-2911/paper3.pdf
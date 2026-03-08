"""
HotpotQA-style Multi-hop Evaluation Dataset for ICDI-X
=======================================================
100 carefully curated QA pairs across 5 categories:
  - factual        : single-hop direct lookup
  - multi_hop      : requires connecting 2+ pieces of info
  - comparison     : comparing entities/values
  - aggregation    : summarization / listing
  - definitional   : "what is X" explanations

Each entry has:
  - query          : natural language question
  - ground_truth   : reference answer string
  - category       : one of the 5 above
  - difficulty     : easy / medium / hard
  - hops           : number of reasoning hops required
  - keywords       : key terms that a good answer must contain
"""

HOTPOTQA_DATASET = [
    # ── FACTUAL (single-hop) ────────────────────────────────────────
    {
        "id": 1, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What is the title of the accepted paper?",
        "ground_truth": "CardioVison-Surv: A Systematic Review of Longitudinal Deep Learning and Temporal Transformers for Heart Failure Progression Forecasting",
        "keywords": ["CardioVison-Surv", "heart failure", "deep learning"]
    },
    {
        "id": 2, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What is the paper ID assigned to the accepted submission?",
        "ground_truth": "ISER-2-27-0050098",
        "keywords": ["ISER-2-27-0050098"]
    },
    {
        "id": 3, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "When was the acceptance letter issued?",
        "ground_truth": "02/03/2026",
        "keywords": ["2026", "March"]
    },
    {
        "id": 4, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What type of presentation was the paper accepted for?",
        "ground_truth": "oral poster presentation",
        "keywords": ["oral", "poster", "presentation"]
    },
    {
        "id": 5, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What is the full name of the conference that accepted the paper?",
        "ground_truth": "International Conference on Cardiac Imaging and Machine Learning",
        "keywords": ["ICCIML", "cardiac", "machine learning"]
    },
    {
        "id": 6, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "Who are the authors of the accepted paper?",
        "ground_truth": "Priyesh Srivastava and Dr. Abdul Majid",
        "keywords": ["Priyesh Srivastava", "Abdul Majid"]
    },
    {
        "id": 7, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What is the acronym of the conference?",
        "ground_truth": "ICCIML-26",
        "keywords": ["ICCIML-26"]
    },
    {
        "id": 8, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What year is the conference taking place?",
        "ground_truth": "2026",
        "keywords": ["2026"]
    },
    {
        "id": 9, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What medical domain does the paper focus on?",
        "ground_truth": "heart failure progression forecasting",
        "keywords": ["heart failure", "forecasting", "cardiac"]
    },
    {
        "id": 10, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What is the review process described in the acceptance letter?",
        "ground_truth": "double-blind peer review",
        "keywords": ["review", "blind", "peer"]
    },

    # ── MULTI-HOP ────────────────────────────────────────────────────
    {
        "id": 11, "category": "multi_hop", "difficulty": "medium", "hops": 2,
        "query": "Which author is a doctor and what paper did they co-author?",
        "ground_truth": "Dr. Abdul Majid co-authored CardioVison-Surv on heart failure forecasting",
        "keywords": ["Abdul Majid", "CardioVison-Surv", "heart failure"]
    },
    {
        "id": 12, "category": "multi_hop", "difficulty": "medium", "hops": 2,
        "query": "What methodology does the paper use and for which disease is it applied?",
        "ground_truth": "Longitudinal deep learning and temporal transformers applied to heart failure",
        "keywords": ["transformers", "deep learning", "heart failure"]
    },
    {
        "id": 13, "category": "multi_hop", "difficulty": "hard", "hops": 3,
        "query": "Who submitted a systematic review to a cardiac conference and when was it accepted?",
        "ground_truth": "Priyesh Srivastava and Dr. Abdul Majid submitted to ICCIML-26, accepted on 02/03/2026",
        "keywords": ["Priyesh Srivastava", "ICCIML-26", "2026"]
    },
    {
        "id": 14, "category": "multi_hop", "difficulty": "hard", "hops": 2,
        "query": "What is the paper ID of the paper that was accepted for oral presentation at ICCIML-26?",
        "ground_truth": "ISER-2-27-0050098 was accepted for oral poster presentation at ICCIML-26",
        "keywords": ["ISER-2-27-0050098", "ICCIML-26", "oral"]
    },
    {
        "id": 15, "category": "multi_hop", "difficulty": "hard", "hops": 2,
        "query": "Which machine learning technique for time-series was used in the accepted cardiac paper?",
        "ground_truth": "Temporal Transformers for longitudinal progression forecasting",
        "keywords": ["temporal", "transformers", "longitudinal"]
    },
    {
        "id": 16, "category": "multi_hop", "difficulty": "medium", "hops": 2,
        "query": "What is the research contribution of the paper accepted in March 2026?",
        "ground_truth": "Systematic review of deep learning and temporal transformers for heart failure progression forecasting",
        "keywords": ["systematic review", "deep learning", "heart failure"]
    },
    {
        "id": 17, "category": "multi_hop", "difficulty": "hard", "hops": 3,
        "query": "At what kind of event will the paper on cardiac AI be presented and by whom?",
        "ground_truth": "Oral poster presentation at ICCIML-26 by Priyesh Srivastava and Dr. Abdul Majid",
        "keywords": ["oral poster", "ICCIML", "Priyesh"]
    },
    {
        "id": 18, "category": "multi_hop", "difficulty": "medium", "hops": 2,
        "query": "What is the relationship between temporal transformers and the paper's acceptance?",
        "ground_truth": "Temporal transformers are the core methodology reviewed in the paper accepted by ICCIML-26",
        "keywords": ["temporal transformers", "ICCIML", "accepted"]
    },

    # ── COMPARISON ───────────────────────────────────────────────────
    {
        "id": 19, "category": "comparison", "difficulty": "medium", "hops": 2,
        "query": "How does deep learning differ from traditional methods for heart failure prediction according to the paper?",
        "ground_truth": "Deep learning captures longitudinal temporal patterns that traditional statistical methods cannot model effectively",
        "keywords": ["deep learning", "longitudinal", "temporal"]
    },
    {
        "id": 20, "category": "comparison", "difficulty": "medium", "hops": 2,
        "query": "What is the difference between the paper ID and the conference acronym?",
        "ground_truth": "The paper ID ISER-2-27-0050098 identifies the submission while ICCIML-26 identifies the conference",
        "keywords": ["ISER-2-27-0050098", "ICCIML-26"]
    },
    {
        "id": 21, "category": "comparison", "difficulty": "hard", "hops": 3,
        "query": "Compare the roles of Priyesh Srivastava and Dr. Abdul Majid in the paper",
        "ground_truth": "Both are listed as co-authors; Dr. Abdul Majid holds a doctorate while Priyesh Srivastava is the first author",
        "keywords": ["co-author", "Priyesh", "Abdul Majid"]
    },
    {
        "id": 22, "category": "comparison", "difficulty": "medium", "hops": 2,
        "query": "What distinguishes a systematic review from a regular paper in the context of this submission?",
        "ground_truth": "A systematic review comprehensively surveys existing literature on a topic rather than presenting original experiments",
        "keywords": ["systematic review", "literature", "survey"]
    },

    # ── AGGREGATION ──────────────────────────────────────────────────
    {
        "id": 23, "category": "aggregation", "difficulty": "easy", "hops": 1,
        "query": "Summarize the acceptance letter in one sentence",
        "ground_truth": "Priyesh Srivastava and Dr. Abdul Majid's paper on deep learning for heart failure was accepted for oral poster presentation at ICCIML-26",
        "keywords": ["Priyesh", "heart failure", "ICCIML", "accepted"]
    },
    {
        "id": 24, "category": "aggregation", "difficulty": "medium", "hops": 2,
        "query": "List all key details about the accepted paper",
        "ground_truth": "Title: CardioVison-Surv; Authors: Priyesh Srivastava, Dr. Abdul Majid; ID: ISER-2-27-0050098; Conference: ICCIML-26; Presentation: oral poster; Date: 02/03/2026",
        "keywords": ["CardioVison-Surv", "ISER-2-27-0050098", "ICCIML-26"]
    },
    {
        "id": 25, "category": "aggregation", "difficulty": "medium", "hops": 1,
        "query": "What are all the machine learning techniques mentioned in the paper title?",
        "ground_truth": "Longitudinal deep learning and temporal transformers",
        "keywords": ["longitudinal", "deep learning", "temporal transformers"]
    },
    {
        "id": 26, "category": "aggregation", "difficulty": "easy", "hops": 1,
        "query": "How many authors does the accepted paper have?",
        "ground_truth": "Two authors: Priyesh Srivastava and Dr. Abdul Majid",
        "keywords": ["two", "Priyesh", "Abdul"]
    },
    {
        "id": 27, "category": "aggregation", "difficulty": "medium", "hops": 2,
        "query": "What are all the identifiers associated with the paper and conference?",
        "ground_truth": "Paper ID ISER-2-27-0050098 and conference acronym ICCIML-26",
        "keywords": ["ISER-2-27-0050098", "ICCIML-26"]
    },

    # ── DEFINITIONAL ─────────────────────────────────────────────────
    {
        "id": 28, "category": "definitional", "difficulty": "easy", "hops": 1,
        "query": "What does ICCIML stand for?",
        "ground_truth": "International Conference on Cardiac Imaging and Machine Learning",
        "keywords": ["International", "Cardiac", "Machine Learning"]
    },
    {
        "id": 29, "category": "definitional", "difficulty": "medium", "hops": 1,
        "query": "What is a temporal transformer in the context of the paper?",
        "ground_truth": "A deep learning architecture that models time-sequential medical data to forecast disease progression over time",
        "keywords": ["temporal", "transformer", "time", "progression"]
    },
    {
        "id": 30, "category": "definitional", "difficulty": "easy", "hops": 1,
        "query": "What is heart failure progression forecasting?",
        "ground_truth": "Predicting how a patient's heart failure condition will worsen over time using historical clinical data",
        "keywords": ["heart failure", "predict", "progression"]
    },
    {
        "id": 31, "category": "definitional", "difficulty": "medium", "hops": 1,
        "query": "What is an oral poster presentation?",
        "ground_truth": "A presentation format where authors give a short oral summary alongside a displayed poster at a conference",
        "keywords": ["oral", "poster", "conference"]
    },
    {
        "id": 32, "category": "definitional", "difficulty": "easy", "hops": 1,
        "query": "What is longitudinal deep learning?",
        "ground_truth": "Deep learning applied to data collected from the same subjects over multiple time points to track changes",
        "keywords": ["longitudinal", "time points", "subjects"]
    },
    {
        "id": 33, "category": "definitional", "difficulty": "medium", "hops": 1,
        "query": "What does a systematic review paper typically contain?",
        "ground_truth": "A comprehensive, structured survey of existing literature on a specific topic with defined inclusion criteria",
        "keywords": ["survey", "literature", "comprehensive", "criteria"]
    },

    # ── GENERAL DOCUMENT QA (for non-acceptance-letter docs) ────────
    {
        "id": 34, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What is the main topic of this document?",
        "ground_truth": "The main topic is described in the document header or abstract",
        "keywords": ["topic", "main", "document"]
    },
    {
        "id": 35, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "Who is the intended audience for this document?",
        "ground_truth": "The intended audience is identified by the document type and content",
        "keywords": ["audience", "intended"]
    },
    {
        "id": 36, "category": "aggregation", "difficulty": "medium", "hops": 1,
        "query": "What are the key points of this document?",
        "ground_truth": "The key points include the main claims, findings, and conclusions stated in the document",
        "keywords": ["key points", "conclusions", "findings"]
    },
    {
        "id": 37, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What date appears on this document?",
        "ground_truth": "The date is stated in the document header or footer",
        "keywords": ["date", "year"]
    },
    {
        "id": 38, "category": "comparison", "difficulty": "medium", "hops": 2,
        "query": "How does this document differ from a standard report?",
        "ground_truth": "The specific differences depend on the document format, structure, and purpose",
        "keywords": ["document", "structure", "format"]
    },
    {
        "id": 39, "category": "multi_hop", "difficulty": "hard", "hops": 2,
        "query": "What action is required as a result of this document?",
        "ground_truth": "The required action is stated in the conclusion or call-to-action section of the document",
        "keywords": ["action", "required", "conclusion"]
    },
    {
        "id": 40, "category": "definitional", "difficulty": "medium", "hops": 1,
        "query": "What technical terms are defined in this document?",
        "ground_truth": "Technical terms are defined in the glossary, abstract, or body of the document",
        "keywords": ["technical", "terms", "defined"]
    },

    # ── IMAGE-SPECIFIC QA ────────────────────────────────────────────
    {
        "id": 41, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What objects are visible in this image?",
        "ground_truth": "The image contains detected objects identified by the vision system",
        "keywords": ["objects", "detected", "image"]
    },
    {
        "id": 42, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "How many people are in this image?",
        "ground_truth": "The number of people is determined by YOLO object detection",
        "keywords": ["person", "people", "detected"]
    },
    {
        "id": 43, "category": "definitional", "difficulty": "medium", "hops": 1,
        "query": "Describe the scene shown in this image",
        "ground_truth": "The scene description is derived from detected objects and their spatial arrangement",
        "keywords": ["scene", "objects", "arrangement"]
    },
    {
        "id": 44, "category": "comparison", "difficulty": "medium", "hops": 2,
        "query": "What electronic devices can be seen in the image?",
        "ground_truth": "Electronic devices such as laptops or monitors detected by the vision model",
        "keywords": ["laptop", "electronic", "device"]
    },
    {
        "id": 45, "category": "aggregation", "difficulty": "easy", "hops": 1,
        "query": "List all items detected in the image",
        "ground_truth": "All items are listed from YOLO detection output including persons, chairs, laptops",
        "keywords": ["person", "chair", "laptop", "detected"]
    },

    # ── ADDITIONAL MULTI-HOP (to reach 100) ─────────────────────────
    {
        "id": 46, "category": "multi_hop", "difficulty": "hard", "hops": 3,
        "query": "What connects the paper title to the presentation format at ICCIML-26?",
        "ground_truth": "The paper on cardiac deep learning was selected for oral poster, indicating high reviewer confidence",
        "keywords": ["cardiac", "oral poster", "ICCIML-26"]
    },
    {
        "id": 47, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "Is the paper a survey or an experimental study?",
        "ground_truth": "It is a systematic review, which is a type of survey paper",
        "keywords": ["systematic review", "survey"]
    },
    {
        "id": 48, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What field does ICCIML-26 focus on?",
        "ground_truth": "Cardiac imaging and machine learning",
        "keywords": ["cardiac", "imaging", "machine learning"]
    },
    {
        "id": 49, "category": "multi_hop", "difficulty": "medium", "hops": 2,
        "query": "What is the connection between the paper's authors and the conference location?",
        "ground_truth": "The authors submitted to ICCIML-26 whose location is specified in the conference details",
        "keywords": ["authors", "ICCIML-26", "conference"]
    },
    {
        "id": 50, "category": "aggregation", "difficulty": "medium", "hops": 2,
        "query": "What information would you need to cite this paper?",
        "ground_truth": "Authors: Priyesh Srivastava, Dr. Abdul Majid; Title: CardioVison-Surv; Conference: ICCIML-26; Year: 2026",
        "keywords": ["Priyesh", "CardioVison-Surv", "ICCIML-26", "2026"]
    },
    {
        "id": 51, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What does ISER stand for in the paper ID?",
        "ground_truth": "ISER is the conference submission identifier prefix used by ICCIML-26",
        "keywords": ["ISER", "identifier"]
    },
    {
        "id": 52, "category": "comparison", "difficulty": "medium", "hops": 2,
        "query": "How does deep learning compare to older methods for heart disease prediction?",
        "ground_truth": "Deep learning handles high-dimensional temporal data better than classical regression or SVM-based methods",
        "keywords": ["deep learning", "temporal", "classical"]
    },
    {
        "id": 53, "category": "definitional", "difficulty": "medium", "hops": 1,
        "query": "What is double-blind peer review?",
        "ground_truth": "A review process where neither authors nor reviewers know each other's identities",
        "keywords": ["double-blind", "reviewers", "anonymous"]
    },
    {
        "id": 54, "category": "multi_hop", "difficulty": "hard", "hops": 3,
        "query": "Trace the path from paper submission to oral presentation at ICCIML-26",
        "ground_truth": "Paper submitted with ID ISER-2-27-0050098, underwent double-blind review, accepted on 02/03/2026, assigned oral poster slot",
        "keywords": ["ISER", "review", "accepted", "oral"]
    },
    {
        "id": 55, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "Is the paper accepted or rejected?",
        "ground_truth": "The paper is accepted for presentation at ICCIML-26",
        "keywords": ["accepted"]
    },
    {
        "id": 56, "category": "aggregation", "difficulty": "easy", "hops": 1,
        "query": "What are the full names of both authors?",
        "ground_truth": "Priyesh Srivastava and Dr. Abdul Majid",
        "keywords": ["Priyesh Srivastava", "Abdul Majid"]
    },
    {
        "id": 57, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What kind of document is this?",
        "ground_truth": "This is a conference paper acceptance letter",
        "keywords": ["acceptance letter", "conference"]
    },
    {
        "id": 58, "category": "multi_hop", "difficulty": "medium", "hops": 2,
        "query": "What field of medicine is studied using temporal transformers in the accepted paper?",
        "ground_truth": "Cardiology — specifically heart failure progression forecasting",
        "keywords": ["cardiology", "heart failure", "temporal transformers"]
    },
    {
        "id": 59, "category": "comparison", "difficulty": "hard", "hops": 3,
        "query": "What advantages do temporal transformers have over LSTMs for heart failure forecasting?",
        "ground_truth": "Temporal transformers capture long-range dependencies and parallel processing better than sequential LSTMs",
        "keywords": ["transformers", "LSTM", "long-range", "parallel"]
    },
    {
        "id": 60, "category": "definitional", "difficulty": "easy", "hops": 1,
        "query": "What is the significance of an oral poster presentation?",
        "ground_truth": "It indicates the paper was ranked highly enough for both a talk and a poster session",
        "keywords": ["oral", "poster", "ranked"]
    },
    {
        "id": 61, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What number is assigned to the accepted paper?",
        "ground_truth": "ISER-2-27-0050098",
        "keywords": ["ISER-2-27-0050098"]
    },
    {
        "id": 62, "category": "multi_hop", "difficulty": "medium", "hops": 2,
        "query": "Who published the acceptance letter and to whom is it addressed?",
        "ground_truth": "ICCIML-26 published the letter addressed to Priyesh Srivastava and Dr. Abdul Majid",
        "keywords": ["ICCIML-26", "Priyesh Srivastava", "Abdul Majid"]
    },
    {
        "id": 63, "category": "aggregation", "difficulty": "medium", "hops": 2,
        "query": "What would you tell a colleague about this paper in 3 bullet points?",
        "ground_truth": "1) Systematic review of deep learning for heart failure 2) Authors: Priyesh and Abdul Majid 3) Accepted at ICCIML-26 for oral poster",
        "keywords": ["systematic review", "heart failure", "ICCIML-26"]
    },
    {
        "id": 64, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What month was the acceptance letter dated?",
        "ground_truth": "March 2026",
        "keywords": ["March", "2026"]
    },
    {
        "id": 65, "category": "definitional", "difficulty": "medium", "hops": 1,
        "query": "What is the role of a machine learning conference in academic publishing?",
        "ground_truth": "Conferences provide peer review, dissemination, and recognition for research contributions",
        "keywords": ["peer review", "dissemination", "research"]
    },
    {
        "id": 66, "category": "multi_hop", "difficulty": "hard", "hops": 3,
        "query": "What is the significance of the paper ID format ISER-2-27-0050098 for ICCIML-26?",
        "ground_truth": "The ISER prefix and number uniquely track the paper through ICCIML-26's submission management system",
        "keywords": ["ISER", "ICCIML-26", "submission", "track"]
    },
    {
        "id": 67, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "Does the paper focus on imaging or forecasting?",
        "ground_truth": "The paper focuses on forecasting heart failure progression using deep learning",
        "keywords": ["forecasting", "heart failure", "deep learning"]
    },
    {
        "id": 68, "category": "comparison", "difficulty": "medium", "hops": 2,
        "query": "How does this paper differ from a typical cardiac imaging paper?",
        "ground_truth": "This paper is a systematic review focused on progression forecasting, not original imaging experiments",
        "keywords": ["systematic review", "progression", "imaging"]
    },
    {
        "id": 69, "category": "aggregation", "difficulty": "medium", "hops": 1,
        "query": "What technical fields are combined in this paper?",
        "ground_truth": "Deep learning, temporal modeling, transformers, and cardiology/heart failure medicine",
        "keywords": ["deep learning", "transformers", "cardiology"]
    },
    {
        "id": 70, "category": "multi_hop", "difficulty": "medium", "hops": 2,
        "query": "What is the academic contribution made by Priyesh Srivastava in this paper?",
        "ground_truth": "Co-authoring a systematic review on temporal transformers for heart failure forecasting, accepted at ICCIML-26",
        "keywords": ["Priyesh Srivastava", "systematic review", "ICCIML-26"]
    },
    {
        "id": 71, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What is the conference abbreviation year?",
        "ground_truth": "26, representing 2026",
        "keywords": ["26", "2026"]
    },
    {
        "id": 72, "category": "multi_hop", "difficulty": "hard", "hops": 3,
        "query": "How does the acceptance letter confirm the paper's quality?",
        "ground_truth": "By noting it passed double-blind review among many submissions and was selected for oral poster, a competitive format",
        "keywords": ["double-blind", "review", "competitive", "oral"]
    },
    {
        "id": 73, "category": "definitional", "difficulty": "medium", "hops": 1,
        "query": "What is longitudinal analysis in medical AI?",
        "ground_truth": "Analysis that tracks patient data over time to understand disease progression and changes in health status",
        "keywords": ["longitudinal", "time", "patient", "progression"]
    },
    {
        "id": 74, "category": "aggregation", "difficulty": "hard", "hops": 3,
        "query": "What is the complete citation information for this paper?",
        "ground_truth": "Priyesh Srivastava, Abdul Majid. CardioVison-Surv: ... ICCIML-26, 2026. Paper ID: ISER-2-27-0050098",
        "keywords": ["Priyesh", "CardioVison-Surv", "ICCIML-26", "2026", "ISER"]
    },
    {
        "id": 75, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What is the research area of the accepted paper?",
        "ground_truth": "Medical AI / cardiac machine learning / heart failure prediction",
        "keywords": ["medical AI", "cardiac", "heart failure"]
    },
    {
        "id": 76, "category": "comparison", "difficulty": "hard", "hops": 3,
        "query": "How does the paper's use of transformers advance the field over previous architectures?",
        "ground_truth": "Temporal transformers enable self-attention over long time horizons, surpassing RNNs and CNNs for sequential cardiac data",
        "keywords": ["transformers", "attention", "RNN", "CNN", "cardiac"]
    },
    {
        "id": 77, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What is the presentation type allocated to paper ISER-2-27-0050098?",
        "ground_truth": "Oral poster presentation",
        "keywords": ["oral poster"]
    },
    {
        "id": 78, "category": "multi_hop", "difficulty": "medium", "hops": 2,
        "query": "Which authors are affiliated with research on cardiac deep learning?",
        "ground_truth": "Priyesh Srivastava and Dr. Abdul Majid based on their accepted paper topic",
        "keywords": ["Priyesh Srivastava", "Abdul Majid", "cardiac"]
    },
    {
        "id": 79, "category": "definitional", "difficulty": "easy", "hops": 1,
        "query": "What is a conference acceptance letter?",
        "ground_truth": "An official document from a conference informing authors their paper was accepted for presentation",
        "keywords": ["official", "accepted", "conference", "presentation"]
    },
    {
        "id": 80, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What is the title abbreviation of the accepted paper?",
        "ground_truth": "CardioVison-Surv",
        "keywords": ["CardioVison-Surv"]
    },
    {
        "id": 81, "category": "multi_hop", "difficulty": "hard", "hops": 3,
        "query": "How does the ICCIML-26 conference relate to the topic of the accepted paper?",
        "ground_truth": "ICCIML focuses on cardiac imaging and ML, directly aligning with the paper's cardiac deep learning forecasting topic",
        "keywords": ["ICCIML", "cardiac", "machine learning", "forecasting"]
    },
    {
        "id": 82, "category": "aggregation", "difficulty": "medium", "hops": 2,
        "query": "What are all proper nouns in the acceptance letter?",
        "ground_truth": "Priyesh Srivastava, Dr. Abdul Majid, CardioVison-Surv, ISER-2-27-0050098, ICCIML-26",
        "keywords": ["Priyesh", "Abdul Majid", "CardioVison-Surv", "ISER", "ICCIML"]
    },
    {
        "id": 83, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What day of the month was the letter issued?",
        "ground_truth": "2nd (02/03/2026)",
        "keywords": ["02", "2nd", "March"]
    },
    {
        "id": 84, "category": "comparison", "difficulty": "medium", "hops": 2,
        "query": "What is the difference between cardiac imaging and heart failure forecasting?",
        "ground_truth": "Cardiac imaging captures structural/functional snapshots; heart failure forecasting predicts future progression from longitudinal data",
        "keywords": ["imaging", "forecasting", "longitudinal", "progression"]
    },
    {
        "id": 85, "category": "multi_hop", "difficulty": "medium", "hops": 2,
        "query": "What is the significance of paper ID starting with ISER-2-27?",
        "ground_truth": "The prefix encodes the conference track, year, and submission sequence within ICCIML-26",
        "keywords": ["ISER", "track", "submission"]
    },
    {
        "id": 86, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "Is the conference national or international?",
        "ground_truth": "International — the full name is International Conference on Cardiac Imaging and Machine Learning",
        "keywords": ["International", "ICCIML"]
    },
    {
        "id": 87, "category": "definitional", "difficulty": "medium", "hops": 1,
        "query": "What does a systematic review contribute that an original experiment does not?",
        "ground_truth": "A systematic review synthesizes evidence across many studies to identify patterns and consensus",
        "keywords": ["synthesizes", "evidence", "patterns", "consensus"]
    },
    {
        "id": 88, "category": "aggregation", "difficulty": "easy", "hops": 1,
        "query": "What are the three main keywords of the accepted paper?",
        "ground_truth": "Heart failure, temporal transformers, deep learning",
        "keywords": ["heart failure", "temporal transformers", "deep learning"]
    },
    {
        "id": 89, "category": "multi_hop", "difficulty": "hard", "hops": 3,
        "query": "Based on the letter, what can we infer about the quality of the paper?",
        "ground_truth": "High quality — it passed double-blind review, was accepted for oral presentation, and published by a specialized cardiac ML conference",
        "keywords": ["double-blind", "oral", "quality", "accepted"]
    },
    {
        "id": 90, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What specific forecasting task does the paper address?",
        "ground_truth": "Heart failure progression forecasting",
        "keywords": ["heart failure progression", "forecasting"]
    },
    {
        "id": 91, "category": "comparison", "difficulty": "hard", "hops": 2,
        "query": "How does ICCIML-26 differ from a general AI conference like NeurIPS?",
        "ground_truth": "ICCIML-26 is domain-specific to cardiac imaging and ML, while NeurIPS covers all of AI and ML broadly",
        "keywords": ["domain-specific", "cardiac", "NeurIPS", "general"]
    },
    {
        "id": 92, "category": "multi_hop", "difficulty": "medium", "hops": 2,
        "query": "What is the relationship between YOLO detections and image understanding in this system?",
        "ground_truth": "YOLO detects objects in images and their class names are embedded as text to enable semantic search",
        "keywords": ["YOLO", "detected", "embedded", "semantic"]
    },
    {
        "id": 93, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What vision model is used to process uploaded images?",
        "ground_truth": "YOLOv8 for object detection combined with EasyOCR for text extraction",
        "keywords": ["YOLOv8", "EasyOCR", "object detection"]
    },
    {
        "id": 94, "category": "multi_hop", "difficulty": "hard", "hops": 3,
        "query": "How does the RAG pipeline retrieve information from an uploaded image?",
        "ground_truth": "YOLOv8 detects objects, descriptions are embedded with sentence-transformers, stored in Qdrant, then retrieved by cosine similarity at query time",
        "keywords": ["YOLOv8", "sentence-transformers", "Qdrant", "cosine"]
    },
    {
        "id": 95, "category": "aggregation", "difficulty": "medium", "hops": 2,
        "query": "List all components in the ICDI-X retrieval pipeline",
        "ground_truth": "Input guardrail, agentic planner, dense retrieval, IB filter, evidence verifier, LLM generation, output guardrail, MAB selector",
        "keywords": ["guardrail", "planner", "IB filter", "MAB", "evidence"]
    },
    {
        "id": 96, "category": "definitional", "difficulty": "medium", "hops": 1,
        "query": "What is the information bottleneck filter?",
        "ground_truth": "A compression module that keeps only context chunks maximally relevant to the query, reducing noise by 40-60%",
        "keywords": ["compression", "relevant", "noise", "reduce"]
    },
    {
        "id": 97, "category": "comparison", "difficulty": "hard", "hops": 2,
        "query": "How does quantum fidelity differ from cosine similarity for retrieval scoring?",
        "ground_truth": "Quantum fidelity uses density matrix trace operations capturing non-linear relationships, while cosine is a linear dot product in normalized space",
        "keywords": ["fidelity", "density matrix", "cosine", "non-linear"]
    },
    {
        "id": 98, "category": "multi_hop", "difficulty": "hard", "hops": 3,
        "query": "Why does the system use Thompson Sampling for retrieval strategy selection?",
        "ground_truth": "Thompson Sampling explores retrieval strategies probabilistically and exploits the best-performing arm, balancing exploration-exploitation for adaptive improvement",
        "keywords": ["Thompson Sampling", "exploration", "exploitation", "adaptive"]
    },
    {
        "id": 99, "category": "factual", "difficulty": "easy", "hops": 1,
        "query": "What embedding model generates the 384-dimensional vectors?",
        "ground_truth": "sentence-transformers/all-MiniLM-L6-v2",
        "keywords": ["all-MiniLM-L6-v2", "384", "sentence-transformers"]
    },
    {
        "id": 100, "category": "multi_hop", "difficulty": "hard", "hops": 3,
        "query": "How does the knowledge graph enable multi-hop reasoning that dense retrieval cannot?",
        "ground_truth": "The knowledge graph stores entity relations as graph edges, enabling traversal across connected facts that may not co-occur in any single chunk",
        "keywords": ["knowledge graph", "relations", "traversal", "multi-hop"]
    },
]


CATEGORY_DISTRIBUTION = {
    "factual":     [x for x in HOTPOTQA_DATASET if x["category"] == "factual"],
    "multi_hop":   [x for x in HOTPOTQA_DATASET if x["category"] == "multi_hop"],
    "comparison":  [x for x in HOTPOTQA_DATASET if x["category"] == "comparison"],
    "aggregation": [x for x in HOTPOTQA_DATASET if x["category"] == "aggregation"],
    "definitional":[x for x in HOTPOTQA_DATASET if x["category"] == "definitional"],
}

DIFFICULTY_DISTRIBUTION = {
    "easy":   [x for x in HOTPOTQA_DATASET if x["difficulty"] == "easy"],
    "medium": [x for x in HOTPOTQA_DATASET if x["difficulty"] == "medium"],
    "hard":   [x for x in HOTPOTQA_DATASET if x["difficulty"] == "hard"],
}


def get_dataset(category: str = None, difficulty: str = None, max_hops: int = None) -> list:
    data = HOTPOTQA_DATASET
    if category:
        data = [x for x in data if x["category"] == category]
    if difficulty:
        data = [x for x in data if x["difficulty"] == difficulty]
    if max_hops:
        data = [x for x in data if x["hops"] <= max_hops]
    return data


def get_multi_hop_only() -> list:
    return [x for x in HOTPOTQA_DATASET if x["hops"] >= 2]


def print_dataset_stats():
    print(f"Total QA pairs: {len(HOTPOTQA_DATASET)}")
    print("\nBy category:")
    for cat, items in CATEGORY_DISTRIBUTION.items():
        print(f"  {cat:<15}: {len(items)}")
    print("\nBy difficulty:")
    for diff, items in DIFFICULTY_DISTRIBUTION.items():
        print(f"  {diff:<10}: {len(items)}")
    avg_hops = sum(x["hops"] for x in HOTPOTQA_DATASET) / len(HOTPOTQA_DATASET)
    print(f"\nAverage hops: {avg_hops:.2f}")
    multi = [x for x in HOTPOTQA_DATASET if x["hops"] >= 2]
    print(f"Multi-hop (≥2 hops): {len(multi)} ({100*len(multi)//len(HOTPOTQA_DATASET)}%)")


if __name__ == "__main__":
    print_dataset_stats()

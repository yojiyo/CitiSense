# CitiSense 🏛️📊

> A Web App in Enhancing the Regional Government Agencies Feedback Systems in Pampanga through Facebook Sentiment Analysis with Decision Support Dashboard

Welcome to the repository for **CitiSense**, an academic thesis project completed in May 2026 by Mark Gio G. Alcuizar, Haidee Adreanne F. Duarte, Angeline L. Ruin, and John Ryan P. Trinidad at Angeles University Foundation.

---

## 📸 Dashboard Preview

<img src="CitiSense Dashboard Preview.png" alt="Logo" width="100%">



---

## ✨ Project Overview

Regional government agencies in Pampanga (DOH, DPWH, DOLE, and DSWD) often rely on slow, traditional feedback systems. CitiSense addresses this gap by leveraging machine learning to systematically analyze public Facebook comments related to agency services. By translating unstructured, code-mixed Filipino-English (Taglish) feedback into actionable intelligence, the system supports accountable and responsive governance in Region III.

## 🚀 Core Features

*   **Automated Sentiment Classification:** Utilizes a fine-tuned Multilingual BERT (mBERT) model to categorize citizen feedback into positive, negative, or neutral sentiments.
*   **Decision Support Dashboard:** A web application featuring sentiment distribution pie charts, weekly sentiment trend line graphs, and word clouds to aid stakeholders in decision-making.
*   **Data Management & Reporting:** Allows administrators to upload CSV datasets, preview data, and securely download customizable PDF or CSV reports with AI-generated insights.
*   **Clean Interface:** The frontend is built to maintain an uncluttered, minimalist layout with low visual weight and neutral tones, ensuring complex public sentiment data is presented clearly and without cognitive overload.

## 🧠 Model Performance & Methodology

The project evaluated four distinct machine learning algorithms: Random Forest, Multinomial Naive Bayes, Support Vector Machine (SVM), and a fine-tuned mBERT.

*   The **mBERT** model emerged as the most robust, achieving an overall accuracy of **82%**.
*   It demonstrated superior capability in capturing context and handling the nuances of code-mixed Taglish, sarcasm, and ambiguity.
*   In comparison, traditional models achieved lower accuracies: Random Forest (78%), SVM (77%), and Multinomial Naive Bayes (74%).

## 🛠️ Technology Stack

*   **Frontend:** HTML, CSS
*   **Backend / Machine Learning:** Python
*   **NLP & Deep Learning Libraries:** NLTK, Scikit-learn, Ktrain, PyTorch, Transformers, TensorFlow
*   **Data Processing:** Pandas, NumPy
*   **Database:** SQLite

## 📈 Project Impact

The CitiSense web application was evaluated by domain experts and stakeholders using the System Usability Scale (SUS) and ISO 25010 standards. The platform effectively aids agencies in prioritizing public issues and proactive management by instantly identifying rising negative sentiments, successfully bridging the communication gap between the community and local government.

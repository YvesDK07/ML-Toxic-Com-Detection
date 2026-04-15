import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

df1 = pd.read_csv('train.csv.zip')
df2 = pd.read_csv('youtoxic_english_1000.csv')

df1['Toxic'] = df1.iloc[:, 2:].any(axis=1)
df1_processed = df1[['comment_text', 'Toxic']].rename(columns={'comment_text': 'Text'})

df2['Toxic'] = df2.iloc[:, 3:].any(axis=1)
df2_processed = df2[['Text', 'Toxic']]

df = pd.concat([df1_processed, df2_processed], ignore_index=True)

print(df.head())
df.describe()

print(df.dtypes)
print(df.isnull().sum())

duplicate_rows = df[df.duplicated(subset=['Text'], keep=False)]
print("Duplicate rows based on 'Text' column:")
print(duplicate_rows)

df.drop_duplicates(subset=['Text'], keep='first', inplace=True)
print("Number of rows after removing duplicates:", len(df))
df.reset_index(drop=True, inplace=True)

toxic_distribution = df['Toxic'].value_counts()
print(toxic_distribution)

plt.figure(figsize=(8, 6))
toxic_counts = df['Toxic'].value_counts()
toxic_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Toxic vs Non-Toxic Comments')
plt.xlabel('Toxic')
plt.ylabel('Count')
plt.xticks(rotation=0)

toxic_comments = ' '.join(df[df['Toxic']]['Text'])
wordcloud_toxic = WordCloud(width=800, height=400, background_color='white').generate(toxic_comments)
plt.imshow(wordcloud_toxic, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Toxic Comments')


non_toxic_comments = ' '.join(df[~df['Toxic']]['Text'])
wordcloud_non_toxic = WordCloud(width=800, height=400, background_color='white').generate(non_toxic_comments)
plt.imshow(wordcloud_non_toxic, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Non-Toxic Comments')


# ============================================
# 2. TEXT CLEANING
# ============================================

df['Toxic'] = df['Toxic'].astype(int)

def clean_text(text):
    
    text = text.lower()

    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub(r"f\*+ing", "fucking", text)  
    text = re.sub(r"f\*+k", "fuck", text)       
    text = re.sub(r"sh\*+t", "shit", text)     
    text = re.sub(r"b\*+tch", "bitch", text) 
    
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    
    return text

df['Text'] = df['Text'].map(clean_text)

print("\n=== CLEANED TEXT ===")
print(df.head())

# ============================================
# 3. TF-IDF VECTORIZATION
# ============================================

vect = TfidfVectorizer(
    max_features=5000, 
    stop_words='english', 
    ngram_range=(1, 2), 
    min_df=2,            
    max_df=0.95         
)
X = vect.fit_transform(df['Text'])
Y = df['Toxic']

# ============================================
# 4. CLASS BALANCING WITH SMOTE
# ============================================

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, Y)

df_resampled = pd.DataFrame(X_resampled.todense(), columns=vect.get_feature_names_out())
df_resampled['Toxic'] = y_resampled

toxic_distribution_after = df_resampled['Toxic'].value_counts()
print("\nDistribution after SMOTE:")
print(toxic_distribution_after)

plt.figure(figsize=(8, 6))
toxic_counts = df_resampled['Toxic'].value_counts()
toxic_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Toxic vs Non-Toxic Comments (After SMOTE)')
plt.xlabel('Toxic')
plt.ylabel('Count')
plt.xticks(rotation=0)


# ============================================
# 5. DATA SPLIT (80-20)
# ============================================

x_train, x_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled  
)

print(f"\nTraining size: {x_train.shape}")
print(f"Testing size: {x_test.shape}")

# ============================================
# 6. MODEL TRAINING AND COMPARISON
# ============================================

print("\n" + "="*50)
print("MODEL TRAINING (5 ALGORITHMS + NEURAL NETWORK)")
print("="*50)

models_performance = {}

# --- 6.1 XGBOOST (OPTIMISÉ) ---
print("\n[1/5] Training XGBoost OPTIMIZED...")

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,          
    min_child_weight=1,          
    gamma=0,                    
    subsample=0.8,
    scale_pos_weight=1,          
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(x_train, y_train)
xgb_pred = xgb_model.predict(x_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
models_performance['XGBoost'] = xgb_accuracy
print(f"XGBoost Accuracy: {xgb_accuracy:.4f} ({xgb_accuracy*100:.2f}%)")
print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_pred))

# --- 6.2 RANDOM FOREST ---
print("\n[2/5] Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
models_performance['Random Forest'] = rf_accuracy
print(f"Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")

# --- 6.3 LINEAR SVM ---
print("\n[3/5] Training Linear SVC...")
svm_linear_model = LinearSVC(max_iter=1000, random_state=42)
svm_linear_model.fit(x_train, y_train)
svm_linear_pred = svm_linear_model.predict(x_test)
svm_linear_accuracy = accuracy_score(y_test, svm_linear_pred)
models_performance['Linear SVC'] = svm_linear_accuracy
print(f"Linear SVM Accuracy: {svm_linear_accuracy:.4f} ({svm_linear_accuracy*100:.2f}%)")

# --- 6.4 DECISION TREE ---
print("\n[4/5] Training Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(x_train, y_train)
dt_pred = dt_model.predict(x_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
models_performance['Decision Tree'] = dt_accuracy
print(f"Decision Tree Accuracy: {dt_accuracy:.4f} ({dt_accuracy*100:.2f}%)")

# --- 6.5 NEURAL NETWORK ---
print("\n[5/5] Training Neural Network (MLP)...")
scaler = StandardScaler(with_mean=False)
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

nn_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=50,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=False
)
nn_model.fit(x_train_scaled, y_train)
nn_pred = nn_model.predict(x_test_scaled)
nn_accuracy = accuracy_score(y_test, nn_pred)
models_performance['Neural Network'] = nn_accuracy
print(f"Neural Network Accuracy: {nn_accuracy:.4f} ({nn_accuracy*100:.2f}%)")
print("\nNeural Network Classification Report:")
print(classification_report(y_test, nn_pred))

# ============================================
# 6.6 ENSEMBLE MODEL (XGBoost + LinearSVC + MLP)
# ============================================

print("\n" + "="*50)
print("MODEL Ensemble (XGBoost + LinearSVC + MLP)")
print("="*50)

print("\nCreation of the overall model with Soft Voting...")

svc_calibrated = CalibratedClassifierCV(svm_linear_model, cv=3, method='sigmoid')

mlp_pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),
    ('mlp', MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    ))
])

ensemble_model = VotingClassifier(
    estimators=[
        ('xgboost', xgb_model),
        ('svc', svc_calibrated),
        ('mlp', mlp_pipeline)
    ],
    voting='soft',  
    weights=[1, 1, 2]
)

print("Training the ensemble model...")
ensemble_model.fit(x_train, y_train)

ensemble_pred = ensemble_model.predict(x_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
models_performance['Ensemble (XGB+SVC+MLP)'] = ensemble_accuracy

print(f"\n✅ Model Ensemble Accuracy: {ensemble_accuracy:.4f} ({ensemble_accuracy*100:.2f}%)")
print("\nModel Ensemble Classification Report:")
print(classification_report(y_test, ensemble_pred))

print("\nConfusion Matrix - Model Ensemble:")
cm_ensemble = confusion_matrix(y_test, ensemble_pred)
print(cm_ensemble)
print(f"False Positives (FP): {cm_ensemble[0, 1]}")
print(f"False Negatives (FN): {cm_ensemble[1, 0]}")

# ============================================
# 7. PERFORMANCE COMPARISON
# ============================================

print("\n" + "="*50)
print("FINAL RESULTS - MODEL COMPARISON")
print("="*50)

sorted_models = dict(sorted(models_performance.items(), key=lambda x: x[1], reverse=True))

print("\nRanking of models by accuracy:")
for rank, (model, accuracy) in enumerate(sorted_models.items(), 1):
    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
    status = "✓ (>85%)" if accuracy >= 0.85 else "✗ (<85%)"
    print(f"{rank}. {model:25s} {accuracy:.4f} ({accuracy*100:.2f}%)  {status} {medal}")

models = list(sorted_models.keys())
accuracies = list(sorted_models.values())
colors = ['#2ecc71' if acc >= 0.85 else '#e74c3c' for acc in accuracies]

plt.figure(figsize=(14, 7))
x_pos = np.arange(len(models))
bars = plt.bar(x_pos, accuracies, color=colors, edgecolor='black', linewidth=1.5)

plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('Detailed Comparison of 5 Classification Algorithms + Neural Network + Ensemble', 
          fontsize=14, fontweight='bold', pad=20)
plt.xticks(x_pos, models, rotation=45, ha='right')
plt.ylim(0.7, 1.0)

for i, (bar, acc, model) in enumerate(zip(bars, accuracies, models)):
    height = bar.get_height()
    emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else ""
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{acc:.4f}\n{emoji}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.axhline(y=0.85, color='red', linestyle='--', linewidth=2, label='Target 85%')
plt.legend(fontsize=10)

plt.tight_layout()


# ============================================
# 8. IMPORTANT FEATURES ANALYSIS (XGBoost)
# ============================================

print("\n" + "="*50)
print("ANALYSIS OF WORDS/PHRASES LEADING TO TOXICITY")
print("="*50)

feature_importance = xgb_model.feature_importances_
feature_names = vect.get_feature_names_out()

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nTop 20 words/phrases leading to toxic comments:")
print(importance_df.head(20))

plt.figure(figsize=(12, 8))
top_features = importance_df.head(20)
plt.barh(top_features['feature'], top_features['importance'], color='#FF6B6B')
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Words/Phrases', fontsize=12)
plt.title('Top 20 Words/Phrases Indicative of Toxicity (XGBoost)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()


# ============================================
# 9. SCORING SYSTEM (1-10 SCALE)
# ============================================

print("\n" + "="*50)
print("COMMENT SCORING SYSTEM (1-10 SCALE)")
print("="*50)

def get_toxicity_score_1_to_10(comment, model, vectorizer, importance_df, top_n=10):
  
    cleaned_comment = clean_text(comment)
    comment_vector = vectorizer.transform([cleaned_comment])
    
    prediction = model.predict(comment_vector)[0]
    probability = model.predict_proba(comment_vector)[0]
    
    toxicity_probability = probability[1]
    toxicity_score_10 = 1 + (toxicity_probability * 9)
    
    if toxicity_score_10 <= 2:
        severity = "NON-TOXIC"
        severity_color = "🟢"
    elif toxicity_score_10 <= 4:
        severity = "SLIGHTLY NEGATIVE"
        severity_color = "🟡"
    elif toxicity_score_10 <= 6:
        severity = "MODERATELY TOXIC"
        severity_color = "🟠"
    elif toxicity_score_10 <= 8:
        severity = "HIGHLY TOXIC"
        severity_color = "🔴"
    else:
        severity = "EXTREMELY TOXIC"
        severity_color = "⚫"
    
    comment_words = set(cleaned_comment.split())
    toxic_words_found = []
    
    for _, row in importance_df.head(50).iterrows():
        feature = row['feature']
        importance = row['importance']
        
        if ' ' in feature:
            if feature in cleaned_comment:
                toxic_words_found.append({
                    'word': feature,
                    'importance': importance,
                    'type': 'phrase'
                })
        else:
            if feature in comment_words:
                toxic_words_found.append({
                    'word': feature,
                    'importance': importance,
                    'type': 'word'
                })
    
    toxic_words_found = sorted(toxic_words_found, key=lambda x: x['importance'], reverse=True)[:top_n]
    
    return {
        'comment': comment,
        'prediction': 'TOXIC' if prediction == 1 else 'NON-TOXIC',
        'toxicity_score_10': toxicity_score_10,
        'toxicity_percentage': toxicity_probability * 100,
        'severity': severity,
        'severity_color': severity_color,
        'confidence': max(probability) * 100,
        'toxic_elements': toxic_words_found
    }

# ============================================
# 10. SCORING SYSTEM TESTS (1-10 SCALE)
# ============================================

test_comments = [
    "I love this website, it's so helpful!",
    "This is the worst product ever, total waste of money.",
    "The author of this article is brilliant!",
    "You are an fucking idiot and your opinion is stupid.",
    "This software is a scam, do not buy it.",
    "The customer service was excellent, very helpful and friendly.",
    "I hate this stupid video, it's garbage.",
    "Great work! Keep it up!",
    "This is absolutely disgusting and offensive.",
    "Thank you for the amazing content!"
]

print("\n=== DETAILED COMMENT ANALYSIS (1-10 SCALE) ===\n")

results = []
for comment in test_comments:
    result = get_toxicity_score_1_to_10(
        comment, 
        xgb_model, 
        vect, 
        importance_df, 
        top_n=5
    )
    results.append(result)
    
    print(f"Comment: {result['comment']}")
    print(f"   > {result['severity_color']} Toxicity Score: {result['toxicity_score_10']:.1f}/10")
    print(f"   > Severity Level: {result['severity']}")
    print(f"   > Prediction: {result['prediction']}")
    print(f"   > Confidence: {result['confidence']:.2f}%")
    
    if result['toxic_elements']:
        print(f"   > Toxic elements detected:")
        for element in result['toxic_elements']:
            print(f"      • {element['word']} ({element['type']}) - importance: {element['importance']:.4f}")
    else:
        print(f"   > No major toxic elements detected")
    print()

# ============================================
# 11. VISUALIZATION OF SCORES (1-10)
# ============================================

print("\n" + "="*50)
print("VISUALIZATION OF TOXICITY SCORES")
print("="*50)

scores = [r['toxicity_score_10'] for r in results]
comments_short = [c[:30] + '...' if len(c) > 30 else c for c in test_comments]

plt.figure(figsize=(14, 8))
colors_scores = ['#2ecc71' if s <= 2 else '#f1c40f' if s <= 4 else '#e67e22' if s <= 6 
                 else '#e74c3c' if s <= 8 else '#34495e' for s in scores]

bars = plt.barh(range(len(scores)), scores, color=colors_scores, edgecolor='black', linewidth=1.5)
plt.yticks(range(len(scores)), comments_short, fontsize=9)
plt.xlabel('Toxicity Score (1-10)', fontsize=12, fontweight='bold')
plt.title('Toxicity Scores for Test Comments (1-10 Scale)', fontsize=14, fontweight='bold')
plt.xlim(0, 11)

for i, (bar, score) in enumerate(zip(bars, scores)):
    width = bar.get_width()
    plt.text(width + 0.2, bar.get_y() + bar.get_height()/2.,
             f'{score:.1f}',
             ha='left', va='center', fontsize=9, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='1-2: Non-toxic'),
    Patch(facecolor='#f1c40f', label='3-4: Slightly negative'),
    Patch(facecolor='#e67e22', label='5-6: Moderately toxic'),
    Patch(facecolor='#e74c3c', label='7-8: Highly toxic'),
    Patch(facecolor='#34495e', label='9-10: Extremely toxic')
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()

# ============================================
# 12. CONFUSION MATRIX (XGBoost)
# ============================================

print("\n" + "="*50)
print("CONFUSION MATRIX (XGBoost OPTIMIZED)")
print("="*50)

cm = confusion_matrix(y_test, xgb_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Toxic', 'Toxic'],
            yticklabels=['Non-Toxic', 'Toxic'])
plt.title('Confusion Matrix - XGBoost OPTIMIZED', fontsize=14, fontweight='bold')
plt.ylabel('True Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.tight_layout()

print(f"\nFalse Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]}")

# ============================================
# 13. CONFUSION MATRIX (NEURAL NETWORK)
# ============================================

print("\n" + "="*50)
print("CONFUSION MATRIX (NEURAL NETWORK)")
print("="*50)

cm_nn = confusion_matrix(y_test, nn_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Non-Toxic', 'Toxic'],
            yticklabels=['Non-Toxic', 'Toxic'])
plt.title('Confusion Matrix - Neural Network', fontsize=14, fontweight='bold')
plt.ylabel('True Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.tight_layout()

# ============================================
# 14. ENSEMBLE MODEL - SCORING SYSTEM TEST
# ============================================

print("\n" + "="*50)
print("ENSEMBLE MODEL - SCORING SYSTEM TEST")
print("="*50)

print("\n=== ANALYSIS OF COMMENTS USING THE ENSEMBLE MODEL ===\n")

results_ensemble = []
for comment in test_comments:
    result = get_toxicity_score_1_to_10(
        comment, 
        ensemble_model,  
        vect, 
        importance_df, 
        top_n=5
    )
    results_ensemble.append(result)
    
    print(f"Comment: {result['comment']}")
    print(f"   > {result['severity_color']} Toxicity Score: {result['toxicity_score_10']:.1f}/10")
    print(f"   > Severity Level: {result['severity']}")
    print(f"   > Prediction: {result['prediction']}")
    print(f"   > Confidence: {result['confidence']:.2f}%")
    
    if result['toxic_elements']:
        print(f"   > Toxic elements detected:")
        for element in result['toxic_elements']:
            print(f"      • {element['word']} ({element['type']}) - importance: {element['importance']:.4f}")
    else:
        print(f"   > No major toxic elements detected")
    print()

# ============================================
# 15. VISUALIZATION - ENSEMBLE VS XGBOOST SCORES
# ============================================

print("\n" + "="*50)
print("SCORE COMPARISON: ENSEMBLE vs XGBoost")
print("="*50)

scores_ensemble = [r['toxicity_score_10'] for r in results_ensemble]
scores_xgb = [r['toxicity_score_10'] for r in results]

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(test_comments))
width = 0.35

bars1 = ax.barh(x - width/2, scores_xgb, width, label='XGBoost', color='#3498db', edgecolor='black')
bars2 = ax.barh(x + width/2, scores_ensemble, width, label='Ensemble (XGB+SVC+MLP)', color='#9b59b6', edgecolor='black')

ax.set_xlabel('Toxicity Score (1-10)', fontsize=12, fontweight='bold')
ax.set_title('Comparison: XGBoost vs Ensemble Model Toxicity Scores', fontsize=14, fontweight='bold')
ax.set_yticks(x)
ax.set_yticklabels(comments_short, fontsize=9)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(0, 11)

for bar, score in zip(bars1, scores_xgb):
    width_val = bar.get_width()
    ax.text(width_val + 0.1, bar.get_y() + bar.get_height()/2., f'{score:.1f}',
            ha='left', va='center', fontsize=8, color='#3498db')

for bar, score in zip(bars2, scores_ensemble):
    width_val = bar.get_width()
    ax.text(width_val + 0.1, bar.get_y() + bar.get_height()/2., f'{score:.1f}',
            ha='left', va='center', fontsize=8, color='#9b59b6')

plt.tight_layout()

# ============================================
# 16. CONFUSION MATRIX (ENSEMBLE MODEL)
# ============================================

print("\n" + "="*50)
print("CONFUSION MATRIX (ENSEMBLE MODEL)")
print("="*50)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_ensemble, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Non-Toxic', 'Toxic'],
            yticklabels=['Non-Toxic', 'Toxic'])
plt.title('Confusion Matrix - Ensemble Model (XGB+SVC+MLP)', fontsize=14, fontweight='bold')
plt.ylabel('True Class', fontsize=12)
plt.xlabel('Predicted Class', fontsize=12)
plt.tight_layout()

plt.show()

# ============================================
# 17. SUMMARY TABLE
# ============================================

print("\n" + "="*50)
print("TABLEAU RÉCAPITULATIF - TOUS LES MODÈLES")
print("="*50)

summary_data = []
for model_name, accuracy in sorted_models.items():
    summary_data.append({
        'Model': model_name,
        'Accuracy': f"{accuracy:.4f}",
        'Percentage': f"{accuracy*100:.2f}%",
        'Status': '✓ PASS' if accuracy >= 0.85 else '✗ FAIL'
    })

summary_df = pd.DataFrame(summary_data)
print("\n")
print(summary_df.to_string(index=False))
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class EmailClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        self.is_trained = False
    
    def train(self, emails, labels):
        """Train the classifier with email texts and labels."""
        self.pipeline.fit(emails, labels)
        self.is_trained = True
    
    def classify(self, email_text):
        """Classify a single email."""
        if not self.is_trained:
            raise ValueError("Classifier must be trained first")
        return self.pipeline.predict([email_text])[0]
    
    def classify_batch(self, emails):
        """Classify multiple emails."""
        if not self.is_trained:
            raise ValueError("Classifier must be trained first")
        return self.pipeline.predict(emails)

# Example usage
if __name__ == "__main__":
    classifier = EmailClassifier()
    
    # Sample training data
    sample_emails = [
        "Limited time offer! Buy now and save 50%",
        "Meeting scheduled for tomorrow at 2 PM",
        "Your password has expired, please reset it"
    ]
    sample_labels = ["spam", "work", "security"]
    
    classifier.train(sample_emails, sample_labels)
    
    # Test classification
    test_email = "Get rich quick scheme!"
    result = classifier.classify(test_email)
    print(f"Classified as: {result}")
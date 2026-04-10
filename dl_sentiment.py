# dl_sentiment.py — Deep Learning Sentiment Analysis (Transformer)
# Uses a pre-trained transformer model for financial sentiment analysis.
# Falls back to VADER if transformer is unavailable.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DeepSentimentAnalyzer:
    """
    Transformer-based sentiment analysis engine.
    
    Attempts to load ProsusAI/finbert (financial domain) first,
    then falls back to distilbert-base-uncased-finetuned-sst-2-english,
    and finally to VADER as last resort.
    """

    def __init__(self):
        self.model_name = None
        self.pipeline = None
        self.vader = None
        self._load_model()

    def _load_model(self):
        """Try loading transformer models in order of preference."""
        # Try FinBERT first (best for financial text)
        models_to_try = [
            ("ProsusAI/finbert", "FinBERT (Financial Domain)"),
            ("distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT (General)"),
        ]

        for model_id, display_name in models_to_try:
            try:
                from transformers import pipeline as hf_pipeline
                self.pipeline = hf_pipeline(
                    "sentiment-analysis",
                    model=model_id,
                    tokenizer=model_id,
                    top_k=None,  # Return all class probabilities
                    device=-1,   # Force CPU
                )
                self.model_name = display_name
                return
            except Exception:
                continue

        # Fallback to VADER
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
            self.model_name = "VADER (Lexicon Fallback)"
        except Exception:
            self.model_name = "None (All models failed)"

    def analyze(self, text: str) -> dict:
        """
        Analyze a single headline/text.
        
        Returns:
            {
                'text': str,
                'label': 'Positive' | 'Negative' | 'Neutral',
                'score': float (-1 to 1),
                'probabilities': {'positive': float, 'negative': float, 'neutral': float},
                'model': str
            }
        """
        if self.pipeline is not None:
            return self._analyze_transformer(text)
        elif self.vader is not None:
            return self._analyze_vader(text)
        else:
            return {
                'text': text,
                'label': 'Neutral',
                'score': 0.0,
                'probabilities': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'model': self.model_name,
            }

    def _analyze_transformer(self, text: str) -> dict:
        """Analyze using the transformer pipeline."""
        try:
            results = self.pipeline(text[:512])  # Truncate to max token length

            # results is a list of lists: [[{'label': ..., 'score': ...}, ...]]
            scores = results[0] if isinstance(results[0], list) else results
            
            probs = {}
            for item in scores:
                label = item['label'].lower()
                # Normalize label names across different models
                if label in ('positive', 'pos'):
                    probs['positive'] = item['score']
                elif label in ('negative', 'neg'):
                    probs['negative'] = item['score']
                elif label in ('neutral', 'neu'):
                    probs['neutral'] = item['score']
                # For binary models (distilbert)  
                elif label == 'label_1' or label == 'label_2':
                    probs['positive'] = item['score']
                elif label == 'label_0':
                    probs['negative'] = item['score']

            # Ensure all keys exist
            probs.setdefault('positive', 0.0)
            probs.setdefault('negative', 0.0)
            probs.setdefault('neutral', 0.0)

            # Determine dominant label
            dominant = max(probs, key=probs.get)
            label_map = {'positive': 'Positive', 'negative': 'Negative', 'neutral': 'Neutral'}

            # Composite score: positive - negative (range: -1 to 1)
            composite = probs['positive'] - probs['negative']

            return {
                'text': text,
                'label': label_map.get(dominant, 'Neutral'),
                'score': round(composite, 4),
                'probabilities': {k: round(v, 4) for k, v in probs.items()},
                'model': self.model_name,
            }
        except Exception as e:
            # Fallback to neutral on error
            return {
                'text': text,
                'label': 'Neutral',
                'score': 0.0,
                'probabilities': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34},
                'model': f"{self.model_name} (error: {str(e)[:50]})",
            }

    def _analyze_vader(self, text: str) -> dict:
        """Analyze using VADER as fallback."""
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']

        if compound > 0.15:
            label = 'Positive'
        elif compound < -0.15:
            label = 'Negative'
        else:
            label = 'Neutral'

        return {
            'text': text,
            'label': label,
            'score': round(compound, 4),
            'probabilities': {
                'positive': round(scores['pos'], 4),
                'negative': round(scores['neg'], 4),
                'neutral': round(scores['neu'], 4),
            },
            'model': self.model_name,
        }

    def analyze_batch(self, headlines: list) -> list:
        """Analyze multiple headlines."""
        return [self.analyze(h) for h in headlines]

    def get_market_sentiment(self, headlines: list) -> dict:
        """
        Aggregate sentiment across all headlines.
        
        Returns:
            {
                'avg_score': float,
                'label': str,
                'headline_count': int,
                'positive_count': int,
                'negative_count': int,
                'neutral_count': int,
                'results': list[dict],
                'model': str,
            }
        """
        if not headlines:
            return {
                'avg_score': 0.0,
                'label': 'Neutral',
                'headline_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'results': [],
                'model': self.model_name,
            }

        results = self.analyze_batch(headlines)

        scores = [r['score'] for r in results]
        avg_score = sum(scores) / len(scores)

        pos_count = sum(1 for r in results if r['label'] == 'Positive')
        neg_count = sum(1 for r in results if r['label'] == 'Negative')
        neu_count = sum(1 for r in results if r['label'] == 'Neutral')

        if avg_score > 0.15:
            label = 'Bullish 📈'
        elif avg_score < -0.15:
            label = 'Bearish 📉'
        else:
            label = 'Neutral ⚖️'

        return {
            'avg_score': round(avg_score, 4),
            'label': label,
            'headline_count': len(headlines),
            'positive_count': pos_count,
            'negative_count': neg_count,
            'neutral_count': neu_count,
            'results': results,
            'model': self.model_name,
        }


# Singleton instance for reuse
_analyzer_instance = None


def get_analyzer() -> DeepSentimentAnalyzer:
    """Get or create the singleton analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = DeepSentimentAnalyzer()
    return _analyzer_instance
